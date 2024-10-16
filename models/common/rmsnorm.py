# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule


TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


class RMSNorm(LightweightModule):
    """
    RMSNorm supporting replication over a MeshDevice and sharding within devices.

    This class implements a Root Mean Square Normalization (RMSNorm) that can be
    distributed across multiple devices and cores. If the `device` parameter is a
    MeshDevice, the weights and computations are replicated across all devices in
    the mesh. Expects an interleaved input tensor, can optionally output a sharded tensor.

    Args:
        device: The device or MeshDevice on which to perform the computations.
        state_dict: The state dictionary containing the model parameters.
        dim: Input dimension (e.g. model hidden dimension size).
        layer_num: The layer number to determine the weight key in the state dictionary.
        weight_key: The key for retrieving the weight from the state dictionary.
        weight_cache_path: Optional path for caching the tilized weights.
        weight_memory_config: Configuration for the weight memory, default is DRAM_MEMORY_CONFIG.
        weight_dtype: The data type for the tensors, bfp8_b hits >0.999 PCC in the models we tested.
        model_config: Optional configuration dictionary for the model.
        eps (float): Small value to avoid division by zero in normalization, default is 1e-05.

    If model_config is provided, it must specify SHARDED_NORM_INPUT_MEMCFG, SHARDED_NORM_PRGM_CFG
    and SHARDED_NORM_OUTPUT_MEMCFG. If not provided, default configurations will be generated.
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat8_b,
        model_config=None,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.eps = eps

        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
        else:
            if layer_num is None:
                weight_name = f"{weight_key}.weight"
            else:
                weight_name = f"layers.{layer_num}.{weight_key}.weight"

        torch_weight = (
            state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )

        print(f"{torch_weight.shape=}")

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        self.is_distributed_norm = model_config and model_config["IS_DISTRIBUTED_NORM"]

        cache_name = None if weight_cache_path is None else weight_cache_path / weight_name

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            # cache_file_name=cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device)
            if is_mesh_device
            else None,  # TODO: test if None needed for single device
        )

        self.weight_sharded = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            # cache_file_name=cache_name,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2)
            if is_mesh_device
            else None,  # TODO: test if None needed for single device
        )

        if model_config:
            self.sharded_input_config = model_config["SHARDED_NORM_INPUT_MEMCFG"]
            self.sharded_program_config = model_config["SHARDED_NORM_PRGM_CFG"]
            self.sharded_output_config = model_config["SHARDED_NORM_OUTPUT_MEMCFG"]
            self.compute_kernel_config_hifi2 = model_config["COMPUTE_KERNEL_CONFIG_HIFI2"]
        else:
            assert (
                dim % SHARD_HEIGHT == 0
            ), f"Input dimension dim ({dim}) must be a multiple of SHARD_HEIGHT ({SHARD_HEIGHT})"
            shard_width_hidden_dim_across_32_cores = dim // SHARD_HEIGHT
            core_grid = ttnn.CoreGrid(x=8, y=SHARD_HEIGHT // 8)
            self.sharded_input_config = ttnn.create_sharded_memory_config(
                shape=(SHARD_HEIGHT, shard_width_hidden_dim_across_32_cores),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.sharded_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[core_grid.x, core_grid.y],
                subblock_w=shard_width_hidden_dim_across_32_cores // TILE,
                block_h=SHARD_HEIGHT // TILE,
                block_w=shard_width_hidden_dim_across_32_cores // TILE,
                inplace=False,
            )
            self.sharded_output_config = self.sharded_input_config
            self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def forward(self, x: ttnn.Tensor, in_sharded=False, out_sharded=False) -> ttnn.Tensor:
        # If input is sharded do sharded RMSNorm and optionally return sharded output
        if in_sharded:
            x = ttnn.rms_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                program_config=self.sharded_program_config,
                memory_config=self.sharded_output_config,
            )
            if out_sharded:
                return x
            x_interleaved = ttnn.sharded_to_interleaved(x)
            x.deallocate(True)
            return x_interleaved
        else:  # Interleaved rmsnorm does not need program or memory configs
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"
            if self.is_distributed_norm:
                return self._distributed_rmsnorm(x)
            else:
                return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)

    def _distributed_rmsnorm(self, inp):
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(
            inp, compute_kernel_config=self.compute_kernel_config_hifi2, dtype=ttnn.bfloat16
        )

        # AllGather stats
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=self.eps,
            weight=self.weight_sharded,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        tt_stats.deallocate(True)

        return tt_out
