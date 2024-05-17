# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh
from models.demos.t3000.mixtral8x7b.tt.create_prg_config import create_matmul_program_config


class TtMixtralMLP(torch.nn.Module):
    def __init__(self, device_mesh, state_dict, args, layer_num, dtypes):
        super().__init__()

        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.dtypes = dtypes
        self.model_args = args
        self.model_config = args.get_model_config()

        base_name = lambda expert_num: f"layers.{layer_num}.feed_forward.experts.{expert_num}"
        torch_weight = lambda name: torch.concat(
            [
                self.state_dict[f"{base_name(expert_num)}.{name}.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0)
                for expert_num in range(8)
            ],
            dim=0,
        )
        cache_name = lambda name: args.weight_cache_path(dtypes[name]) / (
            f"layers.{layer_num}.feed_forward_multidevice_unsqueezed.experts.{name}"
        )
        as_tensor = lambda name: ttnn.as_tensor(
            torch_weight(name),
            dtype=dtypes[name],
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=self.model_config["MLP_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("w1")
        self.w1 = ttnn.to_device(self.w1, device_mesh)
        self.w2 = as_tensor("w2")
        self.w2 = ttnn.to_device(self.w2, device_mesh)
        self.w3 = as_tensor("w3")
        self.w3 = ttnn.to_device(self.w3, device_mesh)

        # (1, 1, 32, 4096/32)
        # (1, 1, 4096, 14336)
        self.x_mem = ttnn.create_sharded_memory_config(
            shape=(32, int(4096 / 32)),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.w1_prg_cfg = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(6, 7),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=ttnn.experimental.tensor.FusibleActivation.SILU,
            mcast_in0=True,
        )
        self.w3_prg_cfg = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(6, 7),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=11,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.w2_prg_cfg = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(6, 7),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=11,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        x = ttnn.to_memory_config(x, self.x_mem)
        prg_cfg = create_matmul_program_config(
            input_tensor_a=x,
            input_tensor_b=self.w1,
            core_grid=self.model_args.max_grid_size,
            use_1d_systolic_array=True,
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
            activation="silu",
        )
        print(prg_cfg)

        w1_out = ttnn.experimental.operations.primary.matmul_1d(
            x,
            self.w1,
            # activation="silu",
            # core_grid=self.model_args.max_grid_size,
            # use_1d_systolic_array=True,
            program_config=self.w1_prg_cfg,
            output_mem_config=self.model_config["FF1_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
        )
        exit()
        w3_out = ttnn.experimental.operations.primary.matmul_1d(
            x,
            self.w3,
            # core_grid=self.model_args.max_grid_size,
            # use_1d_systolic_array=True,
            program_config=self.w3_prg_cfg,
            output_mem_config=self.model_config["FF3_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
        )
        w2_in = ttnn.mul(w1_out, w3_out)
        prg_cfg = create_matmul_program_config(
            input_tensor_a=w2_in,
            input_tensor_b=self.w2,
            core_grid=self.model_args.max_grid_size,
            use_1d_systolic_array=True,
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
        )
        print(prg_cfg)
        exit()
        w2_out = ttnn.experimental.operations.primary.matmul_1d(
            w2_in,
            self.w2,
            # core_grid=self.model_args.max_grid_size,
            # use_1d_systolic_array=True,
            output_mem_config=self.model_config["FF2_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
            output_dtype=ttnn.bfloat16,
        )

        return w2_out
