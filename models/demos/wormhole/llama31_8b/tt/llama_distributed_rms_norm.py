# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
import tt_lib as ttl

TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


class tt_distributed_rmsnorm(torch.nn.Module):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat8_b,
        model_config=None,
        is_sharded=False,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.eps = eps
        self.is_sharded = is_sharded
        self.model_config = model_config
        self.device = device

        if layer_num is None:
            weight_name = f"{weight_key}.weight"
        else:
            weight_name = f"layers.{layer_num}.{weight_key}.weight"

        torch_weight = state_dict[weight_name].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        cache_name = None if weight_cache_path is None else weight_cache_path / weight_name

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )
        print("self.weight memory config:", self.weight.memory_config())

    def forward(self, x: ttnn.Tensor, out_sharded=False) -> ttnn.Tensor:
        print("x shape:", x.shape)
        tt_stats = ttnn.rms_norm_pre_all_gather(
            x, compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"], dtype=ttnn.bfloat16
        )
        print("pre all gather done")
        print("tt_stats shape: ", tt_stats.shape)
        tt_stats = ttnn.to_device(tt_stats, self.device)
        tt_stats = ttnn.line_all_gather(
            tt_stats,
            dim=2,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print("line all gather done")
        print("tt_stats shape: ", tt_stats.shape)  # tt_stats shape:  ttnn.Shape([1, 32, 4096])
        print("x shape:", x.shape)  # x shape: ttnn.Shape([1, 32, 2048])
        print("self.weight shape:", self.weight.shape)  # self.weight shape: ttnn.Shape([1, 32, 4096])
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            tt_stats,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )  # RuntimeError: stats.get_legacy_shape()[2] == a.get_legacy_shape()[2]
        tt_stats.deallocate(True)

        return tt_out
