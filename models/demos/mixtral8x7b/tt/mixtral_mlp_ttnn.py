# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import torch
import ttnn


class TtMixtralMLP(torch.nn.Module):
    def __init__(self, device, state_dict, args, layer_num, expert_num, dtype):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.model_config = args
        self.grid = ttnn.CoreGrid(y=7, x=8)
        self.dtype = dtype

        # Convert block sparse moe representation of e.g. (114688, 4096) into separate tensors for each expert
        base_name = f"layers.{layer_num}.block_sparse_moe"
        torch_weight = (
            lambda name, shape: self.state_dict[f"{base_name}.{name}"]
            .view(8, shape[1], shape[0])[expert_num]
            .permute(1, 0)
        )
        cache_name = lambda name: self.model_config.weight_cache_path(dtype) / (f"{base_name}.{expert_num}.{name}")
        as_tensor = lambda name, shape: ttnn.as_tensor(
            torch_weight(name, shape),
            dtype=dtype,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("w1", (4096, 14336))
        self.w2 = as_tensor("w2", (14336, 4096))
        self.w3 = as_tensor("w3", (4096, 14336))

        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        ff1_output_shape = ttnn.Shape([x.shape.with_tile_padding()[-2], self.w1.shape.with_tile_padding()[-1]])
        # shard = ttnn.create_sharded_memory_config(ff1_output_shape, self.grid, ttnn.ShardStrategy.WIDTH)

        w1_out = ttnn.linear(
            x,
            self.w1,
            activation="silu",
            core_grid=self.grid,
            use_1d_systolic_array=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # , memory_config=shard
            compute_kernel_config=self.compute_kernel,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            core_grid=self.grid,
            use_1d_systolic_array=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # , memory_config=shard
            compute_kernel_config=self.compute_kernel,
        )
        w2_in = ttnn.mul(w1_out, w3_out)  # , memory_config=shard)
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            core_grid=self.grid,
            use_1d_systolic_array=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel,
        )

        return w2_out
