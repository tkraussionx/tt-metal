# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtMistralMLP(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.grid = ttnn.CoreGrid(x=8, y=8)

        base_name = f"layers.{layer_num}.feed_forward"
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{base_name}.{name}.weight"], -2, -1)
        cache_name = lambda name: weight_cache_path / (base_name + f".feed_forward.{name}")
        as_tensor = lambda name: ttnn.as_tensor(
            torch_weight(name),
            dtype=dtype,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("w1")
        self.w2 = as_tensor("w2")
        self.w3 = as_tensor("w3")

        self.kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
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
        shard = ttnn.create_sharded_memory_config(ff1_output_shape, ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.WIDTH)

        w1_out = ttnn.linear(
            x,
            self.w1,
            activation="silu",
            core_grid=self.grid,
            use_1d_systolic_array=True,
            memory_config=shard,  # , compute_kernel_config=self.kernel_config,
        )
        w3_out = ttnn.linear(
            x, self.w3, core_grid=self.grid, use_1d_systolic_array=True, memory_config=shard
        )  # , compute_kernel_config=self.kernel_config)
        w2_in = ttnn.mul(w1_out, w3_out, memory_config=shard)
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            core_grid=self.grid,
            use_1d_systolic_array=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # , compute_kernel_config=self.kernel_config
        )

        return w2_out
