# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtMistralMLP(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        model_config,
        grid=ttnn.CoreGrid(8, 8),
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.model_config = model_config
        self.grid = grid

        torch_weight = lambda name: torch.transpose(self.state_dict[f"{base_address}{name}.weight"], -2, -1)

        self.w1 = ttnn.from_torch(
            torch_weight("w1"),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=self.model_config["FF1_MM_WEIGHTS_DTYPE"],
        )
        self.w2 = ttnn.from_torch(
            torch_weight("w2"),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=self.model_config["FF2_MM_WEIGHTS_DTYPE"],
        )
        self.w3 = ttnn.from_torch(
            torch_weight("w3"),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=self.model_config["FF3_MM_WEIGHTS_DTYPE"],
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        ff1_output_shape = ttnn.Shape([x.shape.with_tile_padding()[-2], self.w1.shape.with_tile_padding()[-1]])
        shard = ttnn.create_sharded_memory_config(ff1_output_shape, self.grid, ttnn.ShardStrategy.WIDTH)

        w1_out = ttnn.linear(
            x, self.w1, activation="silu", core_grid=self.grid, use_1d_systolic_array=True, memory_config=shard
        )
        w3_out = ttnn.linear(x, self.w3, core_grid=self.grid, use_1d_systolic_array=True, memory_config=shard)
        w2_in = ttnn.mul(w1_out, w3_out, memory_config=shard)
        w2_out = ttnn.linear(
            w2_in, self.w2, core_grid=self.grid, use_1d_systolic_array=True, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        return w2_out
