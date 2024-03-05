# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import torch
import ttnn


class TtMixtralMLP(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        args,
        layer_num,
        expert_num,
        grid=ttnn.CoreGrid(8, 8),
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.model_config = args
        self.grid = grid

        # base_name = f"layers.{layer_num}.feed_forward"
        # torch_weight = lambda name: torch.transpose(self.state_dict[f"{base_name}.{name}.weight"], -2, -1)
        # cache_name = lambda name: Path(model_config["DEFAULT_WEIGHT_PATH"]) / (base_name + f".feed_forward.{name}")
        # as_tensor = lambda name, dtype_name: ttnn.as_tensor(
        #     torch_weight(name),
        #     dtype=self.model_config[dtype_name],
        #     device=self.device,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     cache_file_name=cache_name(name),
        # )

        # self.w1 = as_tensor("w1", "FF1_MM_WEIGHTS_DTYPE")
        # self.w2 = as_tensor("w2", "FF2_MM_WEIGHTS_DTYPE")
        # self.w3 = as_tensor("w3", "FF3_MM_WEIGHTS_DTYPE")
        self.w1 = ttnn.from_torch(
            self.state_dict[f"experts.{expert_num}.w1.weight"].permute(1, 0),
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )
        self.w2 = ttnn.from_torch(
            self.state_dict[f"experts.{expert_num}.w2.weight"].permute(1, 0),
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
        )
        self.w3 = ttnn.from_torch(
            self.state_dict[f"experts.{expert_num}.w3.weight"].permute(1, 0),
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
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
            x, self.w1, activation="silu", core_grid=self.grid, use_1d_systolic_array=True  # , memory_config=shard
        )
        w3_out = ttnn.linear(x, self.w3, core_grid=self.grid, use_1d_systolic_array=True)  # , memory_config=shard
        w2_in = ttnn.mul(w1_out, w3_out)  # , memory_config=shard)
        w2_out = ttnn.linear(
            w2_in, self.w2, core_grid=self.grid, use_1d_systolic_array=True, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        return w2_out
