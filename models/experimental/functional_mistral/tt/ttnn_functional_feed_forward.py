# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class MistralMLP:
    def __init__(self, input_shape, parameters, grid=ttnn.CoreGrid(8, 8)):
        """Set up the configs ahead of time as this adds around 1ms of overhead per call."""
        self.w1 = parameters.w1.weight
        self.w2 = parameters.w2.weight
        self.w3 = parameters.w3.weight
        self.grid = grid

        shard_shape = ttnn.ShardShape(
            input_shape.with_tile_padding()[-2], self.w1.shape.with_tile_padding()[-1] // grid.num_cores
        )
        self.shard = ttnn.create_sharded_memory_config(grid, shard_shape, ttnn.ShardStrategy.WIDTH)

    def __call__(self, x):
        w1_out = ttnn.linear(
            x, self.w1, activation="silu", core_grid=self.grid, use_1d_systolic_array=True, memory_config=self.shard
        )
        w3_out = ttnn.linear(x, self.w3, core_grid=self.grid, use_1d_systolic_array=True, memory_config=self.shard)
        w2_in = ttnn.mul(w1_out, w3_out, memory_config=self.shard)
        w2_out = ttnn.linear(
            w2_in, self.w2, core_grid=self.grid, use_1d_systolic_array=True, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        return w2_out


def feed_forward(config, x, parameters, grid=ttnn.CoreGrid(8, 8)):
    """Lower performance but maches functional paradigm"""
    w1 = parameters.w1.weight
    w2 = parameters.w2.weight
    w3 = parameters.w3.weight

    shard_shape = ttnn.ShardShape(x.shape.with_tile_padding()[-2], w1.shape.with_tile_padding()[-1] // grid.num_cores)
    shard = ttnn.create_sharded_memory_config(grid, shard_shape, ttnn.ShardStrategy.WIDTH)

    w1_out = ttnn.linear(x, w1, activation="silu", core_grid=grid, use_1d_systolic_array=True, memory_config=shard)
    w3_out = ttnn.linear(x, w3, core_grid=grid, use_1d_systolic_array=True, memory_config=shard)
    w2_in = ttnn.mul(w1_out, w3_out, memory_config=shard)
    w2_out = ttnn.linear(w2_in, w2, core_grid=grid, use_1d_systolic_array=True, memory_config=ttnn.L1_MEMORY_CONFIG)

    return w2_out
