# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib
from loguru import logger


class MistralMLP:
    def __init__(self, input_shape, parameters, grid=ttnn.CoreGrid(8, 8)):
        """Set up the configs ahead of time as this takes 30-50us"""
        self.w1 = parameters.w1.weight
        self.w2 = parameters.w2.weight
        self.w3 = parameters.w3.weight

        rows = input_shape.with_tile_padding()[-2]
        self.ff_silu = matmul_1d_config(
            m=rows,
            k=self.w1.shape.with_tile_padding()[-2],
            n=self.w1.shape.with_tile_padding()[-1],
            grid=grid,
            act=ttnn.ttl.tensor.FusibleActivation.SILU,
        )
        self.ff_none = matmul_1d_config(
            m=rows,
            k=self.w3.shape.with_tile_padding()[-2],
            n=self.w3.shape.with_tile_padding()[-1],
            grid=grid,
            act=None,
        )
        self.ff2 = matmul_1d_config(
            m=rows,
            k=self.w2.shape.with_tile_padding()[-2],
            n=self.w2.shape.with_tile_padding()[-1],
            grid=grid,
            act=None,
        )
        shard_shape = ttnn.ShardShape(rows, self.w1.shape.with_tile_padding()[-1] // grid.num_cores)
        self.shard = ttnn.create_sharded_memory_config(grid, shard_shape, ttnn.ShardStrategy.WIDTH)
        self.interleave = ttnn.ttl.tensor.MemoryConfig(
            ttnn.ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.ttl.tensor.BufferType.L1
        )

    def __call__(self, x):
        matmul_1d = ttnn.ttl.operations.primary.matmul_1d
        w1_out = matmul_1d(x, self.w1, program_config=self.ff_silu, output_mem_config=self.shard)
        w3_out = matmul_1d(x, self.w3, program_config=self.ff_none, output_mem_config=self.shard)
        w2_in = ttnn.mul(w1_out, w3_out, memory_config=self.shard)
        w2_out = matmul_1d(w2_in, self.w2, program_config=self.ff2, output_mem_config=self.interleave)
        return w2_out


def feed_forward(config, x, parameters):
    """Lower performance but maches functional paradigm"""
    mlp = MistralMLP(input_shape=x.shape, parameters=parameters)
    return mlp(x)


def matmul_1d_config(m, k, n, grid=ttnn.CoreGrid(8, 8), act=None):
    tile_width = 32
    tile_height = 32

    per_core_m = m // tile_height
    per_core_k = k // tile_width // grid.num_cores
    per_core_n = n // tile_width // grid.num_cores

    # find the largest value between 1 and 8 that is a factor of per_core_n
    # e.g. if per_core_n is 14, then out_subblock_w = 7
    out_subblock_w = max([i for i in range(1, 9) if per_core_n % i == 0])

    # find the largest value that is a factor of per_core_m such that
    # out_subblock_w * out_subblock_h <= 8
    out_subblock_h = max([i for i in range(1, 9) if per_core_m % i == 0 and i * out_subblock_w <= 8])

    return ttnn.ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=tt_lib.tensor.CoreCoord(grid.x, grid.y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )
