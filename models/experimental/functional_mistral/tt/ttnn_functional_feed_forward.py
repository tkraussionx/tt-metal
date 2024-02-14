# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib


def matmul_1d(a, b, memory_config, core_grid):
    out = tt_lib.operations.primary.matmul_1d(a, b, output_mem_config=memory_config)  # , core_grid=core_grid)
    return out


def feed_forward(config, x: ttnn.Tensor, *, parameters):
    """
    silu_out = ttnn.silu(x @ parameters.w1.weight)
    x = silu_out * (x @ parameters.w3.weight)
    return x @ parameters.w2.weight
    """
    # move to tt_lib until ttnn supports matmul_1d
    x = x.value
    w1 = parameters.w1.weight.value
    w2 = parameters.w2.weight.value
    w3 = parameters.w3.weight.value

    w1_out = ff1(x, w1, silu=True)
    w3_out = ff1(x, w3, silu=False)

    w2_in = tt_lib.tensor.mul(w1_out, w3_out, output_mem_config=ttnn.L1_MEMORY_CONFIG)
    w2_out = ff2(w2_in, w2)

    # back to ttnn
    result = ttnn.Tensor(w2_out)
    return result


def ff1(x, w, silu):
    mem = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1)
    prog = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=7,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=14,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fuse_batch=True,
        fused_activation=tt_lib.tensor.FusibleActivation.SILU if silu else None,
        mcast_in0=True,
    )
    dtype = tt_lib.tensor.DataType.BFLOAT16

    out = tt_lib.operations.primary.matmul_1d(x, w, program_config=prog, output_mem_config=mem, output_dtype=dtype)
    return out


def ff2(x, w):
    mem = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1)
    prog = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=7,  # 14336 / TILE_WIDTH=32 / Grid_Size
        out_subblock_h=1,
        out_subblock_w=2,  # 8#2, # 4096 / TILE_WIDTH=32 / Grid_Size
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=2,  # 128,#2,
        fuse_batch=True,  # True,
        fused_activation=None,
        mcast_in0=True,
    )
    dtype = tt_lib.tensor.DataType.BFLOAT16

    out = tt_lib.operations.primary.matmul_1d(x, w, program_config=prog, output_mem_config=mem, output_dtype=dtype)
    return out
