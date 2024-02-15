# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib
from loguru import logger


def feed_forward(config, x: ttnn.Tensor, *, parameters):
    """
    silu_out = ttnn.silu(x @ parameters.w1.weight)
    x = silu_out * (x @ parameters.w3.weight)
    return x @ parameters.w2.weight
    """
    w1 = parameters.w1.weight
    w2 = parameters.w2.weight
    w3 = parameters.w3.weight
    rows = x.shape.padded()[-2]
    expanded_dim = w1.shape.padded()[-1]
    memory_config = ttnn.create_sharded_memory_config((8, 8), (rows, expanded_dim // (8 * 8)), ttnn.ShardStrategy.WIDTH)
    silu = tt_lib.tensor.FusibleActivation.SILU

    w1_out = matmul_1d(x, w1, act=silu, memory_config=memory_config)
    w3_out = matmul_1d(x, w3, act=None, memory_config=memory_config)
    w2_in = ttnn.mul(w1_out, w3_out, memory_config=memory_config)
    w2_out = matmul_1d(w2_in, w2)

    return w2_out


def validate_input_tensors(operation_name, x: ttnn.Tensor, w: ttnn.Tensor, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        x,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )
    ttnn.validate_input_tensor(
        operation_name,
        w,
        ranks=(4,),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.TILE_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def matmul_1d_config(m, k, n, grid=(8, 8), act=None):
    grid_size = grid[0] * grid[1]
    tile_width = 32
    tile_height = 32

    per_core_m = m // tile_height
    per_core_k = k // tile_width // grid_size
    per_core_n = n // tile_width // grid_size

    # find the largest value between 1 and 8 that is a factor of per_core_n
    # e.g. if per_core_n is 14, then out_subblock_w = 7
    out_subblock_w = max([i for i in range(1, 9) if per_core_n % i == 0])

    # find the largest value that is a factor of per_core_m such that
    # out_subblock_w * out_subblock_h <= 8
    out_subblock_h = max([i for i in range(1, 9) if per_core_m % i == 0 and i * out_subblock_w <= 8])

    # logger.debug(f"x={x.shape}, w={w.shape}, grid={grid}: per_core_m={per_core_m}, per_core_k={per_core_k}, per_core_n={per_core_n}, out_subblock_w={out_subblock_w}, out_subblock_h={out_subblock_h}")

    return tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


@ttnn.register_operation(name="matmul_1d", validate_input_tensors=validate_input_tensors)
def matmul_1d(
    x,
    w,
    act=None,
    grid=(8, 8),
    memory_config=tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1),
    prog=None,
    output_dtype=None,
):
    if prog is None:
        m = x.shape.padded()[-2]
        k = x.shape.padded()[-1]
        n = w.shape.padded()[-1]
        prog = matmul_1d_config(m, k, n, grid=grid, act=act)
    if output_dtype is None:
        output_dtype = x.value.dtype()

    ttl_x = x.value
    ttl_w = w.value
    ttl_o = tt_lib.operations.primary.matmul_1d(
        ttl_x, ttl_w, program_config=prog, output_mem_config=memory_config, output_dtype=output_dtype
    )
    output_tensor = ttnn.Tensor(ttl_o)
    return output_tensor
