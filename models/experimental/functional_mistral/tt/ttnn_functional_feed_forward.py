# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib


def feed_forward(config, x: ttnn.Tensor, *, parameters):
    """
    silu_out = ttnn.silu(x @ parameters.w1.weight)
    x = silu_out * (x @ parameters.w3.weight)
    return x @ parameters.w2.weight
    """
    w1 = parameters.w1.weight
    w2 = parameters.w2.weight
    w3 = parameters.w3.weight
    dim = w1.shape[-2]
    expanded_dim = w1.shape[-1]
    mul_memory_config = ttnn.create_sharded_memory_config(
        (8, 8), (dim // 8, expanded_dim // 8), ttnn.ShardStrategy.WIDTH
    )

    w1_out = ff1(x, w1, silu=True)
    w3_out = ff1(x, w3, silu=False)
    w2_in = ttnn.mul(w1_out, w3_out, memory_config=mul_memory_config)
    w2_out = ff2(w2_in, w2)

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


@ttnn.register_operation(name="ff1", validate_input_tensors=validate_input_tensors)
def ff1(x, w, silu):
    mem = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.WIDTH_SHARDED, tt_lib.tensor.BufferType.L1)
    prog = tt_lib.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,  # K = 4096 / TILE_WIDTH=32 / Grid_Size
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=7,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
        per_core_M=1,  # M = 4096 / TILE_HEIGHT=32 / 32
        per_core_N=7,  # N = 14336 / TILE_WIDTH=32 / Grid_Size=64
        fuse_batch=True,
        fused_activation=tt_lib.tensor.FusibleActivation.SILU if silu else None,
        mcast_in0=True,
    )
    dtype = tt_lib.tensor.DataType.BFLOAT16

    ttl_x = x.value
    ttl_w = w.value
    ttl_o = tt_lib.operations.primary.matmul_1d(
        ttl_x, ttl_w, program_config=prog, output_mem_config=mem, output_dtype=dtype
    )
    output_tensor = ttnn.Tensor(ttl_o)
    return output_tensor


@ttnn.register_operation(name="ff2", validate_input_tensors=validate_input_tensors)
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

    ttl_x = x.value
    ttl_w = w.value
    ttl_o = tt_lib.operations.primary.matmul_1d(
        ttl_x, ttl_w, program_config=prog, output_mem_config=mem, output_dtype=dtype
    )
    output_tensor = ttnn.Tensor(ttl_o)
    return output_tensor
