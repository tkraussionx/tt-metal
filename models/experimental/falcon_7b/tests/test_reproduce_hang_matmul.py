# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import time

import ttnn
from models.utility_functions import torch2tt_tensor
import torch


# Used to reproduce issue #8665 with matmul 2D (Falcon 7b matmuls)
# ------- SPECS per core -------
# input: 128, 576
# output: 128, 2304 (ie. 4 x 72 tiles)
INPUT_PER_CORE_HEIGHT = 128
INPUT_PER_CORE_WIDTH = 576
OUTPUT_PER_CORE_WIDTH = 2304  # (ie. dense_h_to_4h)
BLACKHOLE_GRID_Y = 9
BLACKHOLE_GRID_X = 14
GRID_Y = BLACKHOLE_GRID_Y
GRID_X = BLACKHOLE_GRID_X


@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (
        (
            INPUT_PER_CORE_HEIGHT * GRID_Y,
            INPUT_PER_CORE_WIDTH * GRID_X,
            OUTPUT_PER_CORE_WIDTH * GRID_X,
            INPUT_PER_CORE_HEIGHT // 32,
            OUTPUT_PER_CORE_WIDTH // 32,
            3,
            1,
            8,
            20000,
        ),
        (
            INPUT_PER_CORE_HEIGHT * GRID_Y,
            INPUT_PER_CORE_WIDTH * GRID_X,
            OUTPUT_PER_CORE_WIDTH * GRID_X,
            INPUT_PER_CORE_HEIGHT // 32,
            OUTPUT_PER_CORE_WIDTH // 32,
            3,
            1,
            1,
            20000,
        ),
    ),
    ids=["ff1-hang", "ff1-pass"],
)
def test_reproduce_matmul_2d_hang(
    device,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
):
    torch.manual_seed(1234)

    in0_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        # Volume must match batch size
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1),
                    ),
                }
            ),
            [
                INPUT_PER_CORE_HEIGHT,
                INPUT_PER_CORE_WIDTH,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    in1_mem_config = ttnn.DRAM_MEMORY_CONFIG

    out_mem_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

    in0_dtype = ttnn.bfloat16
    in1_dtype = ttnn.bfloat8_b
    out_dtype = ttnn.bfloat16

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)

    a_t = torch2tt_tensor(A, device, ttnn.Layout.TILE, in0_mem_config, in0_dtype)
    b_t = torch2tt_tensor(B, device, ttnn.Layout.TILE, in1_mem_config, in1_dtype)

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(GRID_X, GRID_Y),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # First run for a reference output
    out = ttnn.matmul(
        a_t,
        b_t,
        program_config=program_config,
        memory_config=out_mem_config,
        dtype=out_dtype,
        compute_kernel_config=compute_config,
    )

    start_time = time.time()

    # loop_count iterations to test determinism/hang
    for i in range(loop_count):
        out.deallocate(True)
        out = ttnn.matmul(
            a_t,
            b_t,
            program_config=program_config,
            memory_config=out_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_config,
        )

        if i % 100 == 0:
            seconds = time.time() - start_time
            print(f"Iteration {i} done, time elapsed from the beginning: {seconds:.2f} seconds")

    out.deallocate(True)
