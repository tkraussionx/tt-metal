# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import re
from loguru import logger
import pytest

import time

import tt_lib as ttl
import ttnn
from models.utility_functions import comp_pcc, torch2tt_tensor, tt2torch_tensor
import torch


# Used to reproduce issue #8665 with matmul 2D (Falcon 7b matmuls)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    ((1024, 4608, 18432, 4, 72, 3, 1, 8, 50000), (1024, 4608, 18432, 4, 72, 3, 1, 1, 20000)),
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

    in0_initial_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )
    in0_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        # Volume must match batch size
                        ttl.tensor.CoreCoord(0, 0),
                        ttl.tensor.CoreCoord(7, 7),
                    ),
                }
            ),
            [
                128,
                576,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1)

    in0_dtype = ttl.tensor.DataType.BFLOAT16
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT16

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]
    c_shape = [1, 1, 18432, 4608]
    result_shape = [1, 1, seq_len, 4608]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)
    C = torch.randn(c_shape)

    a_t = torch2tt_tensor(A, device, ttl.tensor.Layout.TILE, in0_initial_mem_config, in0_dtype)
    b_t = torch2tt_tensor(B, device, ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype)
    c_t = torch2tt_tensor(C, device, ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype)

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    program_config_2 = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=4,
        per_core_N=18,
        transpose_mcast=False,
        fused_activation=None,
    )

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    final_out = ttnn.from_torch(
        torch.zeros(result_shape),
        ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    a_t_sharded = ttl.tensor.interleaved_to_sharded_partial(
        a_t,
        (8, 8),
        [128, 576],
        1,
        0,
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    # First run for a reference output
    out = ttl.operations.primary.matmul(
        a_t_sharded,
        b_t,
        program_config=program_config,
        output_mem_config=out_mem_config,
        output_dtype=out_dtype,
        compute_kernel_config=compute_config,
    )

    out_2 = ttnn.matmul(
        out,
        c_t,
        program_config=program_config_2,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
    )

    result = ttl.tensor.sharded_to_interleaved_partial(
        out_2,
        final_out,
        1,
        0,
        in0_initial_mem_config,
    )

    RESULT = torch.matmul(torch.matmul(A, B), C)

    torch_out = tt2torch_tensor(result)
    does_pass, output_pcc = comp_pcc(RESULT, torch_out, 0.99)
    logger.info(f"PCC value: {output_pcc}")

    start_time = time.time()

    # loop_count iterations to test determinism/hang
    for i in range(loop_count):
        a_t_sharded.deallocate(True)
        out.deallocate(True)
        out_2.deallocate(True)

        a_t_sharded = ttl.tensor.interleaved_to_sharded_partial(
            a_t,
            (8, 8),
            [128, 576],
            1,
            0,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.ROW_MAJOR,
        )
        out = ttl.operations.primary.matmul(
            a_t,
            b_t,
            program_config=program_config,
            output_mem_config=out_mem_config,
            output_dtype=out_dtype,
            compute_kernel_config=compute_config,
        )
        out_2 = ttnn.matmul(
            out,
            c_t,
            program_config=program_config_2,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        result = ttl.tensor.sharded_to_interleaved_partial(
            out_2,
            final_out,
            1,
            0,
            in0_initial_mem_config,
        )

        if i % 100 == 0:
            seconds = time.time() - start_time
            print(f"Iteration {i} done, time elapsed from the beginning: {seconds:.2f} seconds")

    out.deallocate(True)
