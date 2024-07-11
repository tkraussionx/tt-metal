# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import time
import ttnn

import tt_lib as ttl
from models.utility_functions import comp_pcc, torch2tt_tensor, tt2torch_tensor
import torch


@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    ((1024, 4608, 18432, 4, 72, 3, 1, 8, 20000),),
    ids=[
        "ff1-hang",
    ],
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

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)

    a_t = torch2tt_tensor(A, device, ttl.tensor.Layout.TILE, in0_mem_config, in0_dtype)
    b_t = torch2tt_tensor(B, device, ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype)

    program_config = ttnn._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=[ttnn.experimental.tensor.FusibleActivation.GELU, True],
    )

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.HiFi2,
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

    ref_out = tt2torch_tensor(out)

    # if commented out, segfault happens on some of the latter runs
    out.cpu()

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

            _, output_pcc = comp_pcc(ref_out, tt2torch_tensor(out))
            print(f"PCC: {output_pcc}")

    out.deallocate(True)
