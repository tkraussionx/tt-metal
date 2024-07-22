# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import time

import tt_lib as ttl
from models.utility_functions import (
    torch2tt_tensor,
    get_devices_for_t3000,
)
import torch


# Used to reproduce issue #8665 with matmul 2D (Falcon 7b matmuls)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    ((1024, 4608, 18432, 4, 72, 3, 1, 8, 20000), (1024, 4608, 18432, 4, 72, 3, 1, 1, 20000)),
    ids=["ff1-hang", "ff1-pass"],
)
@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_reproduce_matmul_2d_hang(
    num_devices,
    all_devices,
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

    devices = []
    if num_devices == 8:
        devices = get_devices_for_t3000(all_devices, num_devices)
    else:
        devices = all_devices

    print("Running on ", num_devices, " devices")

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

    a_t = []
    b_t = []

    for device_idx in range(num_devices):
        a_t.append(torch2tt_tensor(A, devices[device_idx], ttl.tensor.Layout.TILE, in0_mem_config, in0_dtype))
        b_t.append(torch2tt_tensor(B, devices[device_idx], ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype))

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

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out = []
    # First run for a reference output
    for device_idx in range(num_devices):
        out.append(
            ttl.operations.primary.matmul(
                a_t[device_idx],
                b_t[device_idx],
                program_config=program_config,
                output_mem_config=out_mem_config,
                output_dtype=out_dtype,
                compute_kernel_config=compute_config,
            )
        )

    nd_output_count = 0

    start_time = time.time()

    # loop_count iterations to test determinism/hang
    for i in range(loop_count):
        for device_idx in range(num_devices):
            out[device_idx].deallocate(True)
            out[device_idx] = ttl.operations.primary.matmul(
                a_t[device_idx],
                b_t[device_idx],
                program_config=program_config,
                output_mem_config=out_mem_config,
                output_dtype=out_dtype,
                compute_kernel_config=compute_config,
            )

        if i % 100 == 0:
            seconds = time.time() - start_time
            print(f"Iteration {i} done, time elapsed from the beginning: {seconds:.2f} seconds")

    for device_idx in range(num_devices):
        out[device_idx].deallocate(True)

    print(f"Iterations with nd output: {nd_output_count}")

    for device_idx in range(num_devices):
        ttl.device.Synchronize(devices[device_idx])

    assert nd_output_count != 0
