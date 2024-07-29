# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import ttnn

from ttnn.multi_device import ListMeshToTensor, ReplicateTensorToMesh
from models.utility_functions import comp_pcc
import torch


@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    ((1024, 4608, 18432, 4, 72, 3, 1, 8, 100000),),
    ids=[
        "ff1-hang",
    ],
)
@pytest.mark.parametrize(
    "device_mesh",
    [
        32,
    ],
    indirect=True,
)
def test_reproduce_matmul_2d_hang(
    device_mesh,
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

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    in0_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        # Volume must match batch size
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    ),
                }
            ),
            [
                128,
                576,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    A = torch.randn(a_shape)
    B = torch.randn(b_shape)
    OUT = A @ B

    a_t = ttnn.from_torch(
        A,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    a_t = ttnn.to_device(a_t, device_mesh, memory_config=in0_mem_config)
    print("Pushed input0 to device.")

    b_t = ttnn.from_torch(
        B,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    b_t = ttnn.to_device(b_t, device_mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("Pushed input1 to device.")

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=[ttnn.UnaryOpType.GELU, True],
    )

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out = ttnn.matmul(
        a_t,
        b_t,
        program_config=program_config,
        memory_config=out_mem_config,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config,
    )

    for i in range(loop_count):
        ttnn.deallocate(out, force=True)
        out = ttnn.matmul(
            a_t,
            b_t,
            program_config=program_config,
            memory_config=out_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_config,
        )

        ttnn.synchronize_devices(device_mesh)

        if i % 100 == 0:
            logger.info(f"Running iteration {i}")

            outputs = ttnn.to_torch(out, mesh_composer=ListMeshToTensor(device_mesh))

            for device_id, output in enumerate(outputs):
                _, output_pcc = comp_pcc(OUT, output)
                print(f"Device {device_id} PCC: {output_pcc}")

    ttnn.deallocate(out, force=True)
