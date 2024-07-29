# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn

import torch
from ttnn.multi_device import ListMeshToTensor, ReplicateTensorToMesh
from models.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "device_mesh",
    [
        32,
    ],
    indirect=True,
)
def test_reproduce_lm_head_nd_32(
    device_mesh,
):
    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    in0_dtype = ttnn.DataType.BFLOAT8_B
    in1_dtype = ttnn.DataType.BFLOAT8_B
    out_dtype = ttnn.DataType.BFLOAT8_B

    torch.manual_seed(1234)

    seq_len = 32
    a_shape = [1, 1, seq_len, 4544]
    b_shape = [1, 1, 4544, 65024]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    OUT = A @ B

    a_t = ttnn.from_torch(
        A,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
        memory_config=in0_mem_config,
    )

    print("Pushed input0 to device.")

    b_t = ttnn.from_torch(
        B,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
        memory_config=in1_mem_config,
    )

    print("Pushed input1 to device.")

    mm_prog_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=1,
        per_core_N=32,
        out_subblock_h=1,
        out_subblock_w=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    wh_compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out = ttnn.matmul(
        a_t,
        b_t,
        program_config=mm_prog_config,
        memory_config=out_mem_config,
        dtype=out_dtype,
        compute_kernel_config=wh_compute_kernel_config,
    )

    for i in range(1000):
        ttnn.deallocate(out, force=True)
        out = ttnn.matmul(
            a_t,
            b_t,
            program_config=mm_prog_config,
            memory_config=out_mem_config,
            dtype=out_dtype,
            compute_kernel_config=wh_compute_kernel_config,
        )
        # ttnn.synchronize_devices(device_mesh)
        for device_id, device in enumerate(device_mesh.get_device_ids()):
            print(f"Sync device {device_id}")
            ttnn._ttnn.deprecated.device.Synchronize(device_mesh.get_device(device))

        if i % 100 == 0:
            logger.info(f"Running iteration {i}")
        #     outputs = ttnn.to_torch(out, mesh_composer=ListMeshToTensor(device_mesh))

        #     for device_id, output in enumerate(outputs):
        #         _, output_pcc = comp_pcc(OUT, output)
        #         print(f"Device {device_id} PCC: {output_pcc}")
