# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from time import time
import pytest
import torch
import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor
from viztracer import VizTracer


def test_ttnn_matmul(device_mesh, use_program_cache):
    x = ttnn.as_tensor(
        torch.randn(1, 1, 32, 4096),
        device=device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    w = ttnn.as_tensor(
        torch.randn(1, 1, 4096, 32000),
        device=device_mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    with VizTracer() as tracer:
        for loop in range(10):
            start = time()
            outputs = ttnn.linear(
                x,
                w,
                core_grid=ttnn.CoreGrid(y=7, x=6),
                use_1d_systolic_array=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=compute_kernel_config,
            )
            dispatch = time() - start

            result = ttnn.to_torch(outputs, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))
            total = time() - start

            print(f"Loop {loop+1}: Dispatch time: {1e6*dispatch:.0f} us, Total time: {1e6*total:.0f} us")
