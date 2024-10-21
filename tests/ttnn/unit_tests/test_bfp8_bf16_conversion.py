import pytest
import torch

import ttnn

TILE_HEIGHT = 32
TILE_WIDTH = 32

cpu_layout = ttnn.Layout.ROW_MAJOR
npu_layout = ttnn.Layout.TILE


def test_typecast_bf16_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # bf16 --> bfp8_b by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    print(cpu_version[0, 0:16])
    print(npu_version[0, 0:16])
    assert passed


def test_typecast_bfp8_b_to_bf16(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> bf16 by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> bf16 by npu.
    bf16_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.bfloat16)
    npu_version = bf16_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    print(cpu_version[0, 0:16])
    print(npu_version[0, 0:16])
    assert passed
