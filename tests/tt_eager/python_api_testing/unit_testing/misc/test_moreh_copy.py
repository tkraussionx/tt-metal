# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger

TILE_HEIGHT = 32
TILE_WIDTH = 32


def to_cpu(npu_tensor, shape, *, cpu_layout=ttl.tensor.Layout.ROW_MAJOR):
    if npu_tensor is None:
        return None
    if not isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(shape).to_torch()
    return cpu_tensor


def to_npu(
    cpu_tensor,
    device,
    *,
    npu_layout=ttl.tensor.Layout.TILE,
    npu_dtype=ttl.tensor.DataType.BFLOAT16,
    shape=None,
):
    if cpu_tensor is None:
        return None
    if shape is not None:
        cpu_tensor = cpu_tensor.view(shape)
    npu_tensor = ttl.tensor.Tensor(cpu_tensor, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    return npu_tensor


if __name__ == "__main__":
    torch.manual_seed(2024)

    torch.set_printoptions(threshold=1024, edgeitems=1000, precision=2, sci_mode=False)

    device = ttl.device.CreateDevice(0)

    # x = torch.randn(2, 2, 2, 2, dtype=torch.bfloat16)

    x = torch.randn(1, 1, 8, 8, dtype=torch.bfloat16)

    # out = torch.full((1, 1, 8, 8), -100.0, dtype=torch.bfloat16)

    xx = to_npu(x, device)

    y = ttl.operations.primary.moreh_copy(xx)

    yy = to_cpu(y, x.shape)

    print("input")
    print(x)
    print("======================================================")
    print("output")
    print(yy)

    # print("======================================================")
    # print(torch.allclose(yy, x))

    ttl.device.CloseDevice(device)
