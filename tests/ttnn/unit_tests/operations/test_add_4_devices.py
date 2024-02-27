# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger


@pytest.mark.parametrize("size", [64])
def test_add_1D_tensor_and_scalar(pcie_devices, size):
    torch.manual_seed(0)
    output_tensors_device = []
    torch_output_tensors = []

    for idx, device in enumerate(pcie_devices):
        torch_input_tensor = torch.ones((size,), dtype=torch.bfloat16) * (idx + 1)
        device_input_tensor_a = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        device_input_tensor_b = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensors_device.append(device_input_tensor_a + device_input_tensor_b)
        torch_output_tensors.append(torch_input_tensor + torch_input_tensor)

    for idx, device in enumerate(pcie_devices):
        output_tensor_device = ttnn.to_torch(output_tensors_device[idx], torch_rank=1)
        output_tensor_cpu = torch_output_tensors[idx]
        logger.info(f"Device id: {idx} Expected value is: {output_tensor_cpu[0]}")

        assert_with_pcc(output_tensor_cpu, output_tensor_device, 0.9999)
