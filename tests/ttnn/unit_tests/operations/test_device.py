# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_2D_tensors(device, mesh_device, h, w):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_2D_tensors_mesh_device(device, mesh_device, h, w):
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    output = ttnn.add(input_tensor_a, input_tensor_b)
    tt_output = ttnn.to_torch(output, mesh_composer=output_mesh_composer, torch_rank=1)
    assert_with_pcc(torch_output_tensor, tt_output, 0.9999)
