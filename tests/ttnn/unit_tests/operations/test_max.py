# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize("batch_size", [(1, 2)])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dim", [-2])
def test_max(device, batch_size, h, w, dim):
    torch.manual_seed(0)
    numbers = torch.arange(1, 65)
    result = numbers.unsqueeze(0).unsqueeze(0).expand((batch_size[0], batch_size[1], h, w))
    torch_input_tensor = result.bfloat16()
    torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=17)
    print("torch_input_tensor", torch_input_tensor)
    # torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=17)
    print("input_tensor", input_tensor)
    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=17)
    print("out", output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


# @pytest.mark.parametrize("batch_size", [1, 16])
# @pytest.mark.parametrize("h", [32, 64])
# @pytest.mark.parametrize("w", [32, 64])
# def test_max_global(device, batch_size, h, w):
#     torch.manual_seed(0)

#     torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
#     torch_output_tensor = torch.max(torch_input_tensor)

#     input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

#     output_tensor = ttnn.max(input_tensor)
#     output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
#     output_tensor = ttnn.from_device(output_tensor)

#     output_tensor = ttnn.to_torch(output_tensor)
#     output_tensor = output_tensor[0, 0, 0, 0]

#     assert_with_pcc(torch_output_tensor, output_tensor)
