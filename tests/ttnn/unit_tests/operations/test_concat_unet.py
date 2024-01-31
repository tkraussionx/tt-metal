# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# @pytest.mark.parametrize("h", [20])
# @pytest.mark.parametrize("w", [4])
@pytest.mark.parametrize(
    "shapes, dim",
    (
        (((8, 132, 20, 32), (8, 132, 20, 64)), 3),
        (((8, 264, 40, 32), (8, 264, 40, 32)), 3),
        (((8, 528, 80, 16), (8, 528, 80, 32)), 3),
        (((8, 1056, 160, 16), (8, 1056, 160, 16)), 3),
    ),
)
def test_concat(device, shapes, dim):
    print("shapes[0]: ", shapes[0])
    print("shapes[1]: ", shapes[1])
    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=dim)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
