# # SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# # SPDX-License-Identifier: Apache-2.0

# import torch
# import ttnn

# device_id = 0
# device = ttnn.open_device(device_id=device_id)

# torch.manual_seed(0)

# torch_input_tensor_a = torch.rand((32, 32), dtype=torch.bfloat16)
# torch_input_tensor_b = torch.rand((32, 32), dtype=torch.bfloat16)

# input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
# input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

# output_tensor = input_tensor_a + input_tensor_b

# print("output_tensor", output_tensor)

# # example_output = ttnn.example(output_tensor)

# example_output = ttnn.moreh_layernorm(output_tensor, output=output_tensor)

# output_tensor = ttnn.to_torch(example_output)

# print("example_output", example_output)

# ttnn.close_device(device)


# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    [
        [4, 4],  # 256x256
    ],
)
def test_moreh_layernorm(device, input_shapes):
    normalized_dims = 1
    eps = 0.01

    torch.manual_seed(0)

    input = torch.rand(input_shapes, dtype=torch.bfloat16)

    # run torch
    normalized_shape = input.shape[-normalized_dims:]
    torch_result = F.layer_norm(input, normalized_shape, weight=None, bias=None, eps=eps)

    # run tt
    tt_input = ttnn.from_torch(input, device=device)
    tt_output = ttnn.from_torch(torch.empty_like(input), device=device)
    ttnn.moreh_layernorm(tt_input, normalized_dims, eps, output=tt_output)
    tt_result = ttnn.to_torch(tt_output)

    print("torch_result", torch_result)
    print("tt_result", tt_result)
    # assert_with_pcc(torch_result, tt_result)

    # allclose = torch.allclose(tt_result, torch_result)

    # assert allclose
