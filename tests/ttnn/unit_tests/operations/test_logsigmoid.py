# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import os
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import torch_random


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("h", [256])
@pytest.mark.parametrize("w", [256])
# @pytest.mark.parametrize("dim", [-1, -2, -3])
def test_logsigmoid(device, batch_size, h, w):
    torch.manual_seed(0)
    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = F.sigmoid(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.log_sigmoid(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
