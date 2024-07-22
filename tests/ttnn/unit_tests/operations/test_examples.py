# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import math
from typing import Union, Tuple

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


TILE_WIDTH = 32


@pytest.mark.parametrize(
    "input_shapes",
    [
        [2, 2, 2, 2],
    ],
)
def test_upsample_single_core(device, input_shapes):
    batch_size, height, width, num_channels = input_shapes

    torch.manual_seed(0)

    # run torch
    input = torch.rand(input_shapes, dtype=torch.bfloat16)
    torch_result = input + 1

    # run TT
    input_tensor = ttnn.from_torch(input, device=device)
    output_tensor = ttnn.example(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_result, output_tensor)

    allclose = torch.allclose(output_tensor, torch_result)

    assert allclose
