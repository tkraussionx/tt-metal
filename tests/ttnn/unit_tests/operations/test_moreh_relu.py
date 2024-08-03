# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import math
from typing import Union, Tuple
import torch
import torch.nn as nn
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "which_relu, bound",
    [
        (0, 0),
        (1, 3),
        (2, 3),
    ],
)
def test_moreh_relu_max(which_relu, bound, device):
    shape = [128, 32, 32]
    torch_dtype = torch.bfloat16
    tt_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    torch_input = torch.randint(-100, 100, shape, dtype=torch_dtype)
    if which_relu == 0:
        torch_output = torch.relu(torch_input)
    elif which_relu == 1:
        torch_output = torch.relu(torch.clamp(torch_input, min=None, max=bound))
    else:
        torch_output = torch.relu(torch.clamp(torch_input, min=bound, max=None))

    tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    print(f"tt_input\n{tt_input}")
    ttnn.moreh_relu(tt_input, tt_output, which_relu=which_relu, bound=bound)
    print(f"tt_output\n{tt_output}")

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    passed = torch.equal(tt_output, torch_output)
    assert passed
