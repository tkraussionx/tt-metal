# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import math
from typing import Union, Tuple
import torch
import torch.nn as nn
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_binary_lt_int32(device):
    shape = [32, 32]
    torch_dtype = torch.bfloat16
    tt_dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT

    torch_input = torch.randint(-100, 100, shape, dtype=torch_dtype)
    torch_output = torch.relu(torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_input = ttnn.to_device(tt_input, device)

    tt_output = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=layout)
    tt_output = ttnn.to_device(tt_output, device)

    print(f"tt_output before\n{tt_output}")
    ttnn.moreh_relu(tt_input, tt_output, False, 0)
    print(f"tt_output before\n{tt_output}")

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    passed = torch.equal(tt_output, torch_output)
    assert passed
