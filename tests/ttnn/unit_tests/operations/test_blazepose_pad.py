# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [16, 128, 64, 32])
@pytest.mark.parametrize(
    "padding,torch_padding",
    [
        (((1, 2), (1, 2)), (1, 2, 1, 2)),
    ],
)
@pytest.mark.parametrize("value", [0])
def test_pad_blazepose(device, h, padding, torch_padding, value):
    torch.manual_seed(0)
    w = h
    torch_input_tensor = torch.rand((1, 128, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.from_device(input_tensor)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    output_tensor = ttnn.to_torch(output_tensor)
    assert output_tensor.shape == torch_output_tensor.shape

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


"""
pytorch:
[1, 128, 16, 16] --> [1, 128, 19, 19]
[1, 128, 128, 128] --> [1, 128, 131, 131]
[1, 128, 64, 64] --> [1, 128, 67, 67]
[1, 128, 32, 32] --> [1, 128, 35, 35]

ttnn:
E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/data_movement/pad/pad.cpp:92: front_padding_is_zero
E       info:
E       ttnn.pad: on device padding does not support front padding
"""
