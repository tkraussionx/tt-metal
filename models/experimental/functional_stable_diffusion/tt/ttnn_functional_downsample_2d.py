# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch.nn as nn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm


def downsample_2d(
    in_channels,
    hidden_states,
    device,
    parameters,
    use_conv=False,
    out_channels=None,
    padding=1,
    name="conv",
):
    stride = 2
    if use_conv:
        conv = fallback_ops.Conv2d(
            parameters.conv.weight,
            parameters.conv.bias,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )

    else:
        assert in_channels == out_channels
        assert False, " we don't support AvgPool2d, and we should not need it either"
        conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

    if name == "conv":
        Conv2d_0 = conv
        conv = conv
    elif name == "Conv2d_0":
        conv = conv
    else:
        conv = conv

    assert hidden_states.shape[1] == in_channels

    if use_conv and padding == 0:
        pad = (0, 1, 0, 1)
        hidden_states = ttnn.pad(hidden_states, pad, value=0)

    assert hidden_states.shape[1] == in_channels

    hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.from_device(hidden_states)
    hidden_states = ttnn.to_torch(hidden_states)
    hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

    hidden_states = conv(hidden_states)

    return hidden_states
