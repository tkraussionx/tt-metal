# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_upsample_nearest_2d import upsample_nearest2d
from tt_lib.fallback_ops import fallback_ops


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


def upsample2d(
    device,
    input,
    parameters,
    in_channels,
    out_channels,
    scale_factor=2,
):
    tt_out = upsample_nearest2d(input, scale_factor)

    tt_out = ttnn.from_device(tt_out)
    tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
    tt_out = ttnn.to_torch(tt_out)
    tt_out = torch_to_tt_tensor_rm(tt_out, device)

    weight = ttnn.to_layout(parameters.conv.weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(parameters.conv.bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)

    weight = torch_to_tt_tensor_rm(weight, device, put_on_device=False)
    bias = torch_to_tt_tensor_rm(bias, device, put_on_device=False)

    conv = fallback_ops.Conv2d(
        weight,
        bias,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    tt_out = conv(tt_out)
    tt_out = tt_to_torch_tensor(tt_out)
    tt_out = torch_to_ttnn(tt_out, device=device)
    return tt_out
