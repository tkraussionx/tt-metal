# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.utility_functions import (
    torch_to_tt_tensor_rm,
)

from models.experimental.functional_stable_diffusion.tt.ttnn_upsample_nearest2d import upsample_nearest2d
from tt_lib.fallback_ops import fallback_ops


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

    conv = fallback_ops.Conv2d(
        parameters.conv.weight,
        parameters.conv.bias,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    tt_out = conv(tt_out)
    return tt_out
