# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
import tt_lib
import tt_lib.fallback_ops
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from tt_lib import tensor as ttl_tensor, device as ttl_device
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


def ttnn_to_torch(input):
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def permute_conv_weights(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


class TtUnet:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.conv_weight = torch_to_tt_tensor_rm(parameters["conv"]["weight"], device, put_on_device=False)
        self.conv_bias = torch_to_tt_tensor_rm(parameters["conv"]["bias"], device, put_on_device=False)
        self.conv = tt_lib.fallback_ops.Conv2d(
            weights=self.conv_weight,
            biases=self.conv_bias,
            in_channels=480,
            out_channels=640,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )

    def __call__(self, device, input_tensor):
        output_tensor_dc1_2 = ttnn.to_torch(input_tensor)
        output_tensor_dc1_2 = output_tensor_dc1_2.reshape(1, 480, 640, 32)
        output_tensor_dc1_2 = torch.permute(output_tensor_dc1_2, (0, 3, 1, 2))
        output_tensor_dc1_2 = torch_to_tt_tensor_rm(output_tensor_dc1_2, device, put_on_device=True)

        output_tensor_conv = self.conv(output_tensor_dc1_2)

        output_tensor_conv = tt_to_torch_tensor(output_tensor_conv)
        output_tensor_conv = torch.permute(output_tensor_conv, (0, 2, 3, 1))
        output_tensor_conv = output_tensor_conv.reshape(
            output_tensor_conv.shape[0],
            1,
            output_tensor_conv.shape[1] * output_tensor_conv.shape[2],
            output_tensor_conv.shape[3],
        )
        output_tensor_conv = ttnn.from_torch(
            output_tensor_conv, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        return ttnn.from_device(output_tensor_conv)
