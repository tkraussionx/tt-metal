# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn
from models.experimental.yolov4.ttnn.common import Conv
import pytest


class Sample(nn.Module):
    def __init__(self):
        super().__init__()

        self.c13 = nn.Conv2d(512, 1024, 3, 1, 1, bias=True)
        self.c14 = nn.Conv2d(1024, 512, 1, 1, 0, bias=True)
        self.c15 = nn.Conv2d(512, 1024, 3, 1, 1, bias=True)

    def forward(self, inputs):
        output_tensor = self.c13(inputs)
        output_tensor = self.c14(output_tensor)
        output_tensor = self.c15(output_tensor)

        return output_tensor


class TtSample:
    def __init__(self) -> None:
        # (batch_size,input_height,input_width,in_channels,out_channels,kernel_h,kernel_w,stride_h,stride_w,pad_h,pad_w)
        self.conv13 = Conv((1, 10, 10, 512, 1024, 3, 3, 1, 1, 1, 1), reshard=True, height_sharding=False, activation="")
        self.conv14 = Conv((1, 10, 10, 1024, 512, 1, 1, 1, 1, 0, 0), reshard=True, height_sharding=False, activation="")
        self.conv15 = Conv((1, 10, 10, 512, 1024, 3, 3, 1, 1, 1, 1), reshard=True, height_sharding=False, activation="")

    def __call__(self, device, input_tensor):
        output_tensor = self.conv13(device, input_tensor)
        # output_tensor = self.conv14(device, output_tensor)
        output_tensor = self.conv15(device, output_tensor)

        return output_tensor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_head(device):
    ttnn_model = TtSample()

    torch_input_tensor = torch.randn(1, 10, 10, 512, dtype=torch.bfloat16)
    torch_model = Sample()
    torch_input = torch_input_tensor.permute(0, 3, 1, 2).float()
    output = torch_model(torch_input)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (1, 1, 100, 512))
    ttnn_input_tensor = ttnn.to_layout(ttnn_input_tensor, layout=ttnn.TILE_LAYOUT)
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device=device)

    result_ttnn = ttnn_model(device, ttnn_input_tensor)
