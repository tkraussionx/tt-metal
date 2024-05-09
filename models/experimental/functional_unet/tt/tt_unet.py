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
    # input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
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
        model,
    ) -> None:
        self.enc1_1 = model.encoder1[:3]
        self.enc1_2 = model.encoder1[3:]

        self.pool1 = model.pool1

        self.enc2_1 = model.encoder2[:3]
        self.enc2_2 = model.encoder2[3:]

        self.pool2 = model.pool2

        # enc3_1 in ttnn, To run in torch have model.encoder3[:3]
        self.enc3_1 = parameters.encoder3_c1  # model.encoder3[:3]

        self.enc3_2 = model.encoder3[3:]

        self.pool3 = model.pool3

        self.enc4_1 = model.encoder4[:3]
        self.enc4_2 = model.encoder4[3:]

        self.pool4 = model.pool4

        self.bnc1_1 = model.bottleneck[:3]
        self.bnc1_2 = model.bottleneck[3:]

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4.weight = torch.nn.Parameter(state_dict["upconv4.weight"])
        self.upconv4.bias = torch.nn.Parameter(state_dict["upconv4.bias"])

        self.dc4_1 = model.decoder4[:3]  # parameters.decoder4_c1
        self.dc4_2 = model.decoder4[3:]

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3.weight = torch.nn.Parameter(state_dict["upconv3.weight"])
        self.upconv3.bias = torch.nn.Parameter(state_dict["upconv3.bias"])

        self.dc3_1 = model.decoder3[:3]  # parameters.decoder3_c1
        self.dc3_2 = model.decoder3[3:]  # parameters.decoder3_c2

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2.weight = torch.nn.Parameter(state_dict["upconv2.weight"])
        self.upconv2.bias = torch.nn.Parameter(state_dict["upconv2.bias"])

        self.dc2_1 = model.decoder2[:3]  # parameters.decoder2_c1
        self.dc2_2 = model.decoder2[3:]  # parameters.decoder2_c2

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv1.weight = torch.nn.Parameter(state_dict["upconv1.weight"])
        self.upconv1.bias = torch.nn.Parameter(state_dict["upconv1.bias"])

        self.dc1_1 = model.decoder1[:3]
        self.dc1_2 = model.decoder1[3:]

        self.conv = model.conv

    def __call__(self, device, input_tensor):
        output_tensor_enc1_1 = self.enc1_1(input_tensor)
        output_tensor_enc1_2 = self.enc1_2(output_tensor_enc1_1)

        output_tensor_pool_1 = self.pool1(output_tensor_enc1_2)

        output_tensor_enc2_1 = self.enc2_1(output_tensor_pool_1)
        output_tensor_enc2_2 = self.enc2_2(output_tensor_enc2_1)

        output_tensor_pool_2 = self.pool2(output_tensor_enc2_2)

        # Enc3_1 alone in ttnn
        output_tensor_pool_2 = torch.permute(output_tensor_pool_2, (0, 2, 3, 1))
        output_tensor_pool_2 = output_tensor_pool_2.reshape(
            output_tensor_pool_2.shape[0],
            1,
            output_tensor_pool_2.shape[1] * output_tensor_pool_2.shape[2],
            output_tensor_pool_2.shape[3],
        )
        output_tensor_pool_2 = ttnn.from_torch(output_tensor_pool_2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        output_tensor_pool_2 = output_tensor_pool_2.to(device, self.enc3_1.conv.input_sharded_memory_config)
        output_tensor_enc3_1 = self.enc3_1(output_tensor_pool_2)

        output_tensor_enc3_1 = ttnn_to_torch(output_tensor_enc3_1)
        output_tensor_enc3_1 = output_tensor_enc3_1.reshape(1, 120, 160, 128)
        output_tensor_enc3_1 = torch.permute(output_tensor_enc3_1, (0, 3, 1, 2))
        output_tensor_enc3_1 = output_tensor_enc3_1.to(dtype=torch.float)

        # To run enc3_1 in torch command line 113 to 124 and uncommand line 127
        # output_tensor_enc3_1 = self.enc3_1(output_tensor_pool_2)

        output_tensor_enc3_2 = self.enc3_2(output_tensor_enc3_1)

        output_tensor_pool_3 = self.pool3(output_tensor_enc3_2)
        output_tensor_enc4_1 = self.enc4_1(output_tensor_pool_3)
        output_tensor_enc4_2 = self.enc4_2(output_tensor_enc4_1)

        output_tensor_pool_4 = self.pool4(output_tensor_enc4_2)

        output_tensor_bnc1_1 = self.bnc1_1(output_tensor_pool_4)
        output_tensor_bnc1_2 = self.bnc1_2(output_tensor_bnc1_1)

        output_tensor_dc_4 = self.upconv4(output_tensor_bnc1_2)
        output_tensor_dc_4 = torch.concat([output_tensor_dc_4, output_tensor_enc4_2], dim=1)

        output_tensor_dc4_1 = self.dc4_1(output_tensor_dc_4)
        output_tensor_dc4_2 = self.dc4_2(output_tensor_dc4_1)

        output_tensor_dc_3 = self.upconv3(output_tensor_dc4_2)
        output_tensor_dc_3 = torch.concat([output_tensor_dc_3, output_tensor_enc3_2], dim=1)

        output_tensor_dc3_1 = self.dc3_1(output_tensor_dc_3)
        output_tensor_dc3_2 = self.dc3_2(output_tensor_dc3_1)

        output_tensor_dc_2 = self.upconv2(output_tensor_dc3_2)
        output_tensor_dc_2 = torch.concat([output_tensor_dc_2, output_tensor_enc2_2], dim=1)

        output_tensor_dc2_1 = self.dc2_1(output_tensor_dc_2)
        output_tensor_dc2_2 = self.dc2_2(output_tensor_dc2_1)

        output_tensor_dc_1 = self.upconv1(output_tensor_dc2_2)

        output_tensor_dc_1 = torch.concat([output_tensor_dc_1, output_tensor_enc1_2], dim=1)

        output_tensor_dc1_1 = self.dc1_1(output_tensor_dc_1)
        output_tensor_dc1_2 = self.dc1_2(output_tensor_dc1_1)

        output_tensor_conv = self.conv(output_tensor_dc1_2)

        output_tensor = torch.sigmoid(output_tensor_conv)

        return output_tensor
