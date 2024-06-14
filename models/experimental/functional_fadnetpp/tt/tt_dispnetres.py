# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.functional_fadnetpp.tt.ttnn_resblock import TtResBlock
import tt_lib
from tt_lib.fallback_ops import fallback_ops


class TtDispNetRes(nn.Module):
    def __init__(
        self, parameters, device, in_planes, resBlock=True, input_channel=3, encoder_ratio=16, decoder_ratio=16
    ):
        super(TtDispNetRes, self).__init__()
        self.input_channel = input_channel
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio
        self.resBlock = resBlock
        self.res_scale = 7
        self.parameters = parameters
        self.device = device

        self.conv1 = parameters.conv1
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = TtResBlock(parameters.conv2, self.basicE, self.basicE * 2, stride=2)
        self.conv3 = TtResBlock(parameters.conv3, self.basicE * 2, self.basicE * 4, stride=2)
        self.conv3_1 = TtResBlock(parameters.conv3_1, self.basicE * 4, self.basicE * 4)
        self.conv4 = TtResBlock(parameters.conv4, self.basicE * 4, self.basicE * 8, stride=2)
        self.conv4_1 = TtResBlock(parameters.conv4_1, self.basicE * 8, self.basicE * 8)
        self.conv5 = TtResBlock(parameters.conv5, self.basicE * 8, self.basicE * 16, stride=2)

    def __call__(self, device, input_tensor, flows, get_features=False):
        input_features = input_tensor
        if type(flows) == tuple:
            base_flow = flows
        else:
            base_flow = [fallback_ops.interpolate(flows, scale_factor=2 ** (-i)) for i in range(7)]

        input_features = tt_lib.tensor.interleaved_to_sharded(
            input_features, self.conv1.conv.input_sharded_memory_config
        )

        conv1 = self.conv1(input_features)
        conv1 = ttnn.to_torch(conv1)
        conv1 = self.leaky_relu(conv1)
        conv1 = ttnn.from_torch(conv1, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        conv2 = self.conv2(device, conv1)
        conv2 = ttnn.to_device(conv2, device)
        conv3a = self.conv3(device, conv2)
        conv3b = self.conv3_1(device, conv3a)
        conv4a = self.conv4(device, conv3b)
        conv4b = self.conv4_1(device, conv4a)
        conv5a = self.conv5(device, conv4b)
        return conv5a
