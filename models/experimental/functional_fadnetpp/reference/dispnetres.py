# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.functional_fadnetpp.reference.resblock import ResBlock


class DispNetRes(nn.Module):
    def __init__(self, in_planes, resBlock=True, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(DispNetRes, self).__init__()

        self.input_channel = input_channel
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio
        self.resBlock = resBlock
        self.res_scale = 7  # number of residuals

        # improved with shrink res-block layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, self.basicE, kernel_size=7, stride=2, padding=(7 - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.conv2 = ResBlock(self.basicE, self.basicE * 2, stride=2)
        self.conv3 = ResBlock(self.basicE * 2, self.basicE * 4, stride=2)
        self.conv3_1 = ResBlock(self.basicE * 4, self.basicE * 4)
        self.conv4 = ResBlock(self.basicE * 4, self.basicE * 8, stride=2)
        self.conv4_1 = ResBlock(self.basicE * 8, self.basicE * 8)
        self.conv5 = ResBlock(self.basicE * 8, self.basicE * 16, stride=2)

    def forward(self, inputs, flows, get_features=False):
        input_features = inputs
        if type(flows) == tuple:
            base_flow = flows
        else:
            base_flow = [F.interpolate(flows, scale_factor=2 ** (-i)) for i in range(7)]

        conv1 = self.conv1(input_features)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3(conv2)
        conv3b = self.conv3_1(conv3a)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        return conv5a
