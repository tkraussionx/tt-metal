# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from models.experimental.functional_fadnet.tt.ttnn_resblock import TtResBlock
import ttnn
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
import math


class TtDispNetC:
    def build_corr(img_left, img_right, max_disp=40):
        B, C, H, W = img_left.shape
        volume = img_left.new_zeros([B, max_disp, H, W])
        for i in range(max_disp):
            if i > 0:
                volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :-i]).mean(dim=1)
            else:
                volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

        volume = volume.contiguous()
        return volume

    def output_preprocessing(self, output_tensor, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = torch.reshape(
            output_tensor,
            [
                output_tensor.shape[0],
                output_tensor.shape[1],
                int(math.sqrt(output_tensor.shape[3])),
                int(math.sqrt(output_tensor.shape[3])),
            ],
        )
        output_tensor = torch_to_tt_tensor_rm(output_tensor, device, put_on_device=True)
        return output_tensor

    def input_preprocessing(self, input_tensor, device):
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.reshape(
            input_tensor,
            (input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]),
        )
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
        return input_tensor

    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        ## self.l1 = parameters.l1
        self.res1 = TtResBlock(parameters.res1, 64, 128, stride=2)
        self.res2 = TtResBlock(parameters.res2, 128, 256, stride=2)
        self.res3 = TtResBlock(parameters.res3, 256, 32, stride=1)
        self.res3_1 = TtResBlock(parameters.res3_1, 72, 256)
        self.res4 = TtResBlock(parameters.res4, 256, 512, stride=2)
        self.res4_1 = TtResBlock(parameters.res4_1, 512, 512)
        self.res5 = TtResBlock(parameters.res5, 512, 512, stride=2)
        self.res5_1 = TtResBlock(parameters.res5_1, 512, 512)
        self.res6 = TtResBlock(parameters.res6, 512, 1024, stride=2)
        self.res6_1 = TtResBlock(parameters.res6_1, 1024, 1024)
        self.pred_flow6 = parameters.pred_flow6

        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv5.weight = nn.parameter(parameters.iconv5["weight"])
        self.iconv5.bias = nn.parameter(parameters.iconv5["bias"])
        self.iconv4 = nn.ConvTranspose2d(769, 256, 3, 1, 1)
        self.iconv4.weight = nn.parameter(parameters.iconv4["weight"])
        self.iconv4.bias = nn.parameter(parameters.iconv4["bias"])
        self.iconv3 = nn.ConvTranspose2d(385, 128, 3, 1, 1)
        self.iconv3.weight = nn.parameter(parameters.iconv3["weight"])
        self.iconv3.bias = nn.parameter(parameters.iconv3["bias"])
        self.iconv2 = nn.ConvTranspose2d(193, 64, 3, 1, 1)
        self.iconv2.weight = nn.parameter(parameters.iconv2["weight"])
        self.iconv2.bias = nn.parameter(parameters.iconv2["bias"])
        self.iconv1 = nn.ConvTranspose2d(97, 32, 3, 1, 1)
        self.iconv1.weight = nn.parameter(parameters.iconv1["weight"])
        self.iconv1.bias = nn.parameter(parameters.iconv1["bias"])
        self.iconv0 = nn.ConvTranspose2d(23, 16, 3, 1, 1)
        self.iconv0.weight = nn.parameter(parameters.iconv0["weight"])
        self.iconv0.bias = nn.parameter(parameters.iconv0["bias"])

        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv5.weight = nn.parameter(parameters.upconv5["weight"])
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv4.weight = nn.parameter(parameters.upconv4["weight"])
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv3.weight = nn.parameter(parameters.upconv3["weight"])
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv2.weight = nn.parameter(parameters.upconv2["weight"])
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv1.weight = nn.parameter(parameters.upconv1["weight"])
        self.upconv0 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv0.weight = nn.parameter(parameters.upconv0["weight"])

        self.pred_flow5 = parameters.pred_flow5
        self.pred_flow4 = parameters.pred_flow4
        self.pred_flow3 = parameters.pred_flow3
        self.pred_flow2 = parameters.pred_flow2
        self.pred_flow1 = parameters.pred_flow1
        self.pred_flow0 = parameters.pred_flow0

        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow6to5.weight = nn.parameter(parameters.upflow6to5["weight"])
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow5to4.weight = nn.parameter(parameters.upflow5to4["weight"])
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow4to3.weight = nn.parameter(parameters.upflow4to3["weight"])
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow3to2.weight = nn.parameter(parameters.upflow3to2["weight"])
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow2to1.weight = nn.parameter(parameters.upflow2to1["weight"])
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow1to0.weight = nn.parameter(parameters.upflow1to0["weight"])

    def __call__(self, device, input_tensor):
        imgs = torch.chunk(input_tensor, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]
        img_left = img_left.to(device, self.c1.conv.input_sharded_memory_config)
        img_right = img_right.to(device, self.c1.conv.input_sharded_memory_config)

        conv1_l = self.c1(img_left)
        # conv1_lr = self.l1(conv1_l)
        conv2_l = self.res1(conv1_l)
        conv3a_l = self.res2(conv2_l)
        conv3a_l = self.output_preprocessing(conv3a_l, device)

        conv1_r = self.c1(img_right)
        # conv1_lr = self.l1(conv1_r)
        conv2_r = self.res1(conv1_r)
        conv3a_r = self.res2(conv2_r)
        conv3a_r = self.output_preprocessing(conv3a_r, device)

        # Correlate corr3a_l and corr3a_r
        # out_corr = self.corr(conv3a_l, conv3a_r)
        out_corr = self.build_corr(conv3a_l, conv3a_r, max_disp=40)
        out_corr = self.input_preprocessing(out_corr, device)
        # out_corr = self.l1(out_corr)
        conv3a_l = self.input_preprocessing(conv3a_l, device)
        out_conv3a_redir = self.res3(conv3a_l)
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

        conv3b = self.res3_1(in_conv3b)
        conv4a = self.res4(conv3b)
        conv4b = self.res4_1(conv4a)
        conv5a = self.res5(conv4b)
        conv5b = self.res5_1(conv5a)
        conv6a = self.res6(conv5b)
        conv6b = self.res6_1(conv6a)

        pr6 = self.pred_flow6(conv6b)
        upconv5 = self.upconv5(conv6b)
        # upconv5 = self.l1(upconv5_1)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5 = self.pred_flow5(iconv5)
        upconv4 = self.upconv4(iconv5)
        # upconv4 = self.l1(upconv4_1)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)

        pr4 = self.pred_flow4(iconv4)
        upconv3 = self.upconv3(iconv4)
        # upconv3 = self.l1(upconv3_1)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        upconv2 = self.upconv2(iconv3)
        # upconv2 = self.l1(upconv2_1)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        upconv1 = self.upconv1(iconv2)
        # upconv1 = self.l1(upconv1_1)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        upconv0 = self.upconv0(iconv1)
        # upconv0 = self.l1(upconv0_1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)

        pr0 = self.pred_flow0(iconv0)
        pr0 = self.relu(pr0)

        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)

        return disps
