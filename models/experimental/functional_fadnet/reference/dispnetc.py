# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from models.experimental.functional_fadnet import ResBlock


class DispNetC(nn.Module):
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

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(6, 64, 7, 2, bias=True)
        self.l1 = nn.LeakyReLU(0.1, inplace=True)

        self.relu = nn.ReLU(inplace=False)

        self.res1 = ResBlock(64, 128, stride=2)
        self.res2 = ResBlock(128, 256, stride=2)
        self.res3 = ResBlock(256, 32, stride=1)
        # LR

        self.res3_1 = ResBlock(72, 256)
        self.res4 = ResBlock(256, 512, stride=2)
        self.res4_1 = ResBlock(512, 512)
        self.res5 = ResBlock(512, 512, stride=2)
        self.res5_1 = ResBlock(512, 512)
        self.res6 = ResBlock(512, 1024, stride=2)
        self.res6_1 = ResBlock(1024, 1024)

        self.pred_flow6 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(769, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(385, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(193, 64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(97, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(23, 16, 3, 1, 1)

        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        # LR
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        # LR
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        # LR
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        # LR
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        # LR
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv0 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        # LR
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        self.pred_flow0 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input: torch.Tensor):
        # split left image and right image
        imgs = torch.chunk(input, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]

        conv1_l = self.c1(img_left)
        conv1_lr = self.l1(conv1_l)
        conv2_l = self.res1(conv1_lr)
        conv3a_l = self.res2(conv2_l)

        conv1_r = self.c1(img_right)
        conv1_lr = self.l1(conv1_r)
        conv2_r = self.res1(conv1_lr)
        conv3a_r = self.res2(conv2_r)

        # Correlate corr3a_l and corr3a_r
        # out_corr = self.corr(conv3a_l, conv3a_r)
        out_corr = self.build_corr(conv3a_l, conv3a_r, max_disp=40)
        out_corr = self.l1(out_corr)
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
        upconv5_1 = self.upconv5(conv6b)
        upconv5 = self.l1(upconv5_1)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5 = self.pred_flow5(iconv5)
        upconv4_1 = self.upconv4(iconv5)
        upconv4 = self.l1(upconv4_1)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)

        pr4 = self.pred_flow4(iconv4)
        upconv3_1 = self.upconv3(iconv4)
        upconv3 = self.l1(upconv3_1)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        upconv2_1 = self.upconv2(iconv3)
        upconv2 = self.l1(upconv2_1)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        upconv1_1 = self.upconv1(iconv2)
        upconv1 = self.l1(upconv1_1)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        upconv0_1 = self.upconv0(iconv1)
        upconv0 = self.l1(upconv0_1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)

        pr0 = self.pred_flow0(iconv0)
        pr0 = self.relu(pr0)

        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)

        return disps
