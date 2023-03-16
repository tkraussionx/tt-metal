import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pymetal import ttmetal as ttm
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize, print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as TtLinear


class ResidualBlock(torch.nn.Module):
    #https://github.com/kjsman/stable-diffusion-pytorch/blob/8c6faa1b87e545b5ab840491f1b7952d803f54ef/stable_diffusion_pytorch/diffusion.py
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(1, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)
        print(time.shape, "time")
        print(feature.shape, "feature")
        time = time.unsqueeze(-1).unsqueeze(-1)
        print(time.shape, "after unsqueeze")
        merged = feature + time
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class TtResidualBlock(torch.nn.Module):
    def __init__(self,  in_channels, out_channels, device, n_time=1280, state_dict=None):
        super().__init__()
        # Note: Only caring about cases where in_channels == out_channels

        # Extract params from state dict
        # if state_dict != None:
        #     fc1_weight = pad_weight(state_dict["fc1.weight"])
        #     fc1_bias = pad_weight(state_dict["fc1.bias"])


        # else:

        #     fc1_weight = pad_weight(state_dict["fc1.weight"])
        #     fc1_bias = pad_weight(state_dict["fc1.bias"])

        # # Get shapes
        # fc1_weight_shape = fc1_weight.shape



        # # Tilize params
        # fc1_weight = tilize_to_list(fc1_weight)
        # fc1_bias = tilize_to_list(fc1_bias)


        ####### what to implement!

        self.torch_groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.torch_conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.linear_time_weight = torch.ones([1, 1, n_time, out_channels]).flatten().tolist()
        self.linear_time_bias = torch.zeros([1, 1, n_time, out_channels]).flatten().tolist()

        self.linear_time = TtLinear(n_time, out_channels, self.linear_time_weight, self.linear_time_bias, device)


        self.torch_groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.torch_conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # if in_channels == out_channels:
        #     self.residual_layer = nn.Identity()
        # else:
        #     self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        ####### What to implement as residual block!


    def SiLU(self, x):
        xs = ttm.tensor.sigmoid(x)
        xs = ttm.tensor.mul(xs, x)
        return xs


    def move_to_cpu(self, x):
        x_shape = x.shape()
        x = x.to(host).data()
        x = torch.tensor(x).reshape(x_shape)
        return untilize(x)


    def move_to_device(self, x):
        x_shape = x.shape
        x = ttm.tensor.Tensor(tilize_to_list(x), x_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
        return x


    def forward(self, feature, time):
        residue = feature # make a copy
        # move to cpu and
        print(feature.shape())

        feature = self.move_to_cpu(feature)
        feature = self.torch_groupnorm_feature(feature)
        feature = self.move_to_device(feature)

        # exec group norm on cpu
        # move from cpu to tensix

        feature = self.SiLU(feature)
        # move to cpu again
        # exec conv_feature
        feature = self.move_to_cpu(feature)
        feature = self.torch_conv_feature(feature)
        # move from CPU to tensix
        feature = self.move_to_device(feature)

        # all on tensix
        time = self.SiLU(time)
        time = self.linear_time(time)
        time = ttm.tensor.reshape(time, 32, 320, 32, 32)
        print("time in tt", time.shape())
        # time.unsqueeze(-1).unsqueeze(-1) # unnecessary since for tt modules time is 4d initially

        # merged = ttm.tensor.add(feature, time) # this is a broadcast?
        # merged = ttm.tensor.bcast(time, feature, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.HW)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)


        # move from tensix to CPU
        merged = self.move_to_cpu(merged)
        merged = self.groupnorm_merged(merged)

        merged = self.move_to_device(merged)
        # move back to tensix

        merged = self.SiLU(merged)
        # move from tensix to CPU

        merged = self.move_to_cpu(merged)

        merged = self.conv_merged(merged)

        merged = self.move_to_device(merged)

        return ttm.tensor.add(merged, residue)

        # return merged + self.residual_layer(residue)




if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    in_channel = 320
    out_channel = 320
    time_shape = [1, 1280]
    feature_shape = [2, 320, 64, 64]


    torch.manual_seed(123)
    time = torch.randn(time_shape)
    feature = torch.randn(feature_shape)


    torch_rb = ResidualBlock(in_channel, out_channel)
    torch_out = torch_rb(feature, time)

    print("pytorch result is ready!")


    tt_time_shape = [32, 32, 32, 1280]
    tt_time = torch.randn(tt_time_shape)

    tt_feature = ttm.tensor.Tensor(tilize_to_list(feature), feature_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_time = ttm.tensor.Tensor(tilize_to_list(tt_time), tt_time_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_rb = TtResidualBlock(in_channel, out_channel, device)
    tt_out = tt_rb(tt_feature, tt_time)

    tt_out = tt_out.to(host).data()
    tt_out = torch.Tensor(tt_out).reshape(torch_out.shape)
    tt_untilized_output = untilize(tt_out)
    print_diff_argmax(tt_untilized_output, torch_out)

    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)


    # compare results!
    # in_channel


    ttm.device.CloseDevice(device)


    # enable_compile_cache()
    # enable_binary_cache()
