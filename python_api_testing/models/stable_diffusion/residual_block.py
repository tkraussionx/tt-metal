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

from diffusers import StableDiffusionPipeline

from pymetal import ttlib as ttl
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize, print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as TtLinear
from fused_ops.group_norm2d import TtGroupNorm2D
from python_api_testing.fused_ops.silu import SiLU as TtSiLU
from utils import move_to_cpu, move_to_device
'''
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
        xs = ttl.tensor.sigmoid(x)
        xs = ttl.tensor.mul(xs, x)
        return xs


    def move_to_cpu(self, x):
        x_shape = x.shape()
        x = x.to(host).data()
        x = torch.tensor(x).reshape(x_shape)
        return untilize(x)


    def move_to_device(self, x):
        x_shape = x.shape
        x = ttl.tensor.Tensor(tilize_to_list(x), x_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
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
        time = ttl.tensor.reshape(time, 32, 320, 32, 32)
        print("time in tt", time.shape())
        # time.unsqueeze(-1).unsqueeze(-1) # unnecessary since for tt modules time is 4d initially

        # merged = ttl.tensor.add(feature, time) # this is a broadcast?
        # merged = ttl.tensor.bcast(time, feature, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW)

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

        return ttl.tensor.add(merged, residue)

        # return merged + self.residual_layer(residue)

'''


class TtResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        device=None,
        host=None,
        state_dict=None,
        base_address="encoder.mid_block.resnets.0"
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True # this is part of the original code
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.device = device
        self.host = host

        if groups_out is None:
            groups_out = groups

        self.ttnorm1 = TtGroupNorm2D(num_groups=groups, num_channels=in_channels, epsf=eps, device=device, host=host)
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)


        norm1_weights = state_dict[f"{base_address}.norm1.weight"]
        norm1_bias = state_dict[f"{base_address}.norm1.bias"]
        self.norm1.weight = nn.Parameter(norm1_weights)
        self.norm1.bias = nn.Parameter(norm1_bias)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) # TODO: we dont have support, so using torch

        conv1_weights = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = state_dict[f"{base_address}.conv1.bias"]
        self.conv1.weight = nn.Parameter(conv1_weights)
        self.conv1.bias = nn.Parameter(conv1_bias)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.time_emb_proj.weight"]))
            bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.time_emb_proj.bias"]))
            self.time_emb_proj = TtLinear(temb_channels, time_emb_proj_out_channels, weights, bias)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)


        norm2_weights = state_dict[f"{base_address}.norm2.weight"]
        norm2_bias = state_dict[f"{base_address}.norm2.bias"]
        self.norm2.weight = nn.Parameter(norm2_weights)
        self.norm2.bias = nn.Parameter(norm2_bias)

        # self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


        conv2_weights = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = state_dict[f"{base_address}.conv2.bias"]
        self.conv1.weight = nn.Parameter(conv2_weights)
        self.conv1.bias = nn.Parameter(conv2_bias)

        if non_linearity == "swish":
            self.nonlinearity = TtSiLU
        elif non_linearity == "mish":
            assert False, "mish is not implemented!"
            # self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = TtSiLU

        self.upsample = self.downsample = None
        if self.up:
            assert False, "we do not have tests that required this yet"
            # if kernel == "fir":
            #     fir_kernel = (1, 3, 3, 1)
            #     self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            # elif kernel == "sde_vp":
            #     self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            # else:
            #     self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            assert False, "we do not have tests that required this yet"
            # if kernel == "fir":
            #     fir_kernel = (1, 3, 3, 1)
            #     self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            # elif kernel == "sde_vp":
            #     self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            # else:
            #     self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")


        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) # TODO

    def  forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = move_to_cpu(hidden_states, self.host)
        hidden_states = self.norm1(hidden_states)
        hidden_states = move_to_device(hidden_states, device=self.device)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            assert False, "we do not support upsample in resnet"
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            # if hidden_states.shape[0] >= 64:
            #     input_tensor = input_tensor.contiguous()
            #     hidden_states = hidden_states.contiguous()
            # input_tensor = self.upsample(input_tensor)
            # hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            assert False, "we do not support downsample in resnet"
            # input_tensor = self.downsample(input_tensor)
            # hidden_states = self.downsample(hidden_states)


        hidden_states = move_to_cpu(hidden_states, self.host)
        hidden_states = self.conv1(hidden_states)
        hidden_states = move_to_device(hidden_states, device=self.device)

        if temb is not None:
            assert False, "not tested since we dont have tests for it yet"
            temp = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = ttl.tensor.add(hidden_states, temb)

        hidden_states = move_to_cpu(hidden_states, self.host)
        hidden_states = self.norm2(hidden_states)
        hidden_states = move_to_device(hidden_states, device=self.device)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            assert False, "this is support but not tested!"
            temb = move_to_cpu(temb, self.host)
            scale, shift = torch.chunk(temb, 2, dim=1)
            temb = move_to_device(temb, self.device)
            shift = move_to_device(shift, self.device)
            scale = move_to_device(scale, self.device)

            ones = torch.ones(scale.shape)
            ones = move_to_device(ones, self.device)

            scale = ttl.tensor.add(ones, scale)
            hidden_states = ttl.tensor.mul(hidden_states, scale)
            hidden_states = ttl.tensor.add(hidden_states, shift)

        hidden_states = self.nonlinearity(hidden_states)

        # hidden_states = self.dropout(hidden_states)
        hidden_states = move_to_cpu(hidden_states, self.host)
        hidden_states = self.conv2(hidden_states)
        hidden_states = move_to_device(hidden_states, device=self.device)

        if self.conv_shortcut is not None:
            input_tensor = move_to_cpu(input_tensor, self.host)
            input_tensor = self.conv_shortcut(input_tensor)
            input_tensor = move_to_device(input_tensor, device=self.device)


        # create a tensor of size output_scale_factor

        output_sc_recip = 1 / self.output_scale_factor
        output_sc_recip = torch.full(input_tensor.shape(), output_sc_recip)
        output_sc_recip = move_to_device(output_sc_recip, self.device)

        output_tensor = ttl.tensor.add(input_tensor, hidden_states)
        output_tensor = ttl.tensor.mul(output_tensor, output_sc_recip)


        return output_tensor



'''
if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

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

    tt_feature = ttl.tensor.Tensor(tilize_to_list(feature), feature_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    tt_time = ttl.tensor.Tensor(tilize_to_list(tt_time), tt_time_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    tt_rb = TtResidualBlock(in_channel, out_channel, device)
    tt_out = tt_rb(tt_feature, tt_time)

    tt_out = tt_out.to(host).data()
    tt_out = torch.Tensor(tt_out).reshape(torch_out.shape)
    tt_untilized_output = untilize(tt_out)
    print_diff_argmax(tt_untilized_output, torch_out)

    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)


    # compare results!
    # in_channel


    ttl.device.CloseDevice(device)


    # enable_compile_cache()
    # enable_binary_cache()
'''

def run_resnet_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)

    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    vae_encoder = pipe.vae.encoder
    resnet = vae_encoder.mid_block.resnets[0]
    resnet.norm1 = resnet.norm1.to('cpu')
    in_channels = 512
    temb_channels = None
    eps = 1e-06
    resnet_groups = 32



    input_shape  = [1, 512, 32, 32]
    input = torch.randn(input_shape, dtype=torch.float16)
    print("type ", resnet.norm1.weight.dtype)
    print("input type", input.dtype)
    print("resnet device", resnet.norm1.weight.device)
    # temb_shape = [1, 1, 1, 1280]
    # temp = torch.randn(temb_shape)


    torch_out = resnet(input, None)

    tt_input = ttl.tensor.Tensor(tilize_to_list(input), input_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    tt_resnet = TtResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, groups=resnet_groups,  state_dict=state_dict, device=device, host=host)

    tt_out = tt_resnet(tt_input, None).to(host).data()

    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_resnet_inference(device)
    ttl.device.CloseDevice(device)
