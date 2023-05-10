from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from diffusers import StableDiffusionPipeline

from libs import tt_lib as ttl
from libs.tt_lib.fallback_ops import fallback_ops
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor, torch_to_tt_tensor_rm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc
from python_api_testing.models.stable_diffusion.mini_ops import Linear

from python_api_testing.models.stable_diffusion.utils import make_linear

class TtResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=1280,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-5,
        non_linearity="silu",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        state_dict=None,
        base_address= None,
        host = None,
        device = None
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

        norm1_weights = state_dict[f"{base_address}.norm1.weight"]
        norm1_bias = state_dict[f"{base_address}.norm1.bias"]
        self.norm1 = fallback_ops.GroupNorm(norm1_weights, norm1_bias, num_groups=groups, num_channels=self.in_channels, eps=eps, affine=True)


        conv1_weights = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = state_dict[f"{base_address}.conv1.bias"]
        self.conv1 = fallback_ops.Conv2d(conv1_weights, conv1_bias, self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)


        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            time_emb_proj_weights = state_dict[f"{base_address}.time_emb_proj.weight"]
            time_emb_proj_bias = state_dict[f"{base_address}.time_emb_proj.bias"]
            self.time_emb_proj = make_linear(in_features=temb_channels,
                                            out_features=time_emb_proj_out_channels,
                                            weights=time_emb_proj_weights,
                                            bias=time_emb_proj_bias,
                                            device=self.device)

        else:
            self.time_emb_proj = None

        norm2_weights = state_dict[f"{base_address}.norm2.weight"]
        norm2_bias = state_dict[f"{base_address}.norm2.bias"]


        self.norm2 = fallback_ops.GroupNorm(norm2_weights, norm2_bias, num_groups=groups, num_channels=self.out_channels, eps=eps, affine=True)

        conv2_weights = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = state_dict[f"{base_address}.conv2.bias"]

        self.conv2 = fallback_ops.Conv2d(conv2_weights, conv2_bias, self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)


        if non_linearity == "swish":

            self.nonlinearity = fallback_ops.silu
        elif non_linearity == "mish":
            assert False, "Mish is not implemented!"
        elif non_linearity == "silu":
            self.nonlinearity = fallback_ops.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            assert False, "Up block within residual block is not implemented!"
        elif self.down:
            assert False, "Down block within residual block is not implemented!"

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            conv_shortcut_weights = state_dict[f"{base_address}.conv_shortcut.weight"]
            conv_shortcut_bias = state_dict[f"{base_address}.conv_shortcut.bias"]
            self.conv_shortcut = fallback_ops.Conv2d(conv_shortcut_weights, conv_shortcut_bias, self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def  forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            assert False, "Upsample in residual block is not implemented!"
        elif self.downsample is not None:
            assert False, "Downsample in residual block is not implemented!"

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.nonlinearity(temb)

            temb = self.time_emb_proj(temb)
            temb = fallback_ops.reshape(temb, temb.shape()[2], temb.shape()[3], 1, 1)

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = ttl.tensor.bcast(hidden_states, temb, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW)

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            assert False, "Time Embedding Norm is not implemented"
            # To be removed in the next refactoring process
            # temb = tt_to_torch_tensor(temb, self.host)
            # scale, shift = torch.chunk(temb, 2, dim=1)
            # temb = torch_to_tt_tensor(temb, self.device)
            # shift = torch_to_tt_tensor(shift, self.device)
            # scale = torch_to_tt_tensor(scale, self.device)

            # ones = torch.ones(scale.shape)
            # ones = torch_to_tt_tensor(ones, self.device)

            # scale = ttl.tensor.add(ones, scale)
            # hidden_states = ttl.tensor.mul(hidden_states, scale)
            # hidden_states = ttl.tensor.add(hidden_states, shift)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)


        # create a tensor of size output_scale_factor
        output_sc_recip = 1 / self.output_scale_factor
        output_sc_recip = fallback_ops.full(input_tensor.shape(), output_sc_recip)
        output_tensor = ttl.tensor.add(input_tensor, hidden_states)
        output_tensor = ttl.tensor.mul(output_tensor, output_sc_recip)

        return output_tensor

def test_run_resnet_inference():
    # setup pytorch model
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    unet = pipe.unet
    unet.eval()
    state_dict = unet.state_dict()

    test = "test2"
    if test == "test1":
        unet_upblock = pipe.unet.up_blocks[2]
        resnet = unet_upblock.resnets[2]
        base_address="up_blocks.2.resnets.2"
        in_channels = resnet.conv1.in_channels
        out_channels = resnet.conv2.in_channels
        temb_channels = 512
        eps = 1e-05
        groups = 32
        torch_resnet = TorchResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=1280,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-6,
            non_linearity="silu",
            time_embedding_norm="default",
            kernel=None,
            output_scale_factor=1.0,
            use_in_shortcut=None,
            up=False,
            down=False,
            base_address=base_address,
            state_dict = state_dict)

        input_shape  = [1, in_channels, 32, 32]
        input = torch.randn(input_shape, dtype=torch.float32)
        temb = None


    if test == "test2":
        ############ start of residual block #############
        in_channels = 1280
        out_channels = 1280
        conv_shortcut = False
        dropout = 0
        temb_channels = 1280
        groups = 32
        groups_out = None
        pre_norm = True
        eps = 1e-05
        non_linearity = "silu"
        time_embedding_norm = "default"
        kernel = None
        output_scale_factor = 1
        use_in_shortcut = False
        up = False
        down = False
        ########## end of residual block #############
        hidden_states_shape = [2, 1280, 8, 8]
        temb_shape = [1, 1, 2, 1280]

        input = torch.randn(hidden_states_shape)
        temb = torch.randn(temb_shape)
        base_address="mid_block.resnets.0"
        resnet = pipe.unet.mid_block.resnets[0]

    # print(resnet)

    unet_out = resnet(input, temb.squeeze(0).squeeze(0))
    # torch_resnet_out = torch_resnet(input, None)

    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    ttl.device.SetDefaultDevice(device)
    host = ttl.device.GetHost()

    # setup tt model
    tt_resnet = TtResnetBlock2D(in_channels=in_channels,
                            out_channels=out_channels,
                            temb_channels=temb_channels,
                            groups=groups,
                            state_dict=state_dict,
                            base_address=base_address,
                            host=host,
                            device=device)

    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
    tt_temb = torch_to_tt_tensor_rm(temb, device, put_on_device=False)

    tt_out = tt_resnet(tt_input, temb)
    tt_out = tt_to_torch_tensor(tt_out, host)

    # print('unet vs torch')
    # print(comp_allclose_and_pcc(unet_out, torch_resnet_out))
    # print(unet_out.shape, tt_out.shape, "unet out and tt out")
    # print('unet vs tt')
    # print(comp_allclose_and_pcc(unet_out, tt_out))

    # print('torch vs tt')
    # print(comp_allclose_and_pcc(torch_resnet_out, tt_out))


if __name__ == "__main__":
    test_run_resnet_inference()
