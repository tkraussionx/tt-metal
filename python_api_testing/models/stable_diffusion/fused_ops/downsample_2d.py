from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from typing import Optional
from pymetal import ttlib as ttl
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as TtLinear
from pymetal.ttlib.fused_ops.softmax import softmax as TtSoftmax
from diffusers import StableDiffusionPipeline
from utils import move_to_cpu, move_to_device





class Downsample2D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels:
        padding:
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv", device=None, host=None, base_address=None, state_dict=None):
        super().__init__()
        self.device = device
        self.host=host
        self.base_address = base_address
        self.state_dict=state_dict
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
            conv.weight = nn.Parameter(self.state_dict[f"{base_address}.conv.weight"])
            conv.bias = nn.Parameter(self.state_dict[f"{base_address}.conv.bias"])
        else:
            assert self.channels == self.out_channels
            assert False, " we don't support AvgPool2d, and we should not need it either"
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape()[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = move_to_cpu(hidden_states, self.host)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
            hidden_states = move_to_device(hidden_states, self.device)

        assert hidden_states.shape[1] == self.channels
        hidden_states = move_to_cpu(hidden_states, self.host)
        hidden_states = self.conv(hidden_states)
        hidden_states = move_to_device(hidden_states)

        return hidden_states
