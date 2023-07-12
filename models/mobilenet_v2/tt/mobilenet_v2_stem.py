import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch
from typing import Optional, Union
from transformers import MobileNetV2Config
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from models.mobilenet_v2.tt.mobilenet_v2_conv_layer import (
    TtMobileNetV2ConvLayer,
)
from models.mobilenet_v2.mobilenet_v2_utils import make_divisible
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtMobileNetV2Stem(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        config: MobileNetV2Config,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.device = device
        self.base_address = base_address
        self.config = config

        # The very first layer is a regular 3x3 convolution with stride 2 that expands to 32 channels.
        # All other expansion layers use the expansion factor to compute the number of output channels.
        self.first_conv = TtMobileNetV2ConvLayer(
            self.device,
            state_dict,
            f"{base_address}.first_conv",
            self.config,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=2,
        )

        if config.first_layer_is_expansion:
            self.expand_1x1 = None
        else:
            self.expand_1x1 = TtMobileNetV2ConvLayer(
                self.device,
                state_dict,
                f"{base_address}.expand_1x1",
                self.config,
                in_channels=expanded_channels,
                out_channels=expanded_channels,
                kernel_size=1,
            )

        self.conv_3x3 = TtMobileNetV2ConvLayer(
            self.device,
            state_dict,
            f"{base_address}.conv_3x3",
            self.config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=1,
            groups=expanded_channels,
        )

        self.reduce_1x1 = TtMobileNetV2ConvLayer(
            self.device,
            state_dict,
            f"{base_address}.reduce_1x1",
            self.config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, features: tt_lib.tensor.Tensor):
        features = self.first_conv(features)
        if self.expand_1x1 is not None:
            features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return features
