import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch
from typing import Optional, Union
from transformers import MobileNetV2Config
import tt_lib
from models.mobilenet_v2.tt.mobilenet_v2_conv_layer import (
    TtMobileNetV2ConvLayer,
)
from models.mobilenet_v2.mobilenet_v2_utils import make_divisible
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtMobileNetV2InvertedResidual(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        config: MobileNetV2Config,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilation: int = 1,
    ):
        super().__init__()

        self.device = device
        self.base_address = base_address
        self.config = config

        expanded_channels = make_divisible(
            int(round(in_channels * config.expand_ratio)),
            config.depth_divisible_by,
            config.min_depth,
        )

        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand_1x1 = TtMobileNetV2ConvLayer(
            self.device,
            state_dict,
            f"{base_address}.expand_1x1",
            self.config,
            in_channels=in_channels,
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
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
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
        residual = features

        features = self.expand_1x1(features)  # 0.999991
        features = self.conv_3x3(features)  # 0.99963
        features = self.reduce_1x1(features)  # 0.99897

        return tt_lib.tensor.add(residual, features) if self.use_residual else features
