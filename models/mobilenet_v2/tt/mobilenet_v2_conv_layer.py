import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch
from typing import Optional, Union
from transformers import MobileNetV2Config
import tt_lib
from models.mobilenet_v2.tt.mobilenet_v2_conv2d import TtConv2D
from models.mobilenet_v2.mobilenet_v2_utils import apply_tf_padding
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtMobileNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        config: MobileNetV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
        layer_norm_eps: Optional[float] = None,
    ):
        super().__init__()

        self.device = device
        self.base_address = base_address
        self.config = config

        if in_channels % groups != 0:
            raise ValueError(
                f"Input channels ({in_channels}) are not divisible by {groups} groups."
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"Output channels ({out_channels}) are not divisible by {groups} groups."
            )

        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2) * dilation

        # Conv2D

        self.convolution = TtConv2D(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.convolution",
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            s=stride,
            p=padding,
            d=dilation,
            g=groups,
        )

        if use_normalization:
            self.bn_weight = torch2tt_tensor(
                state_dict[f"{base_address}.normalization.weight"],
                self.device,
                tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            )
            bn_bias_key = f"{base_address}.normalization.bias"

            if bn_bias_key in state_dict:
                self.bn_bias = torch2tt_tensor(
                    state_dict[bn_bias_key],
                    self.device,
                    tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
                )
            else:
                self.bn_bias = None

            r_m = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.normalization.running_mean"],
                self.device,
                put_on_device=False,
            )
            r_v = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.normalization.running_var"],
                self.device,
                put_on_device=False,
            )
            n_b_t = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.normalization.num_batches_tracked"],
                self.device,
                put_on_device=False,
            )

            self.normalization = fallback_ops.BatchNorm2d(
                weights=self.bn_weight,
                biases=self.bn_bias,
                num_features=out_channels,
                eps=config.layer_norm_eps if layer_norm_eps is None else layer_norm_eps,
                momentum=0.997,
                affine=True,
                track_running_stats=True,
                running_mean=r_m,
                running_var=r_v,
                num_batches_tracked=n_b_t,
            )

            self.normalization.eval()

        else:
            self.normalization = None

        if use_activation:
            if isinstance(use_activation, str):
                raise NotImplementedError
            else:
                self.activation = config.hidden_act
                if self.activation == "relu6":
                    self.activation = nn.ReLU6()
                else:
                    raise NotImplementedError
        else:
            self.activation = None

    def forward(self, features: tt_lib.tensor.Tensor):
        if self.config.tf_padding:
            features = apply_tf_padding(features, self.convolution)

        features = self.convolution(features)

        if self.normalization is not None:
            features = self.normalization(features)

        if self.activation is not None:
            features = tt2torch_tensor(features)
            features = self.activation(features)
            features = torch_to_tt_tensor_rm(features, self.device, put_on_device=False)

        return features
