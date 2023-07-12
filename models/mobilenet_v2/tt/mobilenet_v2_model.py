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
from models.mobilenet_v2.tt.mobilenet_v2_stem import (
    TtMobileNetV2Stem,
)
from models.mobilenet_v2.tt.mobilenet_v2_inverted_residual import (
    TtMobileNetV2InvertedResidual,
)
from models.mobilenet_v2.mobilenet_v2_utils import (
    make_divisible,
    apply_depth_multiplier,
)
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)
from dataclasses import dataclass


@dataclass
class TtBaseModelOutputWithPoolingAndNoAttention:
    last_hidden_state: tt_lib.tensor.Tensor = None
    pooler_output: tt_lib.tensor.Tensor = None
    hidden_states: tt_lib.tensor.Tensor = None


class TtMobileNetV2Model(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        config: MobileNetV2Config,
        add_pooling_layer: bool = True,
    ):
        """The bare MobileNetV2 model outputting raw hidden-states without any specific head on top."""
        super().__init__()

        self.device = device
        self.base_address = base_address
        self.config = config

        # Output channels for the projection layers
        channels = [
            16,
            24,
            24,
            32,
            32,
            32,
            64,
            64,
            64,
            64,
            96,
            96,
            96,
            160,
            160,
            160,
            320,
        ]
        channels = [apply_depth_multiplier(config, x) for x in channels]

        # Strides for the depthwise layers
        strides = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        if self.base_address != "":
            self.base_address = f"{self.base_address}."

        self.conv_stem = TtMobileNetV2Stem(
            base_address=f"{self.base_address}conv_stem",
            state_dict=state_dict,
            device=self.device,
            config=config,
            in_channels=config.num_channels,
            expanded_channels=apply_depth_multiplier(config, 32),
            out_channels=channels[0],
        )

        current_stride = 2  # first conv layer has stride 2
        dilation = 1

        self.layer = nn.ModuleList()
        for i in range(16):
            # Keep making the feature maps smaller or use dilated convolution?
            if current_stride == config.output_stride:
                layer_stride = 1
                layer_dilation = dilation
                dilation *= strides[i]  # larger dilation starts in next block
            else:
                layer_stride = strides[i]
                layer_dilation = 1
                current_stride *= layer_stride

            self.layer.append(
                TtMobileNetV2InvertedResidual(
                    base_address=f"{self.base_address}layer.{i}",
                    state_dict=state_dict,
                    device=self.device,
                    config=config,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=layer_stride,
                    dilation=layer_dilation,
                )
            )

        if config.finegrained_output and config.depth_multiplier < 1.0:
            output_channels = 1280
        else:
            output_channels = apply_depth_multiplier(config, 1280)

        self.conv_1x1 = TtMobileNetV2ConvLayer(
            base_address=f"{self.base_address}conv_1x1",
            state_dict=state_dict,
            device=self.device,
            config=config,
            in_channels=channels[-1],
            out_channels=output_channels,
            kernel_size=1,
        )

        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def forward(
        self,
        pixel_values: Optional[tt_lib.tensor.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, TtBaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.conv_stem(pixel_values)

        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        last_hidden_state = self.conv_1x1(hidden_states)

        if self.pooler is not None:
            torch_last_hidden_state = tt2torch_tensor(last_hidden_state)
            torch_pooled_output = torch.flatten(
                self.pooler(torch_last_hidden_state), start_dim=1
            )
            pooled_output = torch_to_tt_tensor_rm(
                torch_pooled_output, self.device, put_on_device=False
            )
        else:
            pooled_output = None

        if not return_dict:
            return tuple(
                v
                for v in [last_hidden_state, pooled_output, all_hidden_states]
                if v is not None
            )

        return TtBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )
