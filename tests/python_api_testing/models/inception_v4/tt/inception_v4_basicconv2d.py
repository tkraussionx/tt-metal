import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from utility_functions_new import tt2torch_tensor
from tt_lib.fallback_ops import fallback_ops

from python_api_testing.models.inception_v4.tt.inception_v4_conv2d import (
    TtInterceptV4Conv2D,
)

from python_api_testing.models.inception_v4.inception_v4_utils import create_batchnorm2d


class TtIBasicConv2d(nn.Module):
    def __init__(
        self,
        device,
        hugging_face_reference_model,
        basic_block_position,
    ) -> None:
        super().__init__()
        self.device = device

        basic_conv2d = hugging_face_reference_model.features[basic_block_position]

        in_channels = basic_conv2d.conv.in_channels
        out_channels = basic_conv2d.conv.out_channels
        kernel_size = basic_conv2d.conv.kernel_size[0]
        stride = basic_conv2d.conv.stride[0]
        padding = basic_conv2d.conv.padding[0]
        groups = basic_conv2d.conv.groups
        dilation = basic_conv2d.conv.dilation

        self.tt_basic_conv2d = TtInterceptV4Conv2D(
            state_dict=hugging_face_reference_model.state_dict(),
            base_address=f"features.{basic_block_position}.conv",
            device=device,
            c1=in_channels,
            c2=out_channels,
            k=kernel_size,
            s=stride,
            p=padding,
            g=groups,
            d=dilation[0],
        )

        self.tt_batch_norm_2d = create_batchnorm2d(
            out_channels,
            hugging_face_reference_model.state_dict(),
            f"features.{basic_block_position}.bn",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TT implementation ---------------------------------
        x = self.tt_basic_conv2d(x)
        x = self.tt_batch_norm_2d(x)
        x = tt_lib.tensor.relu(x)

        return x
