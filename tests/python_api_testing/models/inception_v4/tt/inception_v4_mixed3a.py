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
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)

from python_api_testing.models.inception_v4.inception_v4_utils import create_batchnorm2d


class TtMixed3a(nn.Module):
    def __init__(
        self,
        device,
        hugging_face_reference_model,
        basic_conv2d_position,
    ) -> None:
        super().__init__()
        self.device = device

        second_level = "conv"

        self.tt_maxpool2d_1 = fallback_ops.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False
        )

        self.tt_basic_conv2d = TtBasicConv2d(
            device,
            hugging_face_reference_model,
            basic_conv2d_position,
            second_level,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TT implementation ---------------------------------
        x = self.tt_maxpool2d_1(x)
        x = self.tt_basic_conv2d(x)

        return x
