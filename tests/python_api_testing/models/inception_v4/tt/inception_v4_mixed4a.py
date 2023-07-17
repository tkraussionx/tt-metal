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
from utility_functions_new import tt2torch_tensor


class TtMixed4a(nn.Module):
    def __init__(
        self,
        device,
        hugging_face_reference_model,
        eps=0.0001,
        momentum=0.1,
    ) -> None:
        super().__init__()
        self.device = device

        # branch0 -------------------------------------
        first_basic_conv2d_position = 4
        second_basic_conv2d_position = "branch0"
        third_basic_conv2d_position = "0"
        self.tt_basic_conv2d_b0_0 = TtBasicConv2d(
            device,
            hugging_face_reference_model,
            first_basic_conv2d_position,
            second_basic_conv2d_position,
            third_basic_conv2d_position,
            eps,
            momentum,
        )

        third_basic_conv2d_position = "1"
        self.tt_basic_conv2d_b0_1 = TtBasicConv2d(
            device,
            hugging_face_reference_model,
            first_basic_conv2d_position,
            second_basic_conv2d_position,
            third_basic_conv2d_position,
            eps,
            momentum,
        )

        # branch1 -------------------------------------
        first_basic_conv2d_position = 4
        second_basic_conv2d_position = "branch1"
        third_basic_conv2d_position = "0"
        self.tt_basic_conv2d_b1_0 = TtBasicConv2d(
            device,
            hugging_face_reference_model,
            first_basic_conv2d_position,
            second_basic_conv2d_position,
            third_basic_conv2d_position,
            eps,
            momentum,
        )

        third_basic_conv2d_position = "1"
        self.tt_basic_conv2d_b1_1 = TtBasicConv2d(
            device,
            hugging_face_reference_model,
            first_basic_conv2d_position,
            second_basic_conv2d_position,
            third_basic_conv2d_position,
            eps,
            momentum,
        )

        third_basic_conv2d_position = "2"
        self.tt_basic_conv2d_b1_2 = TtBasicConv2d(
            device,
            hugging_face_reference_model,
            first_basic_conv2d_position,
            second_basic_conv2d_position,
            third_basic_conv2d_position,
            eps,
            momentum,
        )

        third_basic_conv2d_position = "3"
        self.tt_basic_conv2d_b1_3 = TtBasicConv2d(
            device,
            hugging_face_reference_model,
            first_basic_conv2d_position,
            second_basic_conv2d_position,
            third_basic_conv2d_position,
            eps,
            momentum,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TT implementation ---------------------------------
        x0 = self.tt_basic_conv2d_b0_0(x)
        x0 = self.tt_basic_conv2d_b0_1(x0)
        x0 = tt2torch_tensor(x0)

        x1 = self.tt_basic_conv2d_b1_0(x)
        x1 = self.tt_basic_conv2d_b1_1(x1)
        x1 = self.tt_basic_conv2d_b1_2(x1)
        x1 = self.tt_basic_conv2d_b1_3(x1)
        x1 = tt2torch_tensor(x1)

        out = torch.cat((x0, x1), 1)

        return True
