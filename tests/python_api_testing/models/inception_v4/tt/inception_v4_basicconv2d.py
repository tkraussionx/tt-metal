import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from utility_functions_new import tt2torch_tensor
from tt_lib.fallback_ops import fallback_ops
from loguru import logger

from python_api_testing.models.inception_v4.tt.inception_v4_conv2d import (
    TtInterceptV4Conv2D,
)
from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.inception_v4.inception_v4_utils import create_batchnorm2d


class TtBasicConv2d(nn.Module):
    def __init__(
        self,
        device,
        hugging_face_reference_model,
        first_basic_conv2d_position,
        second_basic_conv2d_position=None,
        third_basic_conv2d_position=None,
    ) -> None:
        super().__init__()
        self.device = device

        if second_basic_conv2d_position is None:
            BasicConv2d = hugging_face_reference_model.features[
                first_basic_conv2d_position
            ]
            basic_conv2d_base_address = f"features.{first_basic_conv2d_position}.conv"
            batch_norm2d_base_address = f"features.{first_basic_conv2d_position}.bn"
        elif (second_basic_conv2d_position is not None) and (
            third_basic_conv2d_position is None
        ):
            BasicConv2d = hugging_face_reference_model.features[
                first_basic_conv2d_position
            ]
            BasicConv2d = getattr(BasicConv2d, second_basic_conv2d_position)
            basic_conv2d_base_address = f"features.{first_basic_conv2d_position}.{second_basic_conv2d_position}.conv"
            batch_norm2d_base_address = f"features.{first_basic_conv2d_position}.{second_basic_conv2d_position}.bn"
        else:
            BasicConv2d = hugging_face_reference_model.features[
                first_basic_conv2d_position
            ]
            BasicConv2d = getattr(BasicConv2d, second_basic_conv2d_position)
            BasicConv2d = getattr(BasicConv2d, third_basic_conv2d_position)
            basic_conv2d_base_address = f"features.{first_basic_conv2d_position}.{second_basic_conv2d_position}.{third_basic_conv2d_position}.conv"
            batch_norm2d_base_address = f"features.{first_basic_conv2d_position}.{second_basic_conv2d_position}.{third_basic_conv2d_position}.bn"

        # get parameters for the TT Conv2D
        in_channels = BasicConv2d.conv.in_channels
        out_channels = BasicConv2d.conv.out_channels
        kernel_size = BasicConv2d.conv.kernel_size[0]
        stride = BasicConv2d.conv.stride[0]
        padding = BasicConv2d.conv.padding[0]
        groups = BasicConv2d.conv.groups
        dilation = BasicConv2d.conv.dilation

        self.tt_basic_conv2d = TtInterceptV4Conv2D(
            state_dict=hugging_face_reference_model.state_dict(),
            base_address=basic_conv2d_base_address,
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
            batch_norm2d_base_address,
        )

        # logger
        # hugging_face_reference_model.eval()
        # state_dict = hugging_face_reference_model.state_dict()

        # self.batch_norm_2d = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.1, track_running_stats=False)
        # self.batch_norm_2d.weight = torch.nn.Parameter(state_dict[f"{batch_norm2d_base_address}.weight"])
        # self.batch_norm_2d.bias = torch.nn.Parameter(state_dict[f"{batch_norm2d_base_address}.bias"])
        # self.batch_norm_2d.running_mean = torch.nn.Parameter(state_dict[f"{batch_norm2d_base_address}.running_mean"])
        # self.batch_norm_2d.running_var = torch.nn.Parameter(state_dict[f"{batch_norm2d_base_address}.running_var"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TT implementation ---------------------------------
        x = self.tt_basic_conv2d(x)

        # x = tt2torch_tensor(x)
        # x = self.batch_norm_2d(x)
        # x = torch.nn.ReLU()(x)

        # x = torch2tt_tensor(x, self.device)

        x = self.tt_batch_norm_2d(x)
        x = tt_lib.tensor.relu(x)

        return x
