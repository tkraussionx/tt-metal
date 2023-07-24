import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from loguru import logger
from models.unet.unet_mini_graphs import (
    TtUnetConv2D,
)
from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
)
from tt_lib.fallback_ops import fallback_ops
from models.unet.unet_utils import create_batchnorm2d
from models.unet.tt.unet_double_conv import TtDoubleConv


class TtDown(nn.Module):
    def __init__(
        self,
        device,
        base_address,
        state_dict,
        in_channels,
        out_channels,
        eps=1e-05,
        momentum=0.1,
    ) -> None:
        super().__init__()
        self.device = device

        # nn.MaxPool2d(2)
        self.tt_maxpool2d = fallback_ops.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
        )

        # DoubleConv(in_channels, out_channels)
        self.double_conv = TtDoubleConv(
            device, base_address, state_dict, in_channels, out_channels
        )

    def forward(self, x):
        # TT implementation ---------------------------------
        x = self.tt_maxpool2d(x)
        x = self.double_conv(x)

        return x
