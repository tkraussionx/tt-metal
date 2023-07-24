import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TtUp(nn.Module):
    def __init__(
        self,
        device,
        up_position,
        state_dict,
        in_channels,
        out_channels,
        bilinear=True,
        eps=1e-05,
        momentum=0.1,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_address = up_position + ".conv.double_conv"

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = TtDoubleConv(
                device,
                self.base_address,
                state_dict,
                in_channels,
                out_channels,
                in_channels // 2,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.up.weight = nn.Parameter(state_dict[f"{up_position}.up.weight"])
            self.up.bias = nn.Parameter(state_dict[f"{up_position}.up.bias"])
            self.conv = TtDoubleConv(
                device, self.base_address, state_dict, in_channels, out_channels
            )

    def forward(self, x1, x2):
        # TT implementation ---------------------------------
        x1 = tt2torch_tensor(x1)
        x2 = tt2torch_tensor(x2)

        x1 = self.up(x1)

        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)

        x = torch2tt_tensor(x, self.device)
        x = self.conv(x)

        return x
