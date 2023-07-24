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
from models.unet.tt.unet_down import TtDown
from models.unet.tt.unet_up import TtUp
from models.unet.tt.unet_out_conv import TtOutConv


class TtUnet(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        n_channels,
        n_classes,
        bilinear=True,
        eps=1e-05,
        momentum=0.1,
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dict = state_dict

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        base_address = "inc.double_conv"
        self.inc = TtDoubleConv(
            self.device, base_address, self.state_dict, self.n_channels, 64
        )

        # self.down1 = Down(64, 128)
        down_position = "down1"
        base_address = down_position + ".maxpool_conv.1.double_conv"
        self.down1 = TtDown(self.device, base_address, self.state_dict, 64, 128)

        # self.down2 = Down(128, 256)
        down_position = "down2"
        base_address = down_position + ".maxpool_conv.1.double_conv"
        self.down2 = TtDown(self.device, base_address, self.state_dict, 128, 256)

        # self.down3 = Down(256, 512)
        down_position = "down3"
        base_address = down_position + ".maxpool_conv.1.double_conv"
        self.down3 = TtDown(self.device, base_address, self.state_dict, 256, 512)

        # bilinear
        factor = 2 if bilinear else 1

        # self.down4 = Down(512, 1024 // factor)
        down_position = "down4"
        base_address = down_position + ".maxpool_conv.1.double_conv"
        self.down4 = TtDown(
            self.device, base_address, self.state_dict, 512, 1024 // factor
        )

        # self.up1 = Up(1024, 512 // factor, bilinear)
        up_position = "up1"
        self.up1 = TtUp(
            self.device,
            up_position,
            self.state_dict,
            1024,
            512 // factor,
            self.bilinear,
        )

        # self.up2 = Up(512, 256 // factor, bilinear)
        up_position = "up2"
        self.up2 = TtUp(
            self.device, up_position, self.state_dict, 512, 256 // factor, self.bilinear
        )

        # self.up3 = Up(256, 128 // factor, bilinear)
        up_position = "up3"
        self.up3 = TtUp(
            self.device, up_position, self.state_dict, 256, 128 // factor, self.bilinear
        )

        # self.up4 = Up(128, 64, bilinear)
        up_position = "up4"
        self.up4 = TtUp(
            self.device, up_position, self.state_dict, 128, 64, self.bilinear
        )

        # self.outc = OutConv(64, n_classes)
        base_address = "outc.conv"
        self.outc = TtOutConv(
            self.device, base_address, self.state_dict, 64, self.n_classes
        )

    def forward(self, x):
        # TT implementation ---------------
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits
