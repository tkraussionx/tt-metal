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
from models.unet.unet_utils import create_batchnorm2d


class TtDoubleConv(nn.Module):
    def __init__(
        self,
        device,
        base_address,
        state_dict,
        in_channels,
        out_channels,
        mid_channels=None,
        eps=1e-05,
        momentum=0.1,
    ) -> None:
        super().__init__()
        self.device = device

        if not mid_channels:
            mid_channels = out_channels

        # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        conv2d_base_address_1 = f"{base_address}.0"
        conv2d_base_address_2 = f"{base_address}.3"
        batch_norm2d_base_address_1 = f"{base_address}.1"
        batch_norm2d_base_address_2 = f"{base_address}.4"

        self.tt_conv2d_1 = TtUnetConv2D(
            state_dict=state_dict,
            base_address=conv2d_base_address_1,
            device=device,
            c1=in_channels,
            c2=mid_channels,
            k=3,
            s=1,
            p=1,
        )

        # nn.BatchNorm2d(mid_channels)
        batch_norm2d_base_address = batch_norm2d_base_address_1
        self.tt_batch_norm_2d_1 = create_batchnorm2d(
            mid_channels,
            eps,
            momentum,
            state_dict,
            batch_norm2d_base_address,
        )
        self.tt_batch_norm_2d_1.eval()

        # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.tt_conv2d_2 = TtUnetConv2D(
            state_dict=state_dict,
            base_address=conv2d_base_address_2,
            device=device,
            c1=mid_channels,
            c2=out_channels,
            k=3,
            s=1,
            p=1,
        )

        # nn.BatchNorm2d(out_channels)
        batch_norm2d_base_address = batch_norm2d_base_address_2
        self.tt_batch_norm_2d_2 = create_batchnorm2d(
            out_channels,
            eps,
            momentum,
            state_dict,
            batch_norm2d_base_address,
        )
        self.tt_batch_norm_2d_2.eval()

    def forward(self, x):
        # TT implementation ---------------------------------
        x = self.tt_conv2d_1(x)
        x = self.tt_batch_norm_2d_1(x)
        x = tt_lib.tensor.relu(x)

        x = self.tt_conv2d_2(x)
        x = self.tt_batch_norm_2d_2(x)
        x = tt_lib.tensor.relu(x)

        return x
