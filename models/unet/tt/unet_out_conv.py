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


class TtOutConv(nn.Module):
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

        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        conv2d_base_address_1 = f"{base_address}"

        self.tt_conv2d = TtUnetConv2D(
            state_dict=state_dict,
            base_address=conv2d_base_address_1,
            device=device,
            c1=in_channels,
            c2=out_channels,
            k=1,
        )

    def forward(self, x):
        # TT implementation ---------------------------------
        x = self.tt_conv2d(x)

        return x
