import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
from python_api_testing.models.yolov7.reference.models.common import autopad
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtMaxPool2D(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        k,
        s,
        p=0,
    ):
        super().__init__()
        self.device = device
        self.host = tt_lib.device.GetHost()

        self.maxpool2d = nn.MaxPool2d(
            kernel_size=k,
            stride=s,
            padding=p,
        )

    def forward(self, x):
        x = tt2torch_tensor(x)
        x = self.maxpool2d(x)
        x = torch2tt_tensor(x, self.device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
        return x


class TtMP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        k=2,
    ):
        super().__init__()

        self.device = device
        self.base_address = base_address

        self.m = TtMaxPool2D(
            base_address=f"{base_address}.m",
            state_dict=state_dict,
            device=device,
            k=k,
            s=k,
        )

    def forward(self, x):
        return self.m(x)
