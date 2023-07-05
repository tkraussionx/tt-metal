import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch
from python_api_testing.models.yolov7.tt.yolov7_conv import TtConv
from python_api_testing.models.yolov7.reference.models.common import autopad
from python_api_testing.models.yolov7.tt.yolov7_mp import TtMaxPool2D
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtSPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5,
        k=(5, 9, 13),
    ):
        super().__init__()
        self.device = device
        self.base_address = base_address

        c_ = int(2 * c2 * e)  # hidden channels

        self.cv1 = TtConv(
            base_address=f"{base_address}.cv1",
            state_dict=state_dict,
            device=device,
            c1=c1,
            c2=c_,
            k=1,
            s=1,
        )
        self.cv2 = TtConv(
            base_address=f"{base_address}.cv2",
            state_dict=state_dict,
            device=device,
            c1=c1,
            c2=c_,
            k=1,
            s=1,
        )
        self.cv3 = TtConv(
            base_address=f"{base_address}.cv3",
            state_dict=state_dict,
            device=device,
            c1=c_,
            c2=c_,
            k=3,
            s=1,
        )
        self.cv4 = TtConv(
            base_address=f"{base_address}.cv4",
            state_dict=state_dict,
            device=device,
            c1=c_,
            c2=c_,
            k=1,
            s=1,
        )
        self.m = nn.ModuleList(
            [
                TtMaxPool2D(device, state_dict, base_address, k=x, s=1, p=x // 2)
                for x in k
            ]
        )
        self.cv5 = TtConv(
            base_address=f"{base_address}.cv5",
            state_dict=state_dict,
            device=device,
            c1=4 * c_,
            c2=c_,
            k=1,
            s=1,
        )
        self.cv6 = TtConv(
            base_address=f"{base_address}.cv6",
            state_dict=state_dict,
            device=device,
            c1=c_,
            c2=c_,
            k=3,
            s=1,
        )
        self.cv7 = TtConv(
            base_address=f"{base_address}.cv7",
            state_dict=state_dict,
            device=device,
            c1=2 * c_,
            c2=c2,
            k=1,
            s=1,
        )

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = tt_lib.fallback_ops.concat([x1] + [m(x1) for m in self.m], 1)
        y1 = self.cv6(self.cv5(x2))
        y2 = self.cv2(x)
        y3 = tt_lib.fallback_ops.concat((y1, y2), dim=1)
        return self.cv7(y3)
