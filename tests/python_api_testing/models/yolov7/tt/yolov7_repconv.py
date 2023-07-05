import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch
from python_api_testing.models.yolov7.tt.yolov7_conv2d import TtConv2D
from python_api_testing.models.yolov7.reference.models.common import autopad
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtRepConv(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        g=1,
        act=True,
        deploy=True,
    ):
        super().__init__()

        self.device = device
        self.base_address = base_address

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        if deploy:
            self.rbr_reparam = TtConv2D(
                base_address=f"{base_address}.rbr_reparam",
                state_dict=state_dict,
                device=device,
                c1=c1,
                c2=c2,
                k=k,
                s=s,
                p=autopad(k, p),
                g=g,
            )
        else:
            # Other ops (batchnorm) are not used in inference that's why we don't implement them here for now
            raise NotImplementedError

        self.act = act
        if self.act != True:
            logger.warning(
                f"Configuration for activation function {self.act} not supported. Using fallback.SiLU act function"
            )
            raise NotImplementedError

    def forward(self, x):
        return fallback_ops.silu(self.rbr_reparam(x))
