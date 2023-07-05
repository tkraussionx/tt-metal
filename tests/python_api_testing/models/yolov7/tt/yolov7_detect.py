import torch.nn as nn
import numpy as np
from loguru import logger
from pathlib import Path
import sys
import torch
from python_api_testing.models.yolov7.reference.models.common import autopad
from python_api_testing.models.yolov7.tt.yolov7_conv import TtConv2D
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(
        self, device, state_dict, base_address, nc=80, anchors=(), ch=(), inplace=True
    ):
        # detection layer
        super().__init__()

        self.device = device
        self.base_address = base_address

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)

        conv_list = []
        for i, x in enumerate(ch):
            conv_list.append(
                TtConv2D(
                    base_address=f"{base_address}.m.{i}",
                    state_dict=state_dict,
                    device=device,
                    c1=x,
                    c2=self.no * self.na,
                    k=1,
                )
            )

        self.m = nn.ModuleList(conv_list)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            x[i] = tt2torch_tensor(x[i])

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)

            # TODO:
            # Cannot be ported further UNTIL 5D tensors are supported for tt_lib ops

            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[
                    i
                ]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=z.device,
        )
        box @= convert_matrix
        return (box, score)
