import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm


class TtInceptionA(nn.Module):
    def __init__(self, device, base_address, state_dict):
        super().__init__()
        self.device = device

        self.branch0 = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch0",
            in_planes=384,
            out_planes=96,
            kernel_size=1,
            stride=1,
        )

        self.branch1 = nn.Sequential(
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.0",
                in_planes=384,
                out_planes=64,
                kernel_size=1,
                stride=1,
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.1",
                in_planes=64,
                out_planes=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.branch2 = nn.Sequential(
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch2.0",
                in_planes=384,
                out_planes=64,
                kernel_size=1,
                stride=1,
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch2.1",
                in_planes=64,
                out_planes=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch2.2",
                in_planes=96,
                out_planes=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.branch3_a = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.branch3_b = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch3.1",
            in_planes=384,
            out_planes=96,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        torch_x = tt2torch_tensor(x)
        x3 = self.branch3_a(torch_x)
        x3 = torch_to_tt_tensor_rm(x3, self.device, put_on_device=False)
        x3 = self.branch3_b(x3)
        out = fallback_ops.concat((x0, x1, x2, x3), 1)
        return out
