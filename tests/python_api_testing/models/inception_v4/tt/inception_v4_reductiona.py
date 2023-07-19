import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm


class TtReductionA(nn.Module):
    def __init__(self, device, base_address, state_dict):
        super().__init__()
        self.device = device

        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch0",
            in_planes=384,
            out_planes=96,
            kernel_size=1,
            stride=1,
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out
