import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm


class TtMixed4a(nn.Module):
    def __init__(
        self,
        device,
        base_address,
        state_dict,
    ):
        super().__init__()

        self.branch0 = nn.Sequential(
            TtBasicConv2d(
                device=device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch0.0",
                in_planes=160,
                out_planes=64,
                kernel_size=1,
                stride=1,
            ),
            TtBasicConv2d(
                device=device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch0.1",
                in_planes=64,
                out_planes=96,
                kernel_size=3,
                stride=1,
            ),
        )

        self.branch1 = nn.Sequential(
            TtBasicConv2d(
                device=device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.0",
                in_planes=160,
                out_planes=64,
                kernel_size=1,
                stride=1,
            ),
            TtBasicConv2d(
                device=device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.1",
                in_planes=64,
                out_planes=64,
                kernel_size=(1, 7),
                stride=1,
                padding=(0, 3),
            ),
            TtBasicConv2d(
                device=device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.2",
                in_planes=64,
                out_planes=64,
                kernel_size=(7, 1),
                stride=1,
                padding=(3, 0),
            ),
            TtBasicConv2d(
                device=device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.3",
                in_planes=64,
                out_planes=96,
                kernel_size=(3, 3),
                stride=1,
            ),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = fallback_ops.concat((x0, x1), 1)
        return out
