import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm


class TtReductionB(nn.Module):
    def __init__(self, device, base_address, state_dict):
        super().__init__()
        self.device = device

        self.branch0 = nn.Sequential(
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch0.0",
                in_planes=1024,
                out_planes=192,
                kernel_size=1,
                stride=1,
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch0.1",
                in_planes=192,
                out_planes=192,
                kernel_size=3,
                stride=2,
            ),
        )

        self.branch1 = nn.Sequential(
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.0",
                in_planes=1024,
                out_planes=256,
                kernel_size=1,
                stride=1,
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.1",
                in_planes=256,
                out_planes=256,
                kernel_size=(1, 7),
                stride=1,
                padding=(0, 3),
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.2",
                in_planes=256,
                out_planes=320,
                kernel_size=(7, 1),
                stride=1,
                padding=(3, 0),
            ),
            TtBasicConv2d(
                device=self.device,
                state_dict=state_dict,
                base_address=f"{base_address}.branch1.3",
                in_planes=320,
                out_planes=320,
                kernel_size=3,
                stride=2,
            ),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        torch_x = tt2torch_tensor(x)
        x2 = self.branch2(torch_x)
        x2 = torch_to_tt_tensor_rm(x2, self.device, put_on_device=False)
        out = fallback_ops.concat((x0, x1, x2), 1)
        return out
