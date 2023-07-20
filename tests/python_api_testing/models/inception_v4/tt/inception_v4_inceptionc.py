import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm


class TtInceptionC(nn.Module):
    def __init__(self, device, base_address, state_dict):
        super().__init__()
        self.device = device

        self.branch0 = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch0",
            in_planes=1536,
            out_planes=256,
            kernel_size=1,
            stride=1,
        )

        self.branch1_0 = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch1_0",
            in_planes=1536,
            out_planes=384,
            kernel_size=1,
            stride=1,
        )
        self.branch1_1a = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch1_1a",
            in_planes=384,
            out_planes=256,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )
        self.branch1_1b = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch1_1b",
            in_planes=384,
            out_planes=256,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )

        self.branch2_0 = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch2_0",
            in_planes=1536,
            out_planes=384,
            kernel_size=1,
            stride=1,
        )
        self.branch2_1 = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch2_1",
            in_planes=384,
            out_planes=448,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )
        self.branch2_2 = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch2_2",
            in_planes=448,
            out_planes=512,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )
        self.branch2_3a = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch2_3a",
            in_planes=512,
            out_planes=256,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )
        self.branch2_3b = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch2_3b",
            in_planes=512,
            out_planes=256,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )

        self.branch3_a = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.branch3_b = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.branch3.1",
            in_planes=1536,
            out_planes=256,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = fallback_ops.concat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = fallback_ops.concat((x2_3a, x2_3b), 1)

        torch_x = tt2torch_tensor(x)
        x3 = self.branch3_a(torch_x)
        x3 = torch_to_tt_tensor_rm(x3, self.device, put_on_device=False)
        x3 = self.branch3_b(x3)

        out = fallback_ops.concat((x0, x1, x2, x3), 1)
        return out
