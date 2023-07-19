import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_basicconv2d import (
    TtBasicConv2d,
)
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm


class TtMixed3a(nn.Module):
    def __init__(
        self,
        device,
        base_address,
        state_dict,
    ):
        super().__init__()
        self.device = device

        self.maxpool = fallback_ops.MaxPool2d(kernel_size=3, stride=2)

        self.conv = TtBasicConv2d(
            device=self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.conv",
            in_planes=64,
            out_planes=96,
            kernel_size=3,
            stride=2,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = fallback_ops.concat((x0, x1), 1)
        return out
