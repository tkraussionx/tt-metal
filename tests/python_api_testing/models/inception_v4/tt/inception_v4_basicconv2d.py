import torch
import torch.nn as nn
import torch.nn.init as init
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.inception_v4.tt.inception_v4_conv2d import (
    TtInterceptV4Conv2D,
)
from utility_functions_new import tt2torch_tensor, torch_to_tt_tensor_rm


class TtBasicConv2d(nn.Module):
    def __init__(
        self,
        device,
        base_address,
        state_dict,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding=0,
    ):
        super().__init__()
        self.device = device

        self.conv = TtInterceptV4Conv2D(
            state_dict=state_dict,
            base_address=f"{base_address}.conv",
            device=self.device,
            c1=in_planes,
            c2=out_planes,
            k=kernel_size,
            s=stride,
            p=padding,
        )

        bn_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.bn.weight"], device, put_on_device=False
        )
        bn_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.bn.bias"], device, put_on_device=False
        )
        running_mean = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.bn.running_mean"], device, put_on_device=False
        )
        running_variance = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.bn.running_var"], device, put_on_device=False
        )
        num_batches_tracked = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.bn.num_batches_tracked"],
            device,
            put_on_device=False,
        )

        self.bn = fallback_ops.BatchNorm2d(
            weights=bn_weight,
            biases=bn_bias,
            running_mean=running_mean,
            running_var=running_variance,
            num_batches_tracked=num_batches_tracked,
            eps=0.001,
            num_features=out_planes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = tt_lib.tensor.relu(x)

        return x
