from typing import List, Union, Optional
import torch.nn as nn
from loguru import logger

import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor


def Linear(
    in_features: int,
    out_features: int,
    weight: tt_lib.tensor.Tensor,
    bias: Optional[tt_lib.tensor.Tensor] = None,
):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be tt_tensor.
    """
    assert weight.shape() == [
        1,
        1,
        out_features,
        in_features,
    ], "weight does not have the expected shape"

    if bias is not None:
        assert bias.shape()[-1] == out_features, "bias does not have the expected shape"

    weight = weight
    bias = bias
    weight_T = tt_lib.tensor.transpose(weight)

    def linear_(activation):
        print(f"activation_shape: {activation.shape}")
        assert (
            activation.shape()[-1] == in_features
        ), "activation tensor do not have the expected shape"
        output = tt_lib.tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tt_lib.tensor.bcast(
                output, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
            )
            return output_plus_bias

        return output

    return linear_


class GetBatchNorm(nn.Module):
    def __init__(self, out_ch, state_dict, base_address, device=None):
        super(GetBatchNorm, self).__init__()
        self.weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.weight"], device, put_on_device=False
        )
        self.bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.bias"], device, put_on_device=False
        )
        self.running_mean = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.running_mean"], device, put_on_device=False
        )
        self.running_variance = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.running_var"], device, put_on_device=False
        )
        self.num_batches_tracked = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.num_batches_tracked"],
            device,
            put_on_device=False,
        )
        self.norm = tt_lib.fallback_ops.BatchNorm2d(
            self.weight,
            self.bias,
            self.running_mean,
            self.running_variance,
            self.num_batches_tracked,
            out_ch,
        )

    def forward(self, x: tt_lib.tensor.Tensor):
        return self.norm(x)
