from typing import List, Union, Optional
import torch.nn as nn
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor


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
