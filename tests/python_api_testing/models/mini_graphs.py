from typing import List, Union, Optional
from tt_lib import tensor
from torch import nn
from loguru import logger

def Linear(in_features: int, out_features: int, weight: tensor.Tensor, bias: Optional[tensor.Tensor] = None):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be tt_tensor.
    """
    assert weight.shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"

    if bias is not None:
        assert bias.shape()[-1] == out_features, "bias does not have the expected shape"

    weight = weight
    bias = bias
    weight_T = tensor.transpose(weight)

    def linear_(activation):
        assert activation.shape()[-1] == in_features, "activation tensor do not have the expected shape"
        output = tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, weight: tensor.Tensor, bias: Optional[tensor.Tensor] = None, transpose_weights=True):

        assert weight.shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        if transpose_weights:
            self.weight_T = tensor.transpose(weight)
        else:
            self.weight_T = weight

    def forward(self, activation: tensor.Tensor) -> tensor.Tensor:
        assert activation.shape()[-1] == self.in_features, "activation tensor do not have the expected shape"
        output = tensor.matmul(activation, self.weight_T)

        if self.bias is not None:
            output_plus_bias = tensor.bcast(output, self.bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output
