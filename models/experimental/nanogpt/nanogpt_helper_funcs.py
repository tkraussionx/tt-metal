# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional
from tt_lib import tensor
import tt_lib
from models.utility_functions import tt_to_torch_tensor, torch2tt_tensor, torch_to_tt_tensor_rm, tt2torch_tensor


def Linear(
    in_features: int,
    out_features: int,
    weight: tensor.Tensor,
    bias: Optional[tensor.Tensor] = None,
    output_mem_config=tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED,
        tt_lib.tensor.BufferType.DRAM,
    ),
    device=None,
):
    """
    Returns a function that performs a Linear operation with optional bias.
    ``weight`` must be tt_tensor.
    """
    assert weight.shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"

    if bias is not None:
        assert bias.shape()[-1] == out_features, "bias does not have the expected shape"

    weight = weight
    bias = bias
    weight_T = tensor.transpose(weight, -2, -1)
    if weight_T.dtype() == tt_lib.tensor.DataType.BFLOAT8_B:
        weight_T = tt_to_torch_tensor(weight_T)
        weight_T = torch_to_tt_tensor_rm(weight_T, device=device)
        if bias is not None:
            bias = tt_to_torch_tensor(bias)
            bias = torch_to_tt_tensor_rm(bias, device=device)

    def linear_(activation):
        assert activation.shape()[-1] == in_features, "activation tensor do not have the expected shape"
        output = tensor.matmul(activation, weight_T, output_mem_config)

        if bias is not None:
            output_plus_bias = tensor.bcast(
                output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H, output_mem_config
            )
            return output_plus_bias

        return output

    return linear_


def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.layout() == target_layout:
        return x
    if x.layout() == tt_lib.tensor.Layout.ROW_MAJOR and target_layout == tt_lib.tensor.Layout.TILE:
        x_padded_shape = tt_lib.tensor.pad_to_tile_shape(x.shape(), False, False, True, True)
        if x.shape() != x_padded_shape:
            return tt_lib.tensor.format_input_tensor(
                x, device, x_padded_shape, pad_value, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.tilize(x, output_mem_config, use_multicore=True)
    elif x.layout() == tt_lib.tensor.Layout.TILE and target_layout == tt_lib.tensor.Layout.ROW_MAJOR:
        if x.shape() != x.shape_without_padding():
            return tt_lib.tensor.format_output_tensor(
                x, x.shape_without_padding(), device, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.untilize(x, output_mem_config, use_multicore=True)
    else:
        assert False


def unpad_from_zero(x, desired_shape):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
            x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        x = x.unpad(
            (0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1)
        )
        x = x.to_torch().to(torch.float)
    return x
