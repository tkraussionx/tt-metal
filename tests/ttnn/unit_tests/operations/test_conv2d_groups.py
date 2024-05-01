# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import torch.nn as nn
import pytest
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull, is_grayskull, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn
import tt_lib
import math
import os


def test_conv_grouped(
    device,
):
    # Generate pytorch golden grouped evaluation
    torch.manual_seed(0)

    # Test parameters
    batch_size = 1
    in_channels = 3
    input_height = 32
    input_width = 32
    out_channels = 6
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    num_groups = 1
    math_fidelity = ttnn.MathFidelity.HiFi4

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Define original tensors and shapes
    conv_input_shape = [batch_size, in_channels, input_height, input_width]
    conv_weight_shape = [out_channels, in_channels // num_groups, kernel_size[0], kernel_size[1]]
    conv_bias_shape = [1, 1, 1, out_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()

    # Define pytorch convolutional layer
    conv_layer_pytorch = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
    )
    conv_layer_pytorch.weight = nn.Parameter(torch_weight_tensor)
    conv_layer_pytorch.bias = nn.Parameter(torch_bias_tensor.reshape(-1))

    # Apply convolution operation
    torch_out_golden_tensor = conv_layer_pytorch(torch_input_tensor_nchw)

    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]

    # Generate ttnn grouped evaluation
    # Define ttnn tensors and shapes
    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
    tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16)

    # Define ttnn convolution operation
    conv_layer_ttnn = ttnn.Conv2d(
        device=device,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        input_height=input_height,
        input_width=input_width,
        math_fidelity=math_fidelity,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        deallocate_activation=False,
        use_shallow_conv_variant=False,
        use_1d_systolic_array=True,
        reader_patterns_cache={},
        weight=tt_weight_tensor,
        bias=tt_bias_tensor,
        conv_blocking_and_parallelization_config_override=None,
        enable_auto_formatting=False,
        padded_input_channels=None,
        compute_kernel_config=compute_kernel_config,
        output_layout=ttnn.TILE_LAYOUT,
        groups=num_groups,
    )

    # Convert torch input tensor to ttnn tensor
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    # Move input tensor to device
    tt_input_tensor_on_device = conv_layer_ttnn.copy_input_to_device(tt_input_tensor)

    # Apply convolution operation on device
    tt_output_tensor_on_device = conv_layer_ttnn(tt_input_tensor_on_device)

    # Convert output and get output from device
    tt_output_tensor_on_device = ttnn.to_layout(tt_output_tensor_on_device, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    torch_output_tensor = torch.split(torch_output_tensor, out_channels, 3)[0]
    torch_output_tensor = torch.reshape(torch_output_tensor, output_shape_nhwc)

    # Permute shape
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.999)
    logger.info(pcc_msg)
    assert passing
