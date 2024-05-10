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


@pytest.mark.parametrize("device_l1_small_size", [16384], indirect=True)
def test_conv_normal(
    device,
):
    # Generate pytorch golden grouped evaluation
    torch.manual_seed(0)

    # Test parameters
    batch_size = 1
    in_channels = 64
    input_height = 8
    input_width = 8
    out_channels = 64
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

    # Generate normal convolution via ttnn
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
    # torch_output_tensor = torch.split(torch_output_tensor, out_channels, 3)[0]
    torch_output_tensor = torch.reshape(torch_output_tensor, output_shape_nhwc)

    # Permute shape
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.999)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("device_l1_small_size", [16384], indirect=True)
def test_pytorch_conv_grouped(
    device,
):
    # Compute groups=2
    # Test parameters
    batch_size_groups_2 = 1
    in_channels_groups_2 = 64
    input_height_groups_2 = 8
    input_width_groups_2 = 8
    out_channels_groups_2 = 64
    kernel_size_groups_2 = (3, 3)
    stride_groups_2 = (1, 1)
    padding_groups_2 = (1, 1)
    num_groups_groups_2 = 2

    # Define original tensors and shapes
    conv_input_shape_groups_2 = [batch_size_groups_2, in_channels_groups_2, input_height_groups_2, input_width_groups_2]
    conv_weight_shape_groups_2 = [
        out_channels_groups_2,
        in_channels_groups_2 // num_groups_groups_2,
        kernel_size_groups_2[0],
        kernel_size_groups_2[1],
    ]
    conv_bias_shape_groups_2 = [1, 1, 1, out_channels_groups_2]
    torch_input_tensor_nchw_groups_2 = torch.randn(conv_input_shape_groups_2, dtype=torch.bfloat16).float()
    torch_input_tensor_groups_2 = torch.permute(torch_input_tensor_nchw_groups_2, (0, 2, 3, 1))
    torch_weight_tensor_groups_2 = torch.randn(conv_weight_shape_groups_2, dtype=torch.bfloat16).float()
    torch_bias_tensor_groups_2 = torch.randn(conv_bias_shape_groups_2, dtype=torch.bfloat16).float()

    # Define pytorch convolutional layer
    conv_layer_pytorch_groups_2 = nn.Conv2d(
        in_channels=in_channels_groups_2,
        out_channels=out_channels_groups_2,
        kernel_size=kernel_size_groups_2,
        stride=stride_groups_2,
        padding=padding_groups_2,
        groups=num_groups_groups_2,
    )
    conv_layer_pytorch_groups_2.weight = nn.Parameter(torch_weight_tensor_groups_2)
    conv_layer_pytorch_groups_2.bias = nn.Parameter(torch_bias_tensor_groups_2.reshape(-1))

    # Apply convolution operation
    torch_out_golden_tensor_groups_2 = conv_layer_pytorch_groups_2(torch_input_tensor_nchw_groups_2)

    output_shape_nhwc = [
        torch_out_golden_tensor_groups_2.shape[0],
        torch_out_golden_tensor_groups_2.shape[2],
        torch_out_golden_tensor_groups_2.shape[3],
        torch_out_golden_tensor_groups_2.shape[1],
    ]

    # Pad the weight and compute with groups=1, this should equal groups=2
    # Test parameters
    batch_size_groups_1 = 1
    in_channels_groups_1 = 64
    input_height_groups_1 = 8
    input_width_groups_1 = 8
    out_channels_groups_1 = 64
    kernel_size_groups_1 = (3, 3)
    stride_groups_1 = (1, 1)
    padding_groups_1 = (1, 1)
    num_groups_groups_1 = 1

    # Define original tensors and shapes
    torch_input_tensor_nchw_groups_1 = torch_input_tensor_nchw_groups_2

    tensor1, tensor2 = torch.chunk(torch_weight_tensor_groups_2, 2, dim=0)
    zero_tensor = torch.zeros_like(tensor1)
    tensor1_resized = torch.cat((tensor1, zero_tensor), dim=1)
    tensor2_resized = torch.cat((zero_tensor, tensor2), dim=1)
    torch_weight_tensor_groups_1 = torch.cat((tensor1_resized, tensor2_resized), dim=0)
    torch_bias_tensor_groups_1 = torch_bias_tensor_groups_2

    # Define pytorch convolutional layer
    conv_layer_pytorch_groups_1 = nn.Conv2d(
        in_channels=in_channels_groups_1,
        out_channels=out_channels_groups_1,
        kernel_size=kernel_size_groups_1,
        stride=stride_groups_1,
        padding=padding_groups_1,
        groups=num_groups_groups_1,
    )
    conv_layer_pytorch_groups_1.weight = nn.Parameter(torch_weight_tensor_groups_1)
    conv_layer_pytorch_groups_1.bias = nn.Parameter(torch_bias_tensor_groups_1.reshape(-1))

    # Apply convolution operation
    torch_out_golden_tensor_groups_1 = conv_layer_pytorch_groups_1(torch_input_tensor_nchw_groups_1)

    output_shape_nhwc = [
        torch_out_golden_tensor_groups_1.shape[0],
        torch_out_golden_tensor_groups_1.shape[2],
        torch_out_golden_tensor_groups_1.shape[3],
        torch_out_golden_tensor_groups_1.shape[1],
    ]

    passing, pcc_msg = check_with_pcc_without_tensor_printout(
        torch_out_golden_tensor_groups_1, torch_out_golden_tensor_groups_2, pcc=0.99999
    )
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("device_l1_small_size", [16384], indirect=True)
def test_pytorch_ttnn_conv_grouped(
    device,
):
    # Pad the weight and compute with groups=1, this should equal groups=2
    # Test parameters
    batch_size = 1
    in_channels = 64
    input_height = 8
    input_width = 8
    out_channels = 64
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
    conv_weight_shape = [
        out_channels,
        in_channels // 2,
        kernel_size[0],
        kernel_size[1],
    ]
    conv_bias_shape = [1, 1, 1, out_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()

    # Convert weights
    tensor1, tensor2 = torch.chunk(torch_weight_tensor, 2, dim=0)
    zero_tensor = torch.zeros_like(tensor1)
    tensor1_resized = torch.cat((tensor1, zero_tensor), dim=1)
    tensor2_resized = torch.cat((zero_tensor, tensor2), dim=1)
    torch_weight_tensor = torch.cat((tensor1_resized, tensor2_resized), dim=0)
    torch_bias_tensor = torch_bias_tensor

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
    # torch_output_tensor = torch.split(torch_output_tensor, out_channels, 3)[0]
    torch_output_tensor = torch.reshape(torch_output_tensor, output_shape_nhwc)

    # Permute shape
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.999)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("device_l1_small_size", [16384], indirect=True)
def test_ttnn_groups(
    device,
):
    # torch implementation
    # Test parameters
    batch_size = 1
    in_channels = 64
    input_height = 8
    input_width = 8
    out_channels = 64
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    num_groups = 2
    math_fidelity = ttnn.MathFidelity.HiFi4

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Define original tensors and shapes
    input_shape = [batch_size, in_channels, input_height, input_width]
    weight_shape = [
        out_channels,
        in_channels // num_groups,
        kernel_size[0],
        kernel_size[1],
    ]
    bias_shape = [1, 1, 1, out_channels]

    torch_input_tensor_nchw = torch.randn(input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor_nhwc = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(bias_shape, dtype=torch.bfloat16).float()

    # Define pytorch convolutional layer
    torch_conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
    )
    torch_conv_layer.weight = nn.Parameter(torch_weight_tensor)
    torch_conv_layer.bias = nn.Parameter(torch_bias_tensor.reshape(-1))

    # Apply convolution operation
    torch_output_tensor_nchw = torch_conv_layer(torch_input_tensor_nchw)

    # ttnn implementation
    # Define ttnn tensors and shapes
    ttnn_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
    ttnn_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16)

    # Define ttnn convolution operation
    ttnn_conv_layer = ttnn.Conv2d(
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
        weight=ttnn_weight_tensor,
        bias=ttnn_bias_tensor,
        conv_blocking_and_parallelization_config_override=None,
        enable_auto_formatting=False,
        padded_input_channels=None,
        compute_kernel_config=compute_kernel_config,
        output_layout=ttnn.TILE_LAYOUT,
        groups=num_groups,
    )

    # Convert torch input tensor to ttnn tensor
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor_nhwc, ttnn.bfloat16)

    # Move input tensor to device
    ttnn_input_tensor_on_device = ttnn_conv_layer.copy_input_to_device(ttnn_input_tensor)

    # Apply convolution operation on device
    ttnn_output_tensor_on_device_tile_layout = ttnn_conv_layer(ttnn_input_tensor_on_device)

    # Convert output and get output from device
    ttnn_output_tensor_on_device_row_layout = ttnn.to_layout(
        ttnn_output_tensor_on_device_tile_layout, ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor_on_device_row_layout)
    torch_ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    # Shape manipulations
    output_shape_nhwc = [
        torch_output_tensor_nchw.shape[0],
        torch_output_tensor_nchw.shape[2],
        torch_output_tensor_nchw.shape[3],
        torch_output_tensor_nchw.shape[1],
    ]
    torch_ttnn_output_tensor_nhwc = torch.reshape(torch_ttnn_output_tensor, output_shape_nhwc)
    torch_ttnn_output_tensor_nchw = torch.permute(torch_ttnn_output_tensor_nhwc, (0, 3, 1, 2))

    passing, pcc_msg = assert_with_pcc(torch_output_tensor_nchw, torch_ttnn_output_tensor_nchw, pcc=0.99)
    logger.info(pcc_msg)
    assert passing
