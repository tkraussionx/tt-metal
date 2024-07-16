# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn
import tt_lib
import math
import os
import torch.nn as nn


def run_conv(
    device,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant=False,
    transpose_mcast=True,
    enable_auto_formatting=False,
    padded_input_channels=None,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
    debug=False,
    groups=1,
):
    # has_bias = False
    has_bias = True
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels // groups, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16)  # .float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16)  # .float()

    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16)  # .float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        groups=groups,
    )
    print(
        "Torch conv :",
        torch_input_tensor_nchw.shape,
        " ",
        torch_weight_tensor.shape,
        " ",
        torch_bias_tensor.shape,
        " ",
        torch_out_golden_tensor.shape,
    )
    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]

    reader_patterns_cache = {}

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.bfloat16
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.bfloat16
        )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
    # breakpoint()
    print("Shape of weight :", tt_weight_tensor.shape, "  ", tt_input_tensor.shape, " ", tt_bias_tensor.shape)
    """
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        height_sharding=True,
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        deallocate_activation=False,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
    )
    """
    """
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        height_sharding=use_1d_systolic_array,
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        deallocate_activation=deallocate_activation,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
    )
    """
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        # weights_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        height_sharding=True,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=16 if tt_input_tensor.shape[-1] < 16 else 32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )

    # if config_override and "act_block_h" in config_override:
    #    conv_config.act_block_h_override = config_override["act_block_h"]
    #    print("Setting Act Block H to ", conv_config.act_block_h_override)

    print("Shapes :", tt_input_tensor.shape, " ", tt_weight_tensor.shape, " ", tt_bias_tensor.shape)
    print("params")
    print(conv_config)
    print("kernel :", filter_height, " ", filter_width)
    print("Stride :", stride_h, " ", stride_w)
    print("Padding :", pad_h, " ", pad_w)
    print("input h and w :", input_height, " ", input_width)
    print("input channels :", input_channels, " ", output_channels)
    print("Groups :", groups)
    [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        # debug=True,
        groups=1,
    )
    print("params")
    print(conv_config)
    print("kernel :", filter_height, " ", filter_width)
    print("Stride :", stride_h, " ", stride_w)
    print("Padding :", pad_h, " ", pad_w)
    print("input h and w :", input_height, " ", input_width)
    print("input channels :", input_channels, " ", output_channels)
    print("Groups :", groups)
    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    # if enable_auto_formatting:
    #     torch_output_tensor = torch.split(torch_output_tensor, output_channels, 3)[0]
    #     torch_output_tensor = torch.reshape(torch_output_tensor, output_shape_nhwc)
    # else:
    #     tt_output_tensor = conv.copy_output_from_device(tt_output_tensor_on_device)
    #     assert tt_output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    #     torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_height, out_width, output_channels)

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()

    if not fp32_accum:
        pcc = 0.995
    elif math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    print(pcc_msg)
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (1, 48, 48, 128, 128, 1, 1, 1, 1, 0, 0, 1, True, None, False),
        (1, 48, 48, 128, 128, 1, 1, 1, 1, 0, 0, 1, True, None, False),
        (1, 48, 48, 128, 128, 1, 1, 1, 1, 0, 0, 1, True, None, False),
        (1, 48, 64, 64, 64, 1, 1, 1, 1, 0, 0, 1, True, None, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_groups(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    groups,
    output_layout,
):
    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=groups,
        output_layout=output_layout,
    )


"""
pytorch:
I/P : torch.Size([1, 48, 128, 128])  weight: torch.Size([48, 48, 1, 1])   bias: torch.Size([1, 1, 1, 48])

ttnn:
I/P: ttnn.Shape([1, 128, 128, 48])   weight : ttnn.Shape([48, 48, 1, 1])   bias: ttnn.Shape([1, 1, 1, 48])


Error:
E           RuntimeError: TT_THROW @ ../ttnn/cpp/ttnn/operations/matmul/matmul.cpp:60: tt::exception
E           info:
E           ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor
E           backtrace:
E            --- ttnn::operations::matmul::matmul(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::__1::optional<tt::tt_metal::Tensor const> const&, tt::operations::primary::Matmul const&)
"""
