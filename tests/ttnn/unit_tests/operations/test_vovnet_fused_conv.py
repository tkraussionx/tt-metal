# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_grayskull,
    is_grayskull,
    is_wormhole_b0,
    is_x2_harvested,
    is_blackhole,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn


def run_conv_with_split(
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
    split_factor=2,
    fp32_accum=False,
    packer_l1_acc=True,
    fuse=False,
):
    torch.manual_seed(0)
    assert input_channels % split_factor == 0
    split_input_channels = input_channels // split_factor
    torch_input_tensor_nhwc = torch.randn([7, 14, 14, 1088], dtype=torch.bfloat16)
    torch_input_tensor_nchw = torch_input_tensor_nhwc.permute(0, 3, 1, 2)
    torch_weight_tensor = torch.randn([768, 1088, 1, 1], dtype=torch.bfloat16)
    bn_weight = torch.randn([768], dtype=torch.bfloat16)
    bn_bias = torch.randn([768], dtype=torch.bfloat16)
    bn_running_mean = torch.randn([768], dtype=torch.bfloat16)
    bn_running_var = torch.randn([768], dtype=torch.bfloat16)

    # Torch impl ================
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw.float(),
        torch_weight_tensor.float(),
        bias=None,  # torch_bias_tensor.reshape(-1),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
    )
    bn = torch.nn.functional.batch_norm(
        input=torch_out_golden_tensor,
        running_mean=bn_running_mean.float(),
        running_var=bn_running_var.float(),
        weight=bn_weight.float(),
        bias=bn_bias.float(),
        training=False,
        momentum=0.1,
        eps=1e-05,
    )
    relu = torch.nn.ReLU()
    torch_out_golden_tensor = relu(bn)
    # Torch impl End ==============

    eps = 1e-05
    bn_weight = bn_weight.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = bn_bias.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = bn_running_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = bn_running_var.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bn__weight = (torch_weight_tensor / torch.sqrt(bn_running_var + eps)) * bn_weight
    bn__1bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bn__bias = bn__1bias.reshape(1, 1, 1, -1)

    split_input_tensors = torch.split(torch_input_tensor_nchw, split_input_channels, 1)
    split_weight_tensors = torch.split(bn__weight, split_input_channels, 1)

    reader_patterns_cache = {}
    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        activation="relu",
        shard_layout=(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        ),
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        input_channels_alignment=32,
    )

    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]

    torch_output_tensor = None
    for i in range(split_factor):
        tt_weight_tensor = ttnn.from_torch(
            split_weight_tensors[i], weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )
        tt_bias_tensor = ttnn.from_torch(bn__bias, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32)
        torch_input_tensor = torch.permute(split_input_tensors[i], (0, 2, 3, 1))
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

        [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor,
            in_channels=split_input_channels,
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
        )
        tt_conv_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
        torch_conv_output_tensor = ttnn.to_torch(tt_conv_output_tensor)

        torch_conv_output_tensor = torch_conv_output_tensor.reshape(batch_size, out_height, out_width, output_channels)

        torch_conv_output_tensor = torch.permute(torch_conv_output_tensor, (0, 3, 1, 2))
        if i == 0:
            torch_output_tensor = torch_conv_output_tensor
        else:
            torch_output_tensor = torch.add(torch_output_tensor, torch_conv_output_tensor)
        print("Split output shapes ", torch_output_tensor.shape, torch_conv_output_tensor.shape)

    if math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998

    assert_with_pcc(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, use_1d_systolic_array, config_override, use_shallow_conv_variant, groups",
    ((7, 768, 1088, 14, 14, 1, 1, 1, 1, 0, 0, True, {"act_block_h": 0 * 32}, False, 1),),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_vovnet_convs(
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
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv_with_split(
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
        split_factor=2,
    )
