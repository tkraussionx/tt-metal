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
    torch.manual_seed(3)
    input_channels = [3, 32, 64]
    output_channels = [32, 64, 128]
    conv_input_shape = [1, input_channels[0], 32, 32]

    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16)
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))

    weights = []
    bias = []
    for i in range(3):
        conv_weight_shape = [output_channels[i], input_channels[i], 3, 3]
        torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16)
        conv_bias_shape = [1, 1, 1, output_channels[i]]
        torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16)
        weights.append(torch_weight_tensor)
        bias.append(torch_bias_tensor)

    for i in range(3):
        torch_out_golden_tensor = torch.nn.functional.conv2d(
            torch_input_tensor_nchw,
            weights[i],
            bias=bias[i].reshape(-1),
            stride=(1, 1),
            groups=1,
        )
        torch_input_tensor_nchw = torch_out_golden_tensor

    reader_patterns_cache = {}

    tt_weight = []
    tt_bias = []
    for i in range(3):
        tt_weight_tensor = ttnn.from_torch(weights[i], dtype=ttnn.bfloat16)
        tt_weight.append(tt_weight_tensor)
        tt_bias_tensor = ttnn.from_torch(bias[i], dtype=ttnn.bfloat16)
        tt_bias.append(tt_bias_tensor)
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="",
        height_sharding=True,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )

    for i in range(3):
        [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight[i],
            in_channels=input_channels[i],
            out_channels=output_channels[i],
            device=device,
            bias_tensor=tt_bias[i],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=1,
            input_height=tt_input_tensor.shape[1],
            input_width=tt_input_tensor.shape[2],
            conv_config=conv_config,
            conv_op_cache=reader_patterns_cache,
            groups=1,
        )
        tt_output_tensor_on_device = ttnn.to_layout(tt_output_tensor_on_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_output_tensor_on_device = ttnn.reshape(
            tt_output_tensor_on_device, (1, out_height, out_width, tt_output_tensor_on_device.shape[-1])
        )
        tt_input_tensor = tt_output_tensor_on_device

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    print("Shape", torch_output_tensor.shape, " ", torch_out_golden_tensor.shape)

    reader_patterns_cache.clear()

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    print(pcc_msg)
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
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
    output_layout,
):
    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_layout=output_layout,
    )
