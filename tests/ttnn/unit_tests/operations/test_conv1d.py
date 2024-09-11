# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    is_wormhole_b0,
    skip_for_grayskull,
    is_grayskull,
    is_wormhole_b0,
    is_x2_harvested,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
import ttnn
import math
import os
import torch.nn as nn


def run_conv(
    device,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant=False,
    transpose_mcast=True,
    enable_auto_formatting=False,
    padded_input_channels=None,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=True,
    debug=False,
    groups=1,
):
    # has_bias = False
    has_bias = False
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_length]
    conv_weight_shape = [output_channels, input_channels // groups, kernel_size]
    conv_bias_shape = [1, 1, output_channels]
    torch_input_tensor_ncl = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_ncl, (0, 2, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv1d(
        torch_input_tensor_ncl,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=stride,
        padding=padding,
        groups=groups,
    )

    reader_patterns_cache = {}

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = None
    if has_bias:
        tt_bias_tensor = ttnn.from_torch(
            torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, ttnn.bfloat16
    )  # , device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    # print("tt_input_tensor memory config: ", tt_input_tensor.memory_config())
    # tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
    # tt_input_tensor = ttnn.sharded_to_interleaved(tt_input_tensor, ttnn.L1_MEMORY_CONFIG)

    conv_config = ttnn.Conv1dConfig(
        dtype=output_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        shard_layout=(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_1d_systolic_array else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        ),
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        deallocate_activation=deallocate_activation,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        # reshard_if_not_optimal=True,
        # enable_split_reader=True,
        # enable_act_double_buffer=True,
    )
    if config_override and "act_block_h" in config_override:
        conv_config.act_block_h_override = config_override["act_block_h"]
        print("Setting Act Block H to ", conv_config.act_block_h_override)
    if config_override and "num_cores_nhw" in config_override:
        if config_override["num_cores_nhw"] == 98:
            conv_config.core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (11, 7)), ttnn.CoreRange((0, 8), (1, 8))})
            conv_config.override_sharding_config = True
            print("Setting num_cores_nhw to 98")

    [tt_output_tensor_on_device, out_length, weights_device, bias_device] = ttnn.Conv1d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=input_channels,
        out_channels=output_channels,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        batch_size=batch_size,
        input_length=input_length,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NLC shape
    # NLC to NCL
    torch_output_tensor = torch_output_tensor.reshape(batch_size, out_length, output_channels)

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 1))
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


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (1, 5120, 5120, 32, 4, 1, 3, 5120, True, None, False),
        (1, 5120, 5120, 1024, 4, 1, 3, 5120, True, None, False),
        (1, 2560, 2560, 1027, 4, 1, 0, 2560, True, None, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv1d_mamba(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (1, 32, 3, 32, 3, 1, 1, 1, True, None, False),
        (1, 128, 32, 1024, 5, 1, 2, 1, True, None, False),
        (1, 512, 32, 5120, 3, 1, 1, 1, True, None, False),
        (1, 64, 64, 2560, 3, 1, 1, 32, True, None, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv1d(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (
            32,
            64,
            3,
            2500,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv1: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 2500, 1, 1, 0, 1, True, None, False),   # Conv2: Passed
        (32, 1024, 128, 2500, 1, 1, 0, 1, True, None, False),  # Conv3: Failed with OOM error
        (
            32,
            64,
            3,
            2500,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv4: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 2500, 1, 1, 0, 1, True, None, False),   # Conv5: Passed
        (32, 1024, 128, 2500, 1, 1, 0, 1, True, None, False),  # Conv6: Failed with OOM error
        (32, 512, 1088, 2500, 1, 1, 0, 1, True, None, False),  # Conv7: Failed with OOM error
        (32, 256, 512, 2500, 1, 1, 0, 1, True, None, False),  # Conv8: Failed with OOM error
        # (32, 128, 256, 2500, 1, 1, 0, 1, True, None, False),  # Conv9: Passed when output_dtype is BFLOAT8_B
        (
            32,
            3,
            128,
            2500,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv10: Failed with RuntimeError: shape '[32, 2500, 3]' is invalid for input of size 2560000
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_pointnet_conv1d(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (
            32,
            64,
            3,
            1024,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv1: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 1024, 1, 1, 0, 1, True, None, False),   # Conv2: Passed
        (32, 1024, 128, 1024, 1, 1, 0, 1, True, None, False),  # Conv3: Failed with OOM error
        (
            32,
            64,
            3,
            1024,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv4: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 1024, 1, 1, 0, 1, True, None, False),   # Conv5: Passed
        (32, 1024, 128, 1024, 1, 1, 0, 1, True, None, False),  # Conv6: Failed with OOM error
        (32, 512, 1088, 1024, 1, 1, 0, 1, True, None, False),  # Conv7: Failed with OOM error
        (32, 256, 512, 1024, 1, 1, 0, 1, True, None, False),  # Conv8: Failed with OOM error
        # (32, 128, 256, 1024, 1, 1, 0, 1, True, None, False),  # Conv9: Passed
        (
            32,
            3,
            128,
            1024,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv10: RuntimeError: shape '[32, 1024, 3]' is invalid for input of size 1048576
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_pointnet_conv1d_low_res_1024(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (
            32,
            64,
            3,
            512,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv1: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 512, 1, 1, 0, 1, True, None, False),   # Conv2: Passed
        (
            32,
            1024,
            128,
            512,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv3: Fails when output dtype is bfloat16 with RuntimeError: TT_THROW @ ../tt_metal/impl/program/program.cpp:519: tt::exception
        (
            32,
            64,
            3,
            512,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv4: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 512, 1, 1, 0, 1, True, None, False),   # Conv5: Passed
        (
            32,
            1024,
            128,
            512,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv6: Fails when output dtype is bfloat16 with RuntimeError: TT_THROW @ ../tt_metal/impl/program/program.cpp:519: tt::exception
        (32, 512, 1088, 512, 1, 1, 0, 1, True, None, False),  # Conv7: Failed with OOM error
        # (32, 256, 512, 512, 1, 1, 0, 1, True, None, False),  # Conv8: Passed
        # (32, 128, 256, 512, 1, 1, 0, 1, True, None, False),  # Conv9: Passed
        (
            32,
            3,
            128,
            512,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv10: RuntimeError: shape '[32, 512, 3]' is invalid for input of size 524288
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_pointnet_conv1d_low_res_512(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (
            32,
            64,
            3,
            256,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv1: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 256, 1, 1, 0, 1, True, None, False),   # Conv2: Passed
        # (32, 1024, 128, 256, 1, 1, 0, 1, True, None, False),  # Conv3: Passed
        (
            32,
            64,
            3,
            256,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv4: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 256, 1, 1, 0, 1, True, None, False),   # Conv5: Passed
        # (32, 1024, 128, 256, 1, 1, 0, 1, True, None, False),  # Conv6: Passed
        (
            32,
            512,
            1088,
            256,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv7: Failed with RuntimeError: TT_THROW @ ../tt_metal/impl/program/program.cpp:519: tt::exception
        # (32, 256, 512, 256, 1, 1, 0, 1, True, None, False),  # Conv8: Passed
        # (32, 128, 256, 256, 1, 1, 0, 1, True, None, False),  # Conv9: Passed
        (
            32,
            3,
            128,
            256,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv10: RuntimeError: shape '[32, 256, 3]' is invalid for input of size 262144
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_pointnet_conv1d_low_res_256(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_length, kernel_size, stride, padding, groups, use_1d_systolic_array, config_override, use_shallow_conv_variant",
    (
        (
            32,
            64,
            3,
            128,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv1: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 128, 1, 1, 0, 1, True, None, False),   # Conv2: Passed
        # (32, 1024, 128, 128, 1, 1, 0, 1, True, None, False),  # Conv3: Passed
        (
            32,
            64,
            3,
            128,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv4: Failed with RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:959: a_shape[-1] == b_shape[-2]
        # (32, 128, 64, 128, 1, 1, 0, 1, True, None, False),   # Conv5: Passed
        # (32, 1024, 128, 128, 1, 1, 0, 1, True, None, False),  # Conv6: Passed
        (
            32,
            512,
            1088,
            128,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv7: Failed when weights dtype is bfloat16 with RuntimeError: TT_THROW @ ../tt_metal/impl/program/program.cpp:519: tt::exception
        # (32, 256, 512, 128, 1, 1, 0, 1, True, None, False),  # Conv8: Passed
        # (32, 128, 256, 128, 1, 1, 0, 1, True, None, False),  # Conv9: Passed
        (
            32,
            3,
            128,
            128,
            1,
            1,
            0,
            1,
            True,
            None,
            False,
        ),  # Conv10: RuntimeError: shape '[32, 128, 3]' is invalid for input of size 131072
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_pointnet_conv1d_low_res_128(
    device,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    output_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_length,
    kernel_size,
    stride,
    padding,
    groups,
    use_1d_systolic_array,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if groups > 5120 or input_channels > 5120 or output_channels > 5120:
        pytest.skip("OOM")
    if (input_channels > 2560 or output_channels > 2560) and output_dtype == ttnn.bfloat16:
        pytest.skip("OOM")

    run_conv(
        device,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        output_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        use_1d_systolic_array,
        config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        transpose_mcast=use_1d_systolic_array,  ## use RM (transpose_mcast=False) with 2D on WH
        padded_input_channels=None,
        output_layout=output_layout,
        groups=groups,
    )
