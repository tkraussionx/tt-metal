# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import pytest
import torch
import ttnn
import tt_lib
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull, is_grayskull


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
    enable_auto_formatting=False,
    padded_input_channels=None,
    fp32_accum=False,
    packer_l1_acc=False,
    output_layout=ttnn.TILE_LAYOUT,
):
    torch.manual_seed(0)
    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    conv_bias_shape = [1, 1, 1, output_channels]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
    # torch_input_tensor_nchw = torch.empty(conv_input_shape, dtype=torch.bfloat16).fill_(0.2).float()
    # torch_weight_tensor = torch.empty(conv_weight_shape, dtype=torch.bfloat16).fill_(0.1).float()
    # torch_bias_tensor = torch.empty(conv_bias_shape, dtype=torch.bfloat16).fill_(0).float()
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
    )
    output_shape_nhwc = [
        torch_out_golden_tensor.shape[0],
        torch_out_golden_tensor.shape[2],
        torch_out_golden_tensor.shape[3],
        torch_out_golden_tensor.shape[1],
    ]

    reader_patterns_cache = {}

    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )

    if not is_grayskull():
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_accum,
            packer_l1_acc=packer_l1_acc,
        )

    conv = ttnn.Conv2d(
        input_channels,
        output_channels,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dtype=activations_dtype,
        device=device,
        use_1d_systolic_array=use_1d_systolic_array,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        reader_patterns_cache=reader_patterns_cache,
        weight=tt_weight_tensor,
        bias=tt_bias_tensor,
        math_fidelity=math_fidelity,
        weights_dtype=weights_dtype,
        conv_blocking_and_parallelization_config_override=config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        enable_auto_formatting=enable_auto_formatting,
        deallocate_activation=True,
        padded_input_channels=padded_input_channels,
        compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        output_layout=output_layout,
    )

    assert "conv" in reader_patterns_cache and "halo" in reader_patterns_cache
    if enable_auto_formatting:
        tt_input_tensor_on_device = prepare_conv_input_and_copy_to_device_interleaved(
            device,
            torch_input_tensor,
            [batch_size, input_height, input_width, input_channels],
            use_shallow_conv_variant,
        )
    else:
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        tt_input_tensor_on_device = conv.copy_input_to_device(tt_input_tensor)
    tt_output_tensor_on_device = conv(tt_input_tensor_on_device)
    if enable_auto_formatting:
        tt_output_tensor_on_device = ttnn.to_layout(tt_output_tensor_on_device, ttnn.ROW_MAJOR_LAYOUT)
        tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
        torch_output_tensor = ttnn.to_torch(tt_output_tensor)
        torch_output_tensor = torch.split(torch_output_tensor, output_channels, 3)[0]
        torch_output_tensor = torch.reshape(torch_output_tensor, output_shape_nhwc)
    else:
        tt_output_tensor = conv.copy_output_from_device(tt_output_tensor_on_device)
        assert tt_output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    reader_patterns_cache.clear()
    if math_fidelity == ttnn.MathFidelity.LoFi and activations_dtype == ttnn.bfloat8_b:
        pcc = 0.9969
    else:
        pcc = 0.998
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=pcc)
    logger.info(pcc_msg)
    assert passing, "pcc message: " + str(pcc_msg)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, weights_dtype, use_1d_systolic_array, conv_blocking_and_parallelization_config_override, use_shallow_conv_variant",
    ((1, 512, 256, 60, 36, 3, 3, 2, 2, 1, 1, ttnn.bfloat8_b, True, None, False),),
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
def test_resnet50_conv_gs(
    use_program_cache,
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
    conv_blocking_and_parallelization_config_override,
    use_shallow_conv_variant,
):
    if batch_size > 8 and (activations_dtype != ttnn.bfloat8_b or weights_dtype != ttnn.bfloat8_b):
        pytest.skip("Batch > 8 must be run fully bfp8")

    if (
        activations_dtype == ttnn.bfloat16
        and batch_size == 20
        and (
            output_channels == 64
            or (
                stride_h == 2
                and (output_channels == 256 or (output_channels == 128 and weights_dtype == ttnn.bfloat16))
            )
        )
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

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
        config_override=conv_blocking_and_parallelization_config_override,
        use_shallow_conv_variant=use_shallow_conv_variant,
        padded_input_channels=16 if input_channels == 16 else None,
    )
