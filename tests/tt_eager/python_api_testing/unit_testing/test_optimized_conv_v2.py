# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import tt_lib
import math
import time
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_allclose_and_pcc
import tests.tt_eager.python_api_testing.unit_testing.test_max_pool
from models.demos.resnet.tt.metalResnetBlock50 import (
    compute_conv_output_shape,
    resnet50_1x1_conv_as_matmul,
    resnet50_optimized_conv,
    _nearest_32,
    _nearest_y,
    format_tensor,
)
from models.utility_functions import skip_for_wormhole_b0

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_composite_conv import (
    TTPyCompositeConv,
    SlidingWindowOpParams,
)

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_max_pool import TTPyMaxPool

stencil_test_layers = []

# -------------------------------------------------------------------------------
# Original test expressing this implementation's max image resolution size,for one GS card
# -------------------------------------------------------------------------------
stencil_test_layers += [
    # ---------------------------------
    # Baseline Benchmarks
    # ---------------------------------
    (1, 32, 32, 448, 448, 3, 3, 2, 2, 1, 1, True),
    # (1, 32, 32, 224, 224, 3, 3, 2, 2, 1, 1, True),
    # (1, 32, 32, 112, 112, 3, 3, 2, 2, 1, 1, True),
    # (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True),
    # ---------------------------------
    # End Baseline Benchmarks
    # ---------------------------------

    # ---------------------------------
    # Runtime error distributiion benchmarks
    # ---------------------------------
    # (1, 32, 32, 200, 200, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 198, 198, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 196, 196, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 194, 194, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 192, 192, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 190, 190, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 188, 188, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 186, 186, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 184, 184, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 182, 182, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 180, 180, 3, 3, 2, 2, 1, 1, True), #   *** fails sharded core range (incompatible coor cordinate args)
    # (1, 32, 32, 178, 178, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 176, 176, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 174, 174, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 172, 172, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 170, 170, 3, 3, 2, 2, 1, 1, True), #   *** fails sharded core range (incompatible coor cordinate args)
    # (1, 32, 32, 168, 168, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 166, 166, 3, 3, 2, 2, 1, 1, True), #   PASSES, runtime: 3.4021406173706055 seconds
    # (1, 32, 32, 164, 164, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 162, 162, 3, 3, 2, 2, 1, 1, True), #   PASSES, runtime: 3.091364860534668 seconds
    # (1, 32, 32, 160, 160, 3, 3, 2, 2, 1, 1, True), #   PASSES, runtime: 3.2494564056396484 seconds
    # (1, 32, 32, 158, 158, 3, 3, 2, 2, 1, 1, True), #   PASSES, runtime: 3.338606119155884 seconds
    # (1, 32, 32, 156, 156, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 154, 154, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 152, 152, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 150, 150, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 148, 148, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 146, 146, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 144, 144, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 142, 142, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 140, 140, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 138, 138, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 136, 136, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 134, 134, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 132, 132, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 130, 130, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 128, 128, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 126, 126, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 124, 124, 3, 3, 2, 2, 1, 1, True), #   *** fails sharded core range (incompatible coor cordinate args)
    # (1, 32, 32, 122, 122, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 120, 120, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 118, 118, 3, 3, 2, 2, 1, 1, True), #   **  fails page size divisible by buffer size condition
    # (1, 32, 32, 116, 116, 3, 3, 2, 2, 1, 1, True), #  PASSES, runtime: 3.241814613342285 seconds
    # (1, 32, 32, 114, 114, 3, 3, 2, 2, 1, 1, True), #  PASSES, runtime: 3.2659966945648193 seconds
    # (1, 32, 32, 112, 112, 3, 3, 2, 2, 1, 1, True), #  PASSES, runtime: 3.23701548576355 seconds
    # (1, 32, 32, 110, 110, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 108, 108, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 106, 106, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 104, 104, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 102, 102, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # (1, 32, 32, 100, 100, 3, 3, 2, 2, 1, 1, True), #   *   fails sharded height num_cores assert
    # More below 100x100 in steps of 4
    # (1, 32, 32, 96, 96, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # (1, 32, 32, 92, 92, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # (1, 32, 32, 88, 88, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # (1, 32, 32, 84, 84, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # (1, 32, 32, 80, 80, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # (1, 32, 32, 76, 76, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # (1, 32, 32, 72, 72, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # (1, 32, 32, 68, 68, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # (1, 32, 32, 64, 64, 3, 3, 2, 2, 1, 1, True),   #   *   fails sharded height num_cores assert
    # ---------------------------------
]

# -------------------------------------------------------------------------------
# Exhaustive list of stencil grids in tensor format, w/ channels in multiples of 32
# and image resolutions in multiples of 56 up to a max of 448
# -------------------------------------------------------------------------------
# for input_channels in [32, 64]:
#     for output_channels in [32, 64]:
#         for size in [56, 112, 168, 224, 280, 336, 392, 448]:
#             # Ensure the size is divisible by 4 to be compatible after convolution and pooling
#             if size % 4 == 0:
#                 stencil_test_layers.append((1, output_channels, input_channels, size, size, 3, 3, 1, 1, 1, 1, True))
#                 stencil_test_layers.append((1, output_channels, input_channels, size, size, 3, 3, 2, 2, 1, 1, True))
#                 stencil_test_layers.append((1, output_channels, input_channels, size, size, 3, 3, 1, 1, 0, 0, True))
#                 stencil_test_layers.append((1, output_channels, input_channels, size, size, 3, 3, 2, 2, 0, 0, True))

# For benchmark labeling purposes
for test_case in stencil_test_layers:
    print(test_case)


# Original rn50 tests (preserved), plus stencil tests as defined above
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, is_1d_systolic",
    [
        # unique convs in rn50 (complete list)
        # first conv post folding and input_channels padding to tile width
        # (8, 64, 16, 115, 115, 4, 4, 1, 1, 0, 0, True),
        # # rn50 layer1
        # (8, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, True),
        # # rn50 layer2
        # (8, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True),
        # (20, 128, 128, 56, 56, 3, 3, 2, 2, 1, 1, True),
        # (8, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1, True),
        # # rn50 layer3
        # (8, 256, 256, 28, 28, 3, 3, 2, 2, 1, 1, False),
        # (8, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, False),
        # # rn50 layer4
        # (8, 512, 512, 14, 14, 3, 3, 2, 2, 1, 1, False),
        # (8, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1, False),
    ]
    + stencil_test_layers,
)
@pytest.mark.parametrize(
    "weights_dtype",
    [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["weights_BFLOAT16", "weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["activations_BFLOAT16", "activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity", [tt_lib.tensor.MathFidelity.HiFi4, tt_lib.tensor.MathFidelity.LoFi], ids=["HiFi4", "LoFi"]
)
def test_optimized_conv_v2(
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
    is_1d_systolic,
):
    # -------------------------------------------------------------------------------
    # Host-side
    # -------------------------------------------------------------------------------

    start_time = time.time()

    if input_channels == 16:
        pytest.skip("These tests are hanging in interleaved_to_sharded after rebase. Issue: #4336")

    if math_fidelity != tt_lib.tensor.MathFidelity.LoFi:
        pytest.skip(
            "By default, only run tests with LoFi math for pipelines. For local unit testing, enable the other variants by uncommenting the skip here!"
        )

    if (
        activations_dtype == tt_lib.tensor.DataType.BFLOAT16
        and batch_size == 20
        and (
            output_channels == 64
            or (
                stride_h == 2
                and (
                    output_channels == 256
                    or (output_channels == 128 and weights_dtype == tt_lib.tensor.DataType.BFLOAT16)
                )
            )
        )
    ):
        pytest.skip("Skipping test because it won't fit in L1!")

    assert output_channels % 32 == 0
    torch.manual_seed(0)

    # -------------------------------------------------------------------------------
    # Original kernel (preserved)
    # -------------------------------------------------------------------------------
    # conv_input_shape = [batch_size, input_channels, input_height, input_width]
    # conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    # conv_bias_shape = [1, 1, 1, output_channels]
    # conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    # conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
    # conv_input_shape_nhwc = conv_input_pyt_nhwc.shape
    # conv_weight_pyt = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    # conv_bias_pyt = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
    # out_golden = torch.nn.functional.conv2d(
    #     conv_input_pyt,
    #     conv_weight_pyt,
    #     bias=conv_bias_pyt.reshape(-1),
    #     stride=(stride_h, stride_w),
    #     padding=(pad_h, pad_w),
    # )
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # 5-point star stencil
    # -------------------------------------------------------------------------------

    conv_input_shape = [batch_size, input_channels, input_height, input_width]
    conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()

    conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    conv_weight_pyt = torch.zeros(conv_weight_shape, dtype=torch.bfloat16)

    star_value = 1.0 / 5
    conv_weight_pyt[:, :, 1, 1] = star_value  # Center
    conv_weight_pyt[:, :, 0, 1] = star_value  # Top neighbor
    conv_weight_pyt[:, :, 2, 1] = star_value  # Bottom neighbor
    conv_weight_pyt[:, :, 1, 0] = star_value  # Left neighbor
    conv_weight_pyt[:, :, 1, 2] = star_value  # Right neighbor
    conv_weight_pyt = conv_weight_pyt.float()

    conv_bias_shape = [1, 1, 1, output_channels]
    conv_bias_pyt = torch.zeros(conv_bias_shape, dtype=torch.bfloat16).float()

    conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
    conv_input_shape_nhwc = conv_input_pyt_nhwc.shape

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # 9-point box stencil
    # -------------------------------------------------------------------------------

    # conv_input_shape = [batch_size, input_channels, input_height, input_width]

    # conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()

    # conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    # blur_value = 1.0 / (filter_height * filter_width)
    # conv_weight_pyt = torch.full(conv_weight_shape, blur_value, dtype=torch.bfloat16).float()

    # conv_bias_shape = [1, 1, 1, output_channels]
    # conv_bias_pyt = torch.zeros(conv_bias_shape, dtype=torch.bfloat16).float()  # Example using zeros

    # conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
    # conv_input_shape_nhwc = conv_input_pyt_nhwc.shape
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # 5-point nonlinear stencil
    # -------------------------------------------------------------------------------

    # conv_input_shape = [batch_size, input_channels, input_height, input_width]
    # conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()

    # conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    # conv_weight_pyt = torch.zeros(conv_weight_shape, dtype=torch.bfloat16)

    # star_value = 1.0 / 5
    # conv_weight_pyt[:, :, 1, 1] = 0.4  # Center with higher weight
    # conv_weight_pyt[:, :, 0, 1] = math.sin(0.1)
    # conv_weight_pyt[:, :, 2, 1] = math.cos(0.1)
    # conv_weight_pyt[:, :, 1, 0] = 0.05
    # conv_weight_pyt[:, :, 1, 2] = 0.05

    # conv_weight_pyt = conv_weight_pyt.float()

    # conv_bias_shape = [1, 1, 1, output_channels]
    # conv_bias_pyt = torch.zeros(conv_bias_shape, dtype=torch.bfloat16).float()

    # conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
    # conv_input_shape_nhwc = conv_input_pyt_nhwc.shape

    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # 5-point spatial gradient stencil (edge detection)
    # -------------------------------------------------------------------------------

    # conv_input_shape = [batch_size, input_channels, input_height, input_width]
    # conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()

    # conv_weight_shape = [output_channels, input_channels, filter_height, filter_width]
    # conv_weight_pyt = torch.zeros(conv_weight_shape, dtype=torch.bfloat16)

    # star_value = 1.0 / 5
    # conv_weight_pyt[:, :, 0, 0] = -1; conv_weight_pyt[:, :, 0, 1] = 0; conv_weight_pyt[:, :, 0, 2] = 1
    # conv_weight_pyt[:, :, 1, 0] = -2; conv_weight_pyt[:, :, 1, 1] = 0; conv_weight_pyt[:, :, 1, 2] = 2
    # conv_weight_pyt[:, :, 2, 0] = -1; conv_weight_pyt[:, :, 2, 1] = 0; conv_weight_pyt[:, :, 2, 2] = 1

    # conv_weight_pyt = conv_weight_pyt.float()

    # conv_bias_shape = [1, 1, 1, output_channels]
    # conv_bias_pyt = torch.zeros(conv_bias_shape, dtype=torch.bfloat16).float()

    # conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
    # conv_input_shape_nhwc = conv_input_pyt_nhwc.shape

    # -------------------------------------------------------------------------------

    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    # Start the timer
    convstart = time.time_ns()

    # -------------------------------------------------------------------------------
    # Golden conv + maxpool
    # -------------------------------------------------------------------------------
    out_golden = torch.nn.functional.conv2d(
        conv_input_pyt,
        conv_weight_pyt,
        bias=conv_bias_pyt.reshape(-1),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
    )

    # End the timer
    convend = time.time_ns()
    # Calculate the elapsed time in nanoseconds
    conv_time = convend - convstart

    print(f"HOST_Conv finished in: {conv_time} nanoseconds")
    # kernel_size = (2, 2)
    # stride = (2, 2)
    # padding = (0, 0)

    # # Start the timer
    maxstart = time.time_ns()

    out_golden = torch.nn.functional.max_pool2d(out_golden, kernel_size=kernel_size, stride=stride, padding=padding)

    # End the timer
    maxend = time.time_ns()

    # Calculate the elapsed time in nanoseconds
    max_pool_time = maxend - maxstart

    print(f"HOST_Maxpool finished in: {max_pool_time} nanoseconds")

    conv_params = [output_channels, input_channels, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, 1, 1]
    conv_output_shape = compute_conv_output_shape(conv_params, conv_input_shape_nhwc)
    logger.info(f"Conv output shape - {conv_output_shape}")

    # -------------------------------------------------------------------------------
    # Device-side
    # -------------------------------------------------------------------------------

    # --------- HOST PREPROCESSING FOR CONVOLUTION

    sliding_window_op_params = SlidingWindowOpParams(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        window_h=filter_height,
        window_w=filter_width,
        batch_size=batch_size,
        input_h=input_height,
        input_w=input_width,
    )

    conv = TTPyCompositeConv(
        sliding_window_op_params,
        conv_weight_pyt.reshape(-1).tolist(),
        output_channels,
        input_channels,
        device,
        is_1d_systolic,
        conv_bias_pyt.reshape(-1).tolist(),
        weights_dtype=weights_dtype,
        output_dtype=activations_dtype,
        math_fidelity=math_fidelity,
    )

    conv_input = tt_lib.tensor.Tensor(
        conv_input_pyt_nhwc.reshape(-1).tolist(),
        conv_input_pyt_nhwc.shape,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )

    # --------- DEVICE EXECUTION FOR CONVOLUTION

    conv_input_on_device = conv.copy_input_to_device(conv_input)

    # Optimized conv v2
    output_on_device = conv(conv_input_on_device)

    # -------------------------------------------------------------------------------
    # Max Pooling todo's (all to be peformed on-device)
    # -------------------------------------------------------------------------------
    # Pre-process max pool tensor
    # Set maxpool params
    # Call ttn maxpool function with output conv tensor as input maxpool tensor
    # ???
    # Profit!
    # Send final tensor back to host
    # -------------------------------------------------------------------------------

    # Copy sharded output on host
    # out is in row major layout and NHWC shape
    out = conv.copy_output_from_device(output_on_device)

    assert out.layout() == tt_lib.tensor.Layout.ROW_MAJOR

    # -------------------------------------------------------------------------------
    # Reshaping for Max Pooling
    # -------------------------------------------------------------------------------
    out_result = out.to_torch()
    # NHWC to NCHW
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Original reshaping for Conv (preserved)
    # -------------------------------------------------------------------------------
    # Convert the output to a PyTorch tensor and change from NHWC to NCHW
    # out_result = out.to_torch()
    # out_result = torch.transpose(out_result, 1, 3)
    # out_result = torch.transpose(out_result, 2, 3)
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # Max Pooling (on host for now)
    # -------------------------------------------------------------------------------
    # Define max pooling parameters
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)

    out_pooled = torch.nn.functional.max_pool2d(out_result, kernel_size=kernel_size, stride=stride, padding=padding)

    maxpool_output_shape = out_pooled.shape
    logger.info(f"Maxpool output shape - {maxpool_output_shape}")
    # -------------------------------------------------------------------------------

    # Clear static cache maps
    TTPyCompositeConv.clear_static_cache_maps()

    # Compare with golden model if necessary
    if math_fidelity == tt_lib.tensor.MathFidelity.LoFi and activations_dtype == tt_lib.tensor.DataType.BFLOAT8_B:
        pcc = 0.998
    else:
        pcc = 0.999
    passing_pcc, info = comp_pcc(out_golden, out_pooled, pcc=pcc)
    print("Passing=", passing_pcc)
    print("Info=", info)
    assert passing_pcc

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the runtime
    logger.info(f"Test runtime: {total_time} seconds")
