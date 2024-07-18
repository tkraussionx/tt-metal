# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from torchvision import models
from loguru import logger
from models.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity", ((1, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.LoFi),)
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "input_channels, output_channels, input_height, input_width",
    (
        (128, 256, 56, 56),
        (256, 256, 56, 56),
        (256, 512, 28, 28),
        (512, 512, 28, 28),
        (512, 512, 14, 14),
    ),
)
def test_vgg_conv(
    device,
    imagenet_sample_input,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight,
    input_channels,
    output_channels,
    input_height,
    input_width,
):
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weight_dtype,
        "ACTIVATIONS_DTYPE": act_dtype,
    }

    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {
            ttnn.experimental.tensor.CoreRange(
                ttnn.experimental.tensor.CoreCoord(0, 0),
                ttnn.experimental.tensor.CoreCoord(7, 0),
            ),
        }
    )
    conv_config = ttnn.Conv2dConfig(
        # core_grid=shard_grid,
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=True,
        activation="relu",
        deallocate_activation=False,
        input_channels_alignment=32,
        reallocate_halo_output=False,
        act_block_h_override=0,
        transpose_shards=True,
        height_sharding=True,
        # reshard_if_not_optimal = True,
        # override_sharding_config=True,
    )

    torch_model.to(torch.bfloat16)
    torch_input_tensor = imagenet_sample_input.to(torch.bfloat16)

    conv_p = [input_channels, output_channels, input_height, input_width]
    has_bias = True
    conv_input_shape = [batch_size, conv_p[0], conv_p[2], conv_p[3]]  # 1, 3, 224, 224
    conv_weight_shape = [conv_p[1], conv_p[0], 3, 3]
    conv_bias_shape = [1, 1, 1, conv_p[1]]
    torch_input_tensor_nchw = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
    torch_weight_tensor = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    torch_bias_tensor = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float() if has_bias else None
    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input_tensor_nchw,
        torch_weight_tensor,
        bias=torch_bias_tensor.reshape(-1) if has_bias else None,
        stride=(1, 1),
        padding=(1, 1),
    )

    weights_dtype = ttnn.bfloat16
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.bfloat16
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.bfloat16
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        in_channels=conv_p[0],
        out_channels=conv_p[1],
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=conv_p[2],
        input_width=conv_p[3],
        conv_config=conv_config,
        conv_op_cache={},
    )

    tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))
    torch_output_tensor = torch_output_tensor.reshape(batch_size, output_channels, out_height, out_width)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    logger.info(f"PCC: {pcc_msg}")
