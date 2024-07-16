# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from torchvision import models
from loguru import logger
from models.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),),
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width",
    (
        (3, 64, 224, 224),
        (64, 64, 224, 224),
        (64, 128, 112, 112),
        (128, 128, 112, 112),
        (128, 256, 56, 56),
        (256, 256, 56, 56),
        (256, 256, 56, 56),
        (256, 512, 28, 28),
        (512, 512, 28, 28),
        (512, 512, 28, 28),
        (512, 512, 14, 14),
        (512, 512, 14, 14),
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
    output_channels,
    input_channels,
    input_height,
    input_width,
):
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weight_dtype,
        "ACTIVATIONS_DTYPE": act_dtype,
    }

    conv1_config = ttnn.Conv2dConfig(
        dtype=model_config["ACTIVATIONS_DTYPE"],
        weights_dtype=model_config["WEIGHTS_DTYPE"],
        math_fidelity=model_config["MATH_FIDELITY"],
        activation="relu",
        deallocate_activation=True,
        input_channels_alignment=16,
        act_block_h_override=0,
        transpose_shards=True,
    )

    # use_pretrained_weight=True,
    # dealloc_input=True,
    # final_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,

    torch_model.to(torch.bfloat16)
    torch_input_tensor = imagenet_sample_input.to(torch.bfloat16)

    conv_p = [output_channels, input_channels, input_height, input_width]
    has_bias = True
    conv_input_shape = [batch_size, conv_p[0], conv_p[2], conv_p[2]]
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

    weights_dtype = ttnn.bfloat8_b
    tt_weight_tensor = ttnn.from_torch(
        torch_weight_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor, weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)

    ds_out, _, _, ds_conv_weight_tensor, ds_conv_bias_tensor = ttnn.conv2d(
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
        conv_config=conv1_config,
        conv_op_cache={},
    )
