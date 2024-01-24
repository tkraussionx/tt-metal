# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from torch import nn
from diffusers import StableDiffusionPipeline

from models.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_downblock_2d import downblock2d


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


def preprocess_groupnorm_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.GroupNorm):
        parameters["weight"] = preprocess_groupnorm_parameter(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_groupnorm_parameter(model.bias, dtype=ttnn.bfloat16)
    return parameters


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 1280, 4, 4),
    ],
)
@pytest.mark.parametrize(
    "temb_shape",
    [
        (1, 1, 2, 1280),
    ],
)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
def test_down_block_2d_256x256_ttnn(input_shape, temb_shape, device, model_name, reset_seeds):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    unet_downblock = pipe.unet.down_blocks[3]
    unet_resnet_downblock_module_list = unet_downblock.resnets
    temb_channels = temb_shape[3]
    batch_size, in_channels, input_height, input_width = input_shape
    torch_hidden_states = torch.randn(input_shape, dtype=torch.float32)
    temb = torch.randn(temb_shape, dtype=torch.float32)

    torch_output, torch_output_states = unet_downblock(torch_hidden_states, temb.squeeze(0).squeeze(0))

    ttnn_hidden_states = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device),
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_temb = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(temb, dtype=ttnn.bfloat16), device),
        layout=ttnn.TILE_LAYOUT,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet,
        device=device,
    )

    resnet_parameters = parameters.down_blocks[3].resnets
    for i in range(len(resnet_parameters)):
        resnet_parameters[i].norm1.weight = torch_to_ttnn(resnet_parameters[i].norm1.weight, device)
        resnet_parameters[i].norm1.bias = torch_to_ttnn(resnet_parameters[i].norm1.bias, device)
        resnet_parameters[i].norm2.weight = torch_to_ttnn(resnet_parameters[i].norm2.weight, device)
        resnet_parameters[i].norm2.bias = torch_to_ttnn(resnet_parameters[i].norm2.bias, device)

    ttnn_down_block = downblock2d(
        temb=ttnn_temb,
        hidden_states=ttnn_hidden_states,
        device=device,
        in_channels=in_channels,
        out_channels=in_channels,
        temb_channels=temb_channels,
        dropout=0.0,
        num_layers=2,
        resnet_eps=1e-05,
        resnet_time_scale_shift="default",
        resnet_act_fn="silu",
        resnet_groups=32,
        resnet_pre_norm=True,
        output_scale_factor=1.0,
        add_downsample=False,
        downsample_padding=1,
        parameters=parameters,
    )

    ttnn_out, ttnn_output_states = ttnn_down_block
    ttnn_output = ttnn.to_layout(ttnn_out, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 1280, 8, 8),
    ],
)
@pytest.mark.parametrize(
    "temb_shape",
    [
        (1, 1, 2, 1280),
    ],
)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
def test_down_block_2d_512x512(input_shape, temb_shape, device, model_name, reset_seeds):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    unet_downblock = pipe.unet.down_blocks[3]
    unet_resnet_downblock_module_list = unet_downblock.resnets
    temb_channels = temb_shape[3]
    batch_size, in_channels, input_height, input_width = input_shape
    torch_hidden_states = torch.randn(input_shape, dtype=torch.float32)
    temb = torch.randn(temb_shape, dtype=torch.float32)

    torch_output, torch_output_states = unet_downblock(torch_hidden_states, temb.squeeze(0).squeeze(0))

    ttnn_hidden_states = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16), device),
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_temb = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(temb, dtype=ttnn.bfloat16), device),
        layout=ttnn.TILE_LAYOUT,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet,
        device=device,
    )

    resnet_parameters = parameters.down_blocks[3].resnets
    for i in range(len(resnet_parameters)):
        resnet_parameters[i].norm1.weight = torch_to_ttnn(resnet_parameters[i].norm1.weight, device)
        resnet_parameters[i].norm1.bias = torch_to_ttnn(resnet_parameters[i].norm1.bias, device)
        resnet_parameters[i].norm2.weight = torch_to_ttnn(resnet_parameters[i].norm2.weight, device)
        resnet_parameters[i].norm2.bias = torch_to_ttnn(resnet_parameters[i].norm2.bias, device)

    ttnn_down_block = downblock2d(
        temb=ttnn_temb,
        hidden_states=ttnn_hidden_states,
        device=device,
        in_channels=in_channels,
        out_channels=in_channels,
        temb_channels=temb_channels,
        dropout=0.0,
        num_layers=2,
        resnet_eps=1e-05,
        resnet_time_scale_shift="default",
        resnet_act_fn="silu",
        resnet_groups=32,
        resnet_pre_norm=True,
        output_scale_factor=1.0,
        add_downsample=False,
        downsample_padding=1,
        parameters=parameters,
    )

    ttnn_out, ttnn_output_states = ttnn_down_block
    ttnn_output = ttnn.to_layout(ttnn_out, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)
