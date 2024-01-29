# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from torch import nn
from diffusers import StableDiffusionPipeline

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_unet_mid_block_2d_cross_attn import (
    unet_mid_block_2d_cross_attn,
)


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
    "hidden_state_shapes,",
    [
        (
            2,
            1280,
            4,
            4,
        ),
    ],
)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
def test_unet_mid_block_2d_cross_attn_256x256(device, model_name, hidden_state_shapes, reset_seeds):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    config = unet.config
    mid_block = pipe.unet.mid_block

    temb_shape = [1, 1, 2, 1280]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    attention_mask = None
    cross_attention_kwargs = None

    timestep = None
    class_labels = None
    return_dict = True

    resnet_groups = 32
    resnet_eps = 1e-05
    resnet_act_fn = "silu"
    resnet_pre_norm = True
    output_scale_factor = 1
    attn_num_head_channels = 8
    resnet_time_scale_shift = "default"
    upcast_attention = False
    dual_cross_attention = False
    use_linear_projection = False

    _, in_channels, _, _ = hidden_state_shapes
    _, _, _, temb_channels = temb_shape

    hidden_states = torch.randn(hidden_state_shapes)
    temb = torch.randn(temb_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = mid_block(
        hidden_states,
        temb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.mid_block

    ttnn_hidden_state = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    ttnn_encoder_hidden_states = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    ttnn_temb = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(temb, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    ttnn_mid_block = unet_mid_block_2d_cross_attn(
        temb=ttnn_temb,
        parameters=parameters,
        in_channels=in_channels,
        temb_channels=temb_channels,
        hidden_states=ttnn_hidden_state,
        config=config,
        timestep=timestep,
        resnet_eps=resnet_eps,
        class_labels=class_labels,
        return_dict=return_dict,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        resnet_time_scale_shift=resnet_time_scale_shift,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        resnet_pre_norm=resnet_pre_norm,
        attn_num_head_channels=attn_num_head_channels,
        output_scale_factor=output_scale_factor,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        upcast_attention=upcast_attention,
        device=device,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn.to_layout(ttnn.from_device(ttnn_mid_block), layout=ttnn.ROW_MAJOR_LAYOUT))

    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)


@pytest.mark.parametrize(
    "hidden_state_shapes,",
    [
        (
            2,
            1280,
            8,
            8,
        ),
    ],
)
@pytest.mark.parametrize("model_name", ["CompVis/stable-diffusion-v1-4"])
def test_unet_mid_block_2d_cross_attn_512x512(device, model_name, hidden_state_shapes, reset_seeds):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    unet = pipe.unet
    unet.eval()
    config = unet.config
    mid_block = pipe.unet.mid_block

    temb_shape = [1, 1, 2, 1280]
    encoder_hidden_states_shape = [1, 2, 77, 768]
    attention_mask = None
    cross_attention_kwargs = None

    timestep = None
    class_labels = None
    return_dict = True

    num_layers = 1
    resnet_groups = 32
    resnet_eps = 1e-05
    resnet_act_fn = "silu"
    resnet_pre_norm = True
    output_scale_factor = 1
    attn_num_head_channels = 8
    resnet_time_scale_shift = "default"
    upcast_attention = False
    dual_cross_attention = False
    use_linear_projection = False

    _, in_channels, _, _ = hidden_state_shapes
    # _, _, _, temb_channels = temb_shape
    temb_channels = 1280

    hidden_states = torch.randn(hidden_state_shapes)
    temb = torch.randn(temb_shape)
    encoder_hidden_states = torch.randn(encoder_hidden_states_shape)

    torch_output = mid_block(
        hidden_states,
        temb.squeeze(0).squeeze(0),
        encoder_hidden_states=encoder_hidden_states.squeeze(0),
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: unet, custom_preprocessor=custom_preprocessor, device=device
    )
    parameters = parameters.mid_block

    ttnn_hidden_state = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    ttnn_encoder_hidden_states = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(encoder_hidden_states, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    ttnn_temb = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(temb, dtype=ttnn.bfloat16), device), layout=ttnn.TILE_LAYOUT
    )

    ttnn_mid_block = unet_mid_block_2d_cross_attn(
        temb=ttnn_temb,
        parameters=parameters,
        in_channels=in_channels,
        temb_channels=temb_channels,
        hidden_states=ttnn_hidden_state,
        num_layers=num_layers,
        config=config,
        timestep=timestep,
        resnet_eps=resnet_eps,
        class_labels=class_labels,
        return_dict=return_dict,
        encoder_hidden_states=ttnn_encoder_hidden_states,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        resnet_time_scale_shift=resnet_time_scale_shift,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        resnet_pre_norm=resnet_pre_norm,
        attn_num_head_channels=attn_num_head_channels,
        output_scale_factor=output_scale_factor,
        dual_cross_attention=dual_cross_attention,
        use_linear_projection=use_linear_projection,
        upcast_attention=upcast_attention,
        device=device,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn.to_layout(ttnn.from_device(ttnn_mid_block), layout=ttnn.ROW_MAJOR_LAYOUT))

    assert_with_pcc(torch_output, ttnn_output_torch, 0.99)
