# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_resnetblock2d import resnetBlock2D
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_transformer_2d import transformer_2d_model
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_downsample_2d import downsample_2d


def cross_attention_down_block_2d(
    hidden_states,
    encoder_hidden_states,
    temb,
    add_downsample=True,
    attention_mask=None,
    cross_attention_kwargs={},
    config=None,
    num_layers=2,
    dual_cross_attention=False,
    temb_channels=1280,
    groups=32,
    time_embedding_norm="default",
    output_scale_factor=1.0,
    use_in_shortcut=False,
    *,
    parameters,
    device,
):
    output_states = ()
    _, in_channels, _, _ = hidden_states.shape

    for resnet, attn in zip(parameters.resnets, parameters.attentions):
        use_in_shortcut = True if "conv_shortcut" in resnet else False
        hidden_states = resnetBlock2D(
            hidden_states,
            temb=temb,
            in_channels=in_channels,
            parameters=resnet,
            device=device,
            use_in_shortcut=use_in_shortcut,
        )

        hidden_states = transformer_2d_model(
            hidden_states,
            attn,
            config,
            encoder_hidden_states,
            in_channels=in_channels,
            out_channels=in_channels,
            device=device,
        )

        output_states += (hidden_states,)

    if add_downsample is not None:
        hidden_states = downsample_2d(
            in_channels=hidden_states.shape[1],
            out_channels=hidden_states.shape[1],
            hidden_states=hidden_states,
            device=device,
            parameters=parameters.downsamplers[0],
            use_conv=True,
        )
        output_states += (hidden_states,)

    return hidden_states, output_states
