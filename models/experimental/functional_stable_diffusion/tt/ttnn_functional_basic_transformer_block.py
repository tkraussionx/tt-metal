# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_cross_attention import cross_attention
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_feed_forward import feedforward


def basic_transformer_block(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    class_labels=None,
    config=None,
    *,
    parameters,
    device,
):
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    norm_hidden_states = ttnn.layer_norm(
        hidden_states, epsilon=1e-05, weight=parameters.norm1.weight, bias=parameters.norm1.bias
    )

    # 1. Self-Attention
    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

    attn_output = cross_attention(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if config.only_cross_attention else None,
        attention_mask=None,
        cross_attention_kwargs=cross_attention_kwargs,
        parameters=parameters.attn1,
        device=device,
    )

    hidden_states = ttnn.add(attn_output, hidden_states)
    if parameters.attn2 is not None:
        norm_hidden_states = ttnn.layer_norm(
            hidden_states, epsilon=1e-05, weight=parameters.norm2.weight, bias=parameters.norm2.bias
        )

        # 2. Cross-Attention
        attn_output = cross_attention(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            cross_attention_kwargs=cross_attention_kwargs,
            parameters=parameters.attn2,
            device=device,
        )

        hidden_states = ttnn.add(attn_output, hidden_states)

    # 3. Feed-forward
    norm_hidden_states = ttnn.layer_norm(
        hidden_states, epsilon=1e-05, weight=parameters.norm3.weight, bias=parameters.norm3.bias
    )
    ff_output = feedforward(config=config, hidden_states=norm_hidden_states, parameters=parameters.ff)

    hidden_states = ttnn.add(ff_output, hidden_states)

    return hidden_states
