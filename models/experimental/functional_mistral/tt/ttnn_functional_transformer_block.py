# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import tt_lib
from models.experimental.functional_mistral.tt.ttnn_functional_rms_norm import rms_norm
from models.experimental.functional_mistral.tt.ttnn_functional_attention import attention
from models.experimental.functional_mistral.tt.ttnn_functional_feed_forward import feed_forward


def transformer_block(
    config,
    x,
    bcast_freq_xq: tt_lib.tensor.complex_tensor,
    bcast_freq_xk: tt_lib.tensor.complex_tensor,
    positions,
    mask,
    seqlen,
    parameter,
    device,
    memory_config,
):
    att_weight = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(parameter.attention_norm.weight, dtype=ttnn.bfloat16), device),
        layout=ttnn.TILE_LAYOUT,
    )

    r = attention(
        config,
        rms_norm(config, input=x, parameters=att_weight),
        bcast_freq_xq,
        bcast_freq_xk,
        positions,
        mask,
        seqlen,
        parameter.attention,
        device,
        memory_config,
    )
    h = x + r
    att_weight = ttnn.to_layout(
        ttnn.to_device(ttnn.from_torch(parameter.ffn_norm.weight, dtype=ttnn.bfloat16), device),
        layout=ttnn.TILE_LAYOUT,
    )
    r = feed_forward(config, x=rms_norm(config, input=x, parameters=att_weight), parameters=parameter.feed_forward)
    del att_weight
    return h + r
