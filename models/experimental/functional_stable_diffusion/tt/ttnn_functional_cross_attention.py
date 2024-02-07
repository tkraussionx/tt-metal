# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.functional_stable_diffusion.configuration_file import PYTORCH_FALLBACK_OPS


def prepare_attention_mask(attention_mask, target_length, heads=8):
    head_size = heads
    if attention_mask is None:
        return attention_mask

    if attention_mask.shape[-1] != target_length:
        assert False, "Attention Mask has always been None, This is not implemented!"

    return attention_mask


def batch_to_head_dim(tensor, heads=8, device=None):
    head_size = heads
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.to_layout(
        tensor, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # TILE_LAYOUT is not compatible with tensor shape, hence we used ROW_MAJOR_LAYOUT.
    if PYTORCH_FALLBACK_OPS["reshape"]:
        tensor = ttnn.to_torch(tensor)
        tensor = torch.reshape(tensor, (batch_size // head_size, head_size, seq_len, dim))
        tensor = ttnn.from_torch(tensor, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        tensor = ttnn.reshape(tensor, (batch_size // head_size, head_size, seq_len, dim))

    if PYTORCH_FALLBACK_OPS["permute"]:
        tensor = ttnn.to_torch(tensor)
        tensor = torch.permute(tensor, (0, 2, 1, 3))
        tensor = ttnn.from_torch(tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        tensor = ttnn.permute(tensor, (0, 2, 1, 3))

    if PYTORCH_FALLBACK_OPS["reshape"]:
        tensor = ttnn.to_torch(tensor)
        tensor = torch.reshape(tensor, (1, batch_size // head_size, seq_len, dim * head_size))
        tensor = ttnn.from_torch(tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        tensor = ttnn.reshape(tensor, (1, batch_size // head_size, seq_len, dim * head_size))

    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return tensor


def head_to_batch_dim(tensor, heads=8, device=None):
    head_size = heads
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.to_layout(
        tensor, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # TILE_LAYOUT is not compatible with tensor shape, hence we used ROW_MAJOR_LAYOUT.

    if PYTORCH_FALLBACK_OPS["reshape"]:
        tensor = ttnn.to_torch(tensor)
        tensor = torch.reshape(tensor, (batch_size, seq_len, head_size, dim // head_size))
        tensor = ttnn.from_torch(tensor, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        tensor = ttnn.reshape(tensor, (batch_size, seq_len, head_size, dim // head_size))

    if PYTORCH_FALLBACK_OPS["permute"]:
        tensor = ttnn.to_torch(tensor)
        tensor = torch.permute(tensor, (0, 2, 1, 3))
        tensor = ttnn.from_torch(tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        tensor = ttnn.permute(tensor, (0, 2, 1, 3))

    if PYTORCH_FALLBACK_OPS["reshape"]:
        tensor = ttnn.to_torch(tensor)
        tensor = torch.reshape(tensor, (1, batch_size * head_size, seq_len, dim // head_size))
        tensor = ttnn.from_torch(tensor, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        tensor = ttnn.reshape(tensor, (1, batch_size * head_size, seq_len, dim // head_size))

    tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    return tensor


def get_attention_scores(query, key, attention_mask=None, scale=None, device=None):
    if PYTORCH_FALLBACK_OPS["permute"]:
        key = ttnn.to_torch(key)
        t_key = torch.permute(key, (0, 1, 3, 2))
        t_key = ttnn.from_torch(t_key, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        t_key = ttnn.permute(key, (0, 1, 3, 2))

    if PYTORCH_FALLBACK_OPS["bmm"]:
        query = ttnn.to_torch(query)
        t_key = ttnn.to_torch(t_key)
        temp = torch.matmul(query, t_key)
        temp = ttnn.from_torch(temp, device=device, layout=ttnn.TILE_LAYOUT)
        query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT)
        t_key = ttnn.from_torch(t_key, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        temp = ttnn.matmul(query, t_key)

    if PYTORCH_FALLBACK_OPS["mul"]:
        temp = ttnn.to_torch(temp)
        attention_scores = torch.mul(temp, scale)
        temp = ttnn.from_torch(temp, device=device, layout=ttnn.TILE_LAYOUT)
        attention_scores = ttnn.from_torch(attention_scores, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        attention_scores = ttnn.mul(temp, scale)

    if attention_mask is not None:
        if PYTORCH_FALLBACK_OPS["add"]:
            attention_scores = ttnn.to_torch(attention_scores)
            attention_mask = ttnn.to_torch(attention_mask)
            attention_probs = torch.add(attention_scores, attention_mask)
            attention_probs = ttnn.from_torch(attention_probs, device=device, layout=ttnn.TILE_LAYOUT)
            attention_scores = ttnn.from_torch(attention_scores, device=device, layout=ttnn.TILE_LAYOUT)
            attention_mask = ttnn.from_torch(attention_mask, device=device, layout=ttnn.TILE_LAYOUT)
        else:
            attention_scores = ttnn.add(attention_scores, attention_mask)

    if PYTORCH_FALLBACK_OPS["softmax"]:
        attention_scores = ttnn.to_torch(attention_scores)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = ttnn.from_torch(attention_probs, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        attention_probs = ttnn.softmax(attention_scores, dim=-1)

    return attention_probs


def cross_attention(
    hidden_states,
    encoder_hidden_states,
    query_dim: int = None,
    cross_attention_dim=None,
    heads: int = 8,
    dim_head: int = 64,
    attention_mask=None,
    upcast_attention: bool = False,
    upcast_softmax: bool = False,
    cross_attention_kwargs={},
    *,
    parameters,
    device,
):
    _, batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = prepare_attention_mask(attention_mask, sequence_length)
    query_weight = parameters.to_q.weight

    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

    if PYTORCH_FALLBACK_OPS["bmm"]:
        hidden_states = ttnn.to_torch(hidden_states)
        query_weight = ttnn.to_torch(query_weight)
        query = torch.matmul(hidden_states, query_weight)
        hidden_states = ttnn.from_torch(hidden_states, device=device, layout=ttnn.TILE_LAYOUT)
        query_weight = ttnn.from_torch(query_weight, device=device, layout=ttnn.TILE_LAYOUT)
        query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        query = ttnn.matmul(hidden_states, query_weight)

    query = head_to_batch_dim(query, device=device)
    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

    key_weight = parameters.to_k.weight
    encoder_hidden_states = ttnn.to_layout(encoder_hidden_states, ttnn.TILE_LAYOUT)

    if PYTORCH_FALLBACK_OPS["bmm"]:
        encoder_hidden_states = ttnn.to_torch(encoder_hidden_states)
        key_weight = ttnn.to_torch(key_weight)
        key = torch.matmul(encoder_hidden_states, key_weight)
        encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, device=device, layout=ttnn.TILE_LAYOUT)
        key_weight = ttnn.from_torch(key_weight, device=device, layout=ttnn.TILE_LAYOUT)
        key = ttnn.from_torch(key, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        key = ttnn.matmul(encoder_hidden_states, key_weight)

    if len(key.shape) <= 3:
        key = ttnn.from_device(key)
        key = ttnn.to_torch(key).unsqueeze(0)
        key = ttnn.from_torch(key)
        key = ttnn.to_device(key, device)

    value_weight = parameters.to_v.weight

    if PYTORCH_FALLBACK_OPS["bmm"]:
        encoder_hidden_states = ttnn.to_torch(encoder_hidden_states)
        value_weight = ttnn.to_torch(value_weight)
        value = torch.matmul(encoder_hidden_states, value_weight)
        encoder_hidden_states = ttnn.from_torch(encoder_hidden_states, device=device, layout=ttnn.TILE_LAYOUT)
        value_weight = ttnn.from_torch(value_weight, device=device, layout=ttnn.TILE_LAYOUT)
        value = ttnn.from_torch(value, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        value = ttnn.matmul(encoder_hidden_states, value_weight)

    if len(value.shape) <= 3:
        value = ttnn.from_device(value)
        value = ttnn.to_torch(value).unsqueeze(0)
        value = ttnn.from_torch(value)
        value = ttnn.to_device(value, device)

    key = head_to_batch_dim(key, device=device)

    value = head_to_batch_dim(value, device=device)

    scale = dim_head**-0.5
    attention_probs = get_attention_scores(query, key, attention_mask, scale=scale, device=device)

    padding_needed = attention_probs.shape[-1] - value.shape[-2]
    if PYTORCH_FALLBACK_OPS["pad"]:
        value = ttnn.to_torch(value)
        value = torch.nn.functional.pad(value, (0, padding_needed, 0, 0), value=0)
        value = ttnn.from_torch(value, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        value = ttnn.pad(value, ((0, padding_needed), (0, 0)), 0)

    if PYTORCH_FALLBACK_OPS["bmm"]:
        attention_probs = ttnn.to_torch(attention_probs)
        value = ttnn.to_torch(value)
        hidden_states = torch.matmul(attention_probs, value)
        attention_probs = ttnn.from_torch(attention_probs, device=device, layout=ttnn.TILE_LAYOUT)
        value = ttnn.from_torch(value, device=device, layout=ttnn.TILE_LAYOUT)
        hidden_states = ttnn.from_torch(hidden_states, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        hidden_states = ttnn.matmul(attention_probs, value)

    hidden_states = batch_to_head_dim(hidden_states, device=device)

    out_weight = parameters.to_out[0].weight

    if PYTORCH_FALLBACK_OPS["bmm"]:
        hidden_states = ttnn.to_torch(hidden_states)
        out_weight = ttnn.to_torch(out_weight)
        hidden_states = torch.matmul(hidden_states, out_weight)
        out_weight = ttnn.from_torch(out_weight, device=device, layout=ttnn.TILE_LAYOUT)
        hidden_states = ttnn.from_torch(hidden_states, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        hidden_states = ttnn.matmul(hidden_states, out_weight)

    if parameters.to_out[0].bias is not None:
        out_bias = parameters.to_out[0].bias
        if PYTORCH_FALLBACK_OPS["add"]:
            hidden_states = ttnn.to_torch(hidden_states)
            out_bias = ttnn.to_torch(out_bias)
            hidden_states = torch.add(hidden_states, out_bias)
            hidden_states = ttnn.from_torch(hidden_states, device=device, layout=ttnn.TILE_LAYOUT)
            out_bias = ttnn.from_torch(out_bias, device=device, layout=ttnn.TILE_LAYOUT)
        else:
            hidden_states = ttnn.add(hidden_states, out_bias)

    return hidden_states
