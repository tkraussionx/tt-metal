# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def prepare_attention_mask(attention_mask, target_length, heads=8):
    head_size = heads
    if attention_mask is None:
        return attention_mask

    if attention_mask.shape != target_length:
        assert False, "Attention Mask has always been None, This is not implemented!"

    return attention_mask


def batch_to_head_dim(tensor, heads=8):
    head_size = heads
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.to_layout(
        tensor, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # using ROW_MAJOR_LAYOUT as the tensor shape is not compatible for using TILE_LAYOUT
    tensor = ttnn.reshape(tensor, (batch_size // head_size, head_size, seq_len, dim))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (1, batch_size // head_size, seq_len, dim * head_size))
    return tensor


def head_to_batch_dim(tensor, heads=8):
    head_size = heads
    _, batch_size, seq_len, dim = tensor.shape
    tensor = ttnn.to_layout(
        tensor, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # using ROW_MAJOR_LAYOUT as the tensor shape is not compatible for using TILE_LAYOUT
    tensor = ttnn.reshape(tensor, (batch_size, seq_len, head_size, dim // head_size))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (1, batch_size * head_size, seq_len, dim // head_size))
    return tensor


def get_attention_scores(query, key, attention_mask=None, has_encoder_hidden_states=False, scale=64, device=None):
    if has_encoder_hidden_states:
        key = ttnn.pad(key, ((0, 1), (0, 0)), 0)
    key = ttnn.to_device(key, device)
    t_key = ttnn.permute(key, (0, 1, 3, 2))
    temp = ttnn.matmul(query, t_key)

    scale_tensor = query @ t_key  # instead of full
    scale_tensor = ttnn.to_layout(scale_tensor, ttnn.TILE_LAYOUT)
    scale_tensor = ttnn.mul(scale_tensor, scale)

    temp = ttnn.to_layout(temp, ttnn.TILE_LAYOUT)
    attention_scores = ttnn.mul(scale_tensor, temp)

    if attention_mask is not None:
        attention_scores = ttnn.add(attention_scores, attention_mask)

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    return attention_probs


def cross_attention(
    hidden_states,
    encoder_hidden_states,
    attention_mask=None,
    cross_attention_kwargs={},
    *,
    parameters,
    device,
):
    _, batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = prepare_attention_mask(attention_mask, sequence_length)
    query_weight = parameters.to_q.weight

    query = ttnn.matmul(hidden_states, query_weight)

    query = head_to_batch_dim(query)
    has_encoder_hidden_states = False if encoder_hidden_states is None else True
    encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

    key_weight = parameters.to_k.weight
    key = ttnn.matmul(encoder_hidden_states, key_weight)

    if len(key.shape) <= 3:
        key = ttnn.from_device(key)
        key = ttnn.to_torch(key).unsqueeze(0)
        key = ttnn.from_torch(key)
        key = ttnn.to_device(key, device)

    value_weight = parameters.to_v.weight
    value = ttnn.matmul(encoder_hidden_states, value_weight)

    if len(key.shape) <= 3:
        value = ttnn.from_device(value)
        value = ttnn.to_torch(value).unsqueeze(0)
        value = ttnn.from_torch(value)
        value = ttnn.to_device(value, device)

    key = head_to_batch_dim(key)

    value = head_to_batch_dim(value)
    value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

    attention_probs = get_attention_scores(query, key, attention_mask, has_encoder_hidden_states, device=device)

    if has_encoder_hidden_states:
        value = ttnn.pad(value, ((0, 1), (0, 0)), 0)

    if not has_encoder_hidden_states:
        attention_probs = ttnn.permute(attention_probs, (0, 1, 3, 2))

    hidden_states = ttnn.matmul(attention_probs, value)

    hidden_states = batch_to_head_dim(hidden_states)

    out_weight = parameters.to_out[0].weight

    padding_needed = out_weight.shape[-1] - hidden_states.shape[-1]
    hidden_states = ttnn.pad(hidden_states, ((0, 0), (0, padding_needed)), 0)

    hidden_states = ttnn.matmul(hidden_states, out_weight)
    if parameters.to_out[0].bias is not None:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        out_bias = parameters.to_out[0].bias
        hidden_states = ttnn.add(hidden_states, out_bias)

    return hidden_states
