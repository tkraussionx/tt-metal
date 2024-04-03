# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Optional

import torch
import math
import ttnn

from models.experimental.functional_common.attention_mask_functions import (
    get_extended_attention_mask,
    invert_attention_mask,
)


def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def compute_bias(query_length, key_length, config=None, device=None):
    context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = _relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not config.is_decoder),
        num_buckets=config.relative_attention_num_buckets,
        max_distance=config.relative_attention_max_distance,
    )
    values = relative_position_bucket
    # shape (1, num_heads, query_length, key_length)
    return values


def t5_layer_norm(config, hidden_states, *, weight, iteration=-1, device=None, flag=0, base_address=None):
    # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
    # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
    # half-precision inputs is done in fp32

    # return ttnn.rms_norm(hidden_states, weight, epsilon=config.layer_norm_epsilon)

    squared_hidden_states = ttnn.pow(hidden_states, 2)
    torch.save(
        ttnn.to_torch(squared_hidden_states),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "pow.pt",
    )
    averaged_squared_hidden_states = ttnn.mean(
        squared_hidden_states,
        dim=-1,
        keepdim=True,
    )
    torch.save(
        ttnn.to_torch(averaged_squared_hidden_states),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "mean.pt",
    )

    variance = averaged_squared_hidden_states + config.layer_norm_epsilon

    std = ttnn.rsqrt(variance)
    torch.save(ttnn.to_torch(std), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "sqrt.pt")

    hidden_states = hidden_states * std
    hidden_states = hidden_states * weight
    torch.save(
        ttnn.to_torch(hidden_states), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "mul.pt"
    )
    return hidden_states


def get_activation_function(dense_act_fn):
    if dense_act_fn == "relu":
        return ttnn.relu
    elif dense_act_fn == "gelu_new":
        return ttnn.gelu
    else:
        raise RuntimeError(f"Unsupported activation function: {dense_act_fn}")


def t5_dense_act_dense(config, hidden_states, parameters, device=None, iteration=None, base_address=None):
    if config.dense_act_fn == "relu":
        ff1_activation = "relu"
    elif config.dense_act_fn == "gelu_new":
        ff1_activation = "gelu"
    else:
        raise RuntimeError(f"Unsupported activation function: {config.dense_act_fn}")

    _, height, _ = hidden_states.shape
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.wi.weight,
        dtype=ttnn.bfloat8_b,
        # activation=ff1_activation,
        core_grid=ttnn.CoreGrid(y=height // 32, x=12),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    torch.save(
        ttnn.to_torch(hidden_states), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "wi.pt"
    )

    if config.dense_act_fn == "gelu_new":
        hidden_states = ttnn.gelu(hidden_states)
    else:
        hidden_states = ttnn.relu(hidden_states)

    torch.save(
        ttnn.to_torch(hidden_states),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "activation.pt",
    )

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.wo.weight,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=9, x=12),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    torch.save(
        ttnn.to_torch(hidden_states), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "wo.pt"
    )

    return hidden_states


def t5_dense_gated_act_dense(config, hidden_states, parameters, device=None, base_address=None):
    activation_function = get_activation_function(config.dense_act_fn)

    hidden_gelu = hidden_states @ parameters.wi_0.weight
    hidden_gelu = activation_function(hidden_gelu)
    hidden_linear = hidden_states @ parameters.wi_1.weight
    hidden_states = hidden_gelu * hidden_linear

    hidden_states = hidden_states @ parameters.wo.weight
    return hidden_states


def t5_layer_ff(config, hidden_states, parameters, iteration=0, device=None, base_address=None):
    forwarded_states = t5_layer_norm(
        config,
        hidden_states,
        weight=parameters.layer_norm.weight,
        iteration=iteration,
        device=device,
        base_address=base_address + "layer_norm.",
    )
    # torch.save(ttnn.to_torch(forwarded_states),"tests/ttnn/integration_tests/t5/t5_ttnn_outputs/"+base_address+"layernorm_output.pt")
    if config.is_gated_act:
        forwarded_states = t5_dense_gated_act_dense(
            config,
            forwarded_states,
            parameters.DenseReluDense,
            device=device,
            base_address=base_address + "DenseReluDense.",
        )
        # torch.save(ttnn.to_torch(forwarded_states),"tests/ttnn/integration_tests/t5/t5_ttnn_outputs/"+base_address+"DenseReluDense.output.pt")

    else:
        forwarded_states = t5_dense_act_dense(
            config,
            forwarded_states,
            parameters.DenseReluDense,
            device=device,
            base_address=base_address + "DenseReluDense.",
            iteration=iteration,
        )
        # torch.save(ttnn.to_torch(forwarded_states),"tests/ttnn/integration_tests/t5/t5_ttnn_outputs/"+base_address+"DenseReluDense.output.pt")

    hidden_states = ttnn.add(hidden_states, forwarded_states, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch.save(
        ttnn.to_torch(forwarded_states),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "_layer_ff_add.pt",
    )
    return hidden_states


def t5_attention(
    config,
    hidden_states,
    key_value_states=None,
    mask=None,
    layer_head_mask=None,
    *,
    parameters,
    num_cores_x=12,
    device=None,
    has_relative_attention_bias=None,
    relative_attention_bias_weight=None,
    position_bias=None,
    iteration=None,
    base_address=None,
):
    batch_size, real_seq_length, *_ = hidden_states.shape
    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    if key_value_states is None:
        query_key_value_output = ttnn.linear(
            hidden_states,
            parameters.query_key_value.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            query_key_value_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_heads=config.num_heads,
        )
        ttnn.deallocate(query_key_value_output)

    else:
        query_proj = ttnn.linear(
            hidden_states,
            parameters.q.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )

        key_value_proj = ttnn.linear(
            key_value_states,
            parameters.key_value.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )

        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            query_proj, key_value_proj, num_heads=config.num_heads
        )
        ttnn.deallocate(query_proj)
        ttnn.deallocate(key_value_proj)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    torch.save(
        ttnn.to_torch(query), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "matmul_1_query.pt"
    )
    torch.save(
        ttnn.to_torch(key), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "matmul_1_key.pt"
    )
    torch.save(
        ttnn.to_torch(attention_scores),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "matmul_1.pt",
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    if position_bias is None:
        if not has_relative_attention_bias:
            position_bias = torch.zeros((1, config.num_heads, real_seq_length, key_length), dtype=torch.float32)
        else:
            relative_position_bucket = compute_bias(real_seq_length, key_length, config=config)

            relative_attention_bias_embedding = torch.nn.Embedding(
                config.relative_attention_num_buckets, config.num_heads
            )
            relative_attention_bias_embedding.weight.data = ttnn.to_torch(relative_attention_bias_weight)
            values = relative_attention_bias_embedding(relative_position_bucket)

            values = ttnn.from_torch(values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            values = ttnn.permute(values, (2, 0, 1))
            position_bias = ttnn.to_torch(values).unsqueeze(0)

    if mask is not None:
        mask_torch = ttnn.to_torch(mask)
        position_bias = position_bias + mask_torch  # (batch_size, n_heads, seq_length, key_length)
        position_bias = ttnn.from_torch(position_bias, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    torch.save(
        ttnn.to_torch(attention_scores),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "addd_attention_scores.pt",
    )
    attention_scores = ttnn.add(attention_scores, position_bias)
    torch.save(
        ttnn.to_torch(position_bias),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "addd_position_biasposition_bias.pt",
    )
    torch.save(
        ttnn.to_torch(attention_scores), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "addd.pt"
    )

    if mask is None:
        attention_probs = ttnn.softmax(attention_scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        # attention_scores = ttnn.add(attention_scores, mask)
        attention_scores = torch.add(
            ttnn.to_torch(attention_scores), ttnn.to_torch(mask)
        )  # used torch add for shape compatibility
        attention_scores = ttnn.from_torch(
            attention_scores, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        attention_probs = ttnn.softmax(
            attention_scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG
        )  # gives improvement in PCC
        # attention_probs = ttnn.transformer.attention_softmax_(attention_scores, attention_mask=mask, head_size=None)
    torch.save(
        ttnn.to_torch(attention_probs), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "softmax.pt"
    )

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        # dtype=ttnn.bfloat8_b,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    torch.save(
        ttnn.to_torch(attention_probs),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "matmul_2_attention_probs.pt",
    )
    torch.save(
        ttnn.to_torch(value), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "matmul_2_value.pt"
    )
    torch.save(
        ttnn.to_torch(context_layer), "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "matmul_2.pt"
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    self_output = ttnn.linear(
        context_layer,
        parameters.o.weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    ttnn.deallocate(context_layer)

    return self_output


def t5_layer_self_attention(
    config,
    hidden_states,
    attention_mask=None,
    *,
    parameters,
    iteration=0,
    device=None,
    has_relative_attention_bias=None,
    relative_attention_bias_weight=None,
    base_address=None,
):
    normed_hidden_states = t5_layer_norm(
        config,
        hidden_states,
        weight=parameters.layer_norm.weight,
        device=device,
        base_address=base_address + "layer_norm.",
    )
    # torch.save(ttnn.to_torch(hidden_states),"tests/ttnn/integration_tests/t5/t5_ttnn_outputs/"+base_address+"layernorm_output.pt")

    attention_output = t5_attention(
        config,
        normed_hidden_states,
        mask=attention_mask,
        parameters=parameters.SelfAttention,
        device=device,
        has_relative_attention_bias=has_relative_attention_bias,
        relative_attention_bias_weight=relative_attention_bias_weight,
        iteration=iteration,
        base_address=base_address + "SelfAttention.",
    )
    # torch.save(ttnn.to_torch(hidden_states),"tests/ttnn/integration_tests/t5/t5_ttnn_outputs/"+base_address+"self_attention_attention_output.pt")

    hidden_states = ttnn.add(hidden_states, attention_output, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch.save(
        ttnn.to_torch(hidden_states),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "self_attention_add.pt",
    )

    return hidden_states


def t5_layer_cross_attention(
    config,
    hidden_states,
    key_value_states,
    attention_mask=None,
    *,
    parameters,
    iteration=0,
    device=None,
    base_address=None,
):
    normed_hidden_states = t5_layer_norm(
        config,
        hidden_states,
        weight=parameters.layer_norm.weight,
        device=device,
        base_address=base_address + "layer_norm.",
    )
    # torch.save(ttnn.to_torch(normed_hidden_states),"tests/ttnn/integration_tests/t5/t5_ttnn_outputs/"+base_address+"layernorm_output.pt")

    attention_output = t5_attention(
        config,
        normed_hidden_states,
        key_value_states=key_value_states,
        mask=attention_mask,
        parameters=parameters.EncDecAttention,
        device=device,
        iteration=iteration,
        base_address=base_address + "EncDecAttention.",
    )
    # torch.save(ttnn.to_torch(attention_output),"tests/ttnn/integration_tests/t5/t5_ttnn_outputs/"+base_address+"attention_output.pt")

    layer_output = ttnn.add(hidden_states, attention_output, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch.save(
        ttnn.to_torch(layer_output),
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/" + base_address + "cross_attention_add.pt",
    )
    return layer_output


def t5_block(
    config,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    *,
    parameters,
    iteration=0,
    device=None,
    has_relative_attention_bias=None,
    relative_attention_bias_weight=None,
    base_address=None,
):
    layer_cnt = 1
    hidden_states = t5_layer_self_attention(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters.layer[0],
        iteration=iteration,
        device=device,
        has_relative_attention_bias=has_relative_attention_bias,
        relative_attention_bias_weight=relative_attention_bias_weight,
        base_address=base_address + "layer.0.",
    )

    do_cross_attention = encoder_hidden_states is not None
    if do_cross_attention:
        hidden_states = t5_layer_cross_attention(
            config,
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            parameters=parameters.layer[1],
            iteration=iteration,
            device=device,
            base_address=base_address + "layer.1.",
        )
        layer_cnt += 1
    # Apply Feed Forward layer

    hidden_states = t5_layer_ff(
        config,
        hidden_states,
        parameters.layer[-1],
        iteration=iteration,
        device=device,
        base_address=base_address + "layer." + str(len(parameters.layer) - 1) + ".",
    )

    return hidden_states  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


def t5_stack(
    config,
    input_ids,
    shared_embedding_weight,
    encoder_hidden_states=None,
    *,
    parameters,
    device=None,
    relative_attention_bias_weight=None,
    base_address=None,
):
    input_shape = tuple(input_ids.shape)

    hidden_states = ttnn.embedding(
        input_ids, shared_embedding_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    attention_mask = create_attention_mask(
        input_shape, input_ids.device(), is_decoder=encoder_hidden_states is not None
    )
    if encoder_hidden_states is not None:
        encoder_attention_mask = create_encoder_attention_mask(input_shape, input_ids.device())
    else:
        encoder_attention_mask = None
    iteration = 0
    for block_parameters in parameters.block:
        hidden_states = t5_block(
            config,
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            parameters=block_parameters,
            iteration=iteration,
            device=device,
            has_relative_attention_bias=bool(iteration == 0),
            relative_attention_bias_weight=relative_attention_bias_weight,
            base_address=base_address + "block." + str(iteration) + ".",
        )
        iteration += 1
    hidden_states = t5_layer_norm(
        config,
        hidden_states,
        weight=parameters.final_layer_norm.weight,
        flag=1,
        device=device,
        base_address=base_address + "final_layer_norm.",
    )
    return hidden_states


def t5_for_conditional_generation(
    config,
    input_ids: Optional[torch.LongTensor],
    decoder_input_ids: Optional[torch.LongTensor],
    parameters,
    *,
    encoder_last_hidden_state=None,
    device=None,
    relative_attention_bias_weight_encoder=None,
    relative_attention_bias_weight_decoder=None,
) -> torch.FloatTensor:
    # Encode
    if encoder_last_hidden_state is None:
        encoder_last_hidden_state = t5_stack(
            config,
            input_ids=input_ids,
            shared_embedding_weight=parameters.shared.weight,
            parameters=parameters.encoder,
            device=device,
            relative_attention_bias_weight=relative_attention_bias_weight_encoder,
            base_address="encoder.",
        )

    # Decode
    sequence_output = t5_stack(
        config,
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_last_hidden_state,
        shared_embedding_weight=parameters.shared.weight,
        parameters=parameters.decoder,
        device=device,
        relative_attention_bias_weight=relative_attention_bias_weight_decoder,
        base_address="decoder.",
    )

    lm_logits = ttnn.linear(sequence_output, parameters.lm_head.weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    return lm_logits, encoder_last_hidden_state


@functools.lru_cache
def create_attention_mask(input_shape, device, is_decoder):
    batch_size, seq_length = input_shape

    attention_mask = torch.ones(batch_size, seq_length)

    extended_attention_mask = get_extended_attention_mask(
        attention_mask, input_shape, is_decoder=is_decoder, dtype=torch.bfloat16
    )

    extended_attention_mask = extended_attention_mask.expand((-1, -1, seq_length, -1))
    extended_attention_mask = ttnn.from_torch(extended_attention_mask)
    extended_attention_mask = ttnn.to_layout(extended_attention_mask, ttnn.TILE_LAYOUT)
    extended_attention_mask = ttnn.to_device(extended_attention_mask, device)
    return extended_attention_mask


@functools.lru_cache
def create_encoder_attention_mask(input_shape, device):
    batch_size, seq_length = input_shape

    encoder_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)

    encoder_extended_attention_mask = encoder_extended_attention_mask.expand((-1, -1, seq_length, -1))
    encoder_extended_attention_mask = ttnn.from_torch(encoder_extended_attention_mask)
    encoder_extended_attention_mask = ttnn.to_layout(encoder_extended_attention_mask, ttnn.TILE_LAYOUT)
    encoder_extended_attention_mask = ttnn.to_device(encoder_extended_attention_mask, device)
    return encoder_extended_attention_mask


def custom_preprocessor(model, name):
    import transformers
    from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_layernorm_parameter

    parameters = {}
    if isinstance(model, transformers.models.t5.modeling_t5.T5LayerNorm):
        parameters["weight"] = preprocess_layernorm_parameter(model.weight, dtype=ttnn.bfloat16)

    elif isinstance(model, transformers.models.t5.modeling_t5.T5Attention):
        if "EncDecAttention" in name:
            # Cross Attention
            preprocessed_kv_weight = torch.cat([model.k.weight, model.v.weight], dim=0)
            parameters = {
                "q": {"weight": preprocess_linear_weight(model.q.weight, dtype=ttnn.bfloat16)},
                "key_value": {"weight": preprocess_linear_weight(preprocessed_kv_weight, dtype=ttnn.bfloat16)},
                "o": {"weight": preprocess_linear_weight(model.o.weight, dtype=ttnn.bfloat16)},
            }
        else:
            # Self Attention
            preprocessed_qkv_weight = torch.cat([model.q.weight, model.k.weight, model.v.weight], dim=0)
            parameters = {
                "query_key_value": {"weight": preprocess_linear_weight(preprocessed_qkv_weight, dtype=ttnn.bfloat16)},
                "o": {"weight": preprocess_linear_weight(model.o.weight, dtype=ttnn.bfloat16)},
            }

    return parameters
