# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import transformers
import torch
from transformers.models.bloom.configuration_bloom import BloomConfig

import ttnn
from ttnn.model_preprocessing import (
    ParameterDict,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
import math
from typing import Tuple

import transformers
import torch
from torch.nn import functional as F
from transformers.models.bloom.configuration_bloom import BloomConfig
from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import comp_allclose
from loguru import logger
import csv
import matplotlib.pyplot as plt
import os
import numpy as np

BLOOM_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
BLOOM_DTYPE = ttnn.bfloat8_b
ASSUME_FUSED_SOFTMAX = False
num_tokens = 0
iter = 0
# Global CSV file path
csv_file_path = "tests/ttnn/integration_tests/bloom/bloom_block_analysis.csv"
torch_attn_dumps_path = "tests/ttnn/integration_tests/bloom/opwise_intermediate_dumps/"
tensor_csv_path = "tests/ttnn/integration_tests/bloom/tensor_csv/"


def write_header():
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        column_names = [
            "Layer/Sub-module",
            "PCC",
            "tolerance = atol_delta*0.02",
            "elements_count",
            "count(abs(diff) > tolerance)",
            "mismatch %",
            "allclose",
            "atol_delta",
            "rtol_delta",
            "torch_min",
            "torch_max",
            "ttnn_min",
            "ttnn_max",
        ]
        writer.writerow(column_names)  # Write only column names


def append_to_csv(data):
    with open(csv_file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


def get_2dsliced_tensor(computed_tensor):
    comp_num_dims = computed_tensor.ndim
    # Extract the last two dimensions and squeeze
    if comp_num_dims >= 2:
        last_two_dims = computed_tensor[(*[0] * (comp_num_dims - 2), slice(None), slice(None))]
        computed_tensor = last_two_dims.squeeze()

    if computed_tensor.shape[0] == 384:  # Remove the padded outputs
        computed_tensor = computed_tensor[:num_tokens, :]
    if computed_tensor.shape[1] == 384:
        computed_tensor = computed_tensor[:, :num_tokens]
    return computed_tensor


def dump_csv(key, golden_tensor, computed_tensor):
    if not os.path.exists(tensor_csv_path):
        os.makedirs(tensor_csv_path)
        if not os.path.exists(tensor_csv_path + "torch/"):
            os.makedirs(tensor_csv_path + "torch/")
        if not os.path.exists(tensor_csv_path + "tt/"):
            os.makedirs(tensor_csv_path + "tt/")

    golden_tensor = get_2dsliced_tensor(golden_tensor)
    computed_tensor = get_2dsliced_tensor(computed_tensor)

    golden_tensor = golden_tensor.detach().numpy()
    computed_tensor = computed_tensor.float().detach().numpy()

    np.savetxt(f"{tensor_csv_path}torch/{key}.csv", golden_tensor, delimiter=",")
    np.savetxt(f"{tensor_csv_path}tt/{key}.csv", computed_tensor, delimiter=",")


def make_histogram(key, sliced_g, sliced_c):
    path = "tests/ttnn/integration_tests/bloom/plots"
    plt.figure(figsize=(13, 7))
    num_bins = 100

    # Define the colors for each histogram
    color_x1 = "blue"
    color_x2 = "orange"

    plt.hist(
        sliced_g, bins=num_bins, color=color_x1, alpha=0.5, label="torch_tensor"
    )  # alpha controls the transparency
    plt.hist(sliced_c, bins=num_bins, color=color_x2, alpha=0.5, label="tt_tensor")

    plt.xlim([min(min(sliced_g), min(sliced_c)), max(max(sliced_g), max(sliced_c))])
    title_font = {"weight": "bold", "size": 16}
    plt.title(f"{key}'s torch vs tt distribution", fontdict=title_font)
    plt.legend()
    legend_font = {"weight": "bold", "size": 10}
    plt.xlabel("Value", fontdict=legend_font)
    plt.ylabel("Frequency", fontdict=legend_font)
    plt.legend()
    plt.grid(True)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + f"/{key}_histogram.png")


def plot_figure(key, golden_tensor, computed_tensor, is_sliced=False):
    global num_tokens
    if not is_sliced:
        computed_tensor = computed_tensor[:1, :num_tokens, :]
        sliced_g = golden_tensor[:1, :, :].flatten().detach().numpy()
        sliced_c = (torch.flatten(computed_tensor[:1, :, :].float())).detach().numpy()
    else:
        sliced_g = golden_tensor.flatten().detach().numpy()
        sliced_c = torch.flatten(computed_tensor.float()).detach().numpy()

    golden_tensor = get_2dsliced_tensor(golden_tensor)
    computed_tensor = get_2dsliced_tensor(computed_tensor)

    plt.figure(figsize=(13, 7))
    plt.scatter(range(len(sliced_g)), sliced_g, label="torch_tensor", marker="o", color="blue", s=1.333)
    plt.scatter(range(len(sliced_c)), sliced_c, label="tt_tensor", marker="o", color="orange", s=1.333)
    plt.scatter(
        range(len(sliced_c)),
        abs(sliced_c - sliced_g),
        label="Absolute(torch_tensor - tt_tensor)",
        marker="o",
        color="red",
        s=1.333,
    )
    legend_font = {"weight": "bold", "size": 10}
    plt.xlabel("Index", fontdict=legend_font)
    plt.ylabel("Value", fontdict=legend_font)
    title_font = {"weight": "bold", "size": 16}  # Adjust the size as needed
    plt.title(f"{key}'s torch vs tt ", fontdict=title_font)
    plt.legend()
    plt.grid(True)
    path = "tests/ttnn/integration_tests/bloom/plots"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + f"/{key}_scatter.png")
    make_histogram(key, sliced_g, sliced_c)


def write_data_to_csv(key, golden_tensor, computed_tensor, is_sliced=False):
    global num_tokens
    if not is_sliced:
        computed_tensor = computed_tensor[:, :num_tokens, :]

    golden_tensor = get_2dsliced_tensor(golden_tensor)
    computed_tensor = get_2dsliced_tensor(computed_tensor)

    g_min = torch.min(golden_tensor)
    g_max = torch.max(golden_tensor)
    c_min = torch.min(computed_tensor)
    c_max = torch.max(computed_tensor)
    gt = torch.flatten(golden_tensor)
    ct = torch.flatten(computed_tensor.float())
    dt = torch.abs(gt - ct)
    pcc = check_with_pcc(golden_tensor, computed_tensor)[1]
    allclose, atol_delta, rtol_delta = comp_allclose(golden_tensor, computed_tensor)
    tolerance = atol_delta * 0.02  # setting tolerance to 2% of max difference
    num_values_gt_tolerance = (dt > tolerance).sum().item()
    logger.info(f"{key}:  allclose: {allclose}, atol_delta: {atol_delta}, rtol_delta: {rtol_delta}")
    data = [
        f"{key}",
        pcc,
        tolerance,
        len(gt),
        num_values_gt_tolerance,
        (num_values_gt_tolerance / len(gt)) * 100,
        allclose,
        atol_delta,
        rtol_delta,
        g_min.item(),
        g_max.item(),
        c_min.item(),
        c_max.item(),
    ]
    append_to_csv(data)
    dump_csv(key, golden_tensor, computed_tensor)


# From transformers/models/bloom/modeling_bloom.py
def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size, num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size, num_heads, 1, seq_length).to(dtype)


def split_query_key_value_and_split_heads(
    query_key_value: torch.Tensor, num_heads: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value, memory_config=BLOOM_MEMORY_CONFIG, num_heads=num_heads
    )
    return output


def create_query_key_value(
    config: BloomConfig, hidden_states, *, parameters: ParameterDict, base_address="", intermediate_outputs=None
):
    query_key_value = ttnn.linear(
        hidden_states,
        input_tensor_b=parameters.query_key_value.weight,
        bias=parameters.query_key_value.bias,
        core_grid=ttnn.CoreGrid(y=9, x=12),
        memory_config=BLOOM_MEMORY_CONFIG,
        dtype=BLOOM_DTYPE,
    )
    ttnn.deallocate(hidden_states)
    if intermediate_outputs is not None:
        key = base_address + "query_key_value"
        golden_tensor = intermediate_outputs[key]
        computed_tensor = ttnn.to_torch(query_key_value)
        write_data_to_csv(key, golden_tensor, computed_tensor)
        plot_figure(key, golden_tensor, computed_tensor)

    query, key, value = split_query_key_value_and_split_heads(query_key_value, num_heads=config.n_head)
    ttnn.deallocate(query_key_value)

    return query, key, value


def compute_attention_scores(query_layer, key_layer, alibi):
    *_, head_size = query_layer.shape
    attention_scores = ttnn.matmul(
        query_layer,
        key_layer,
        core_grid=ttnn.CoreGrid(y=9, x=12),
        memory_config=BLOOM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(query_layer)
    ttnn.deallocate(key_layer)

    if ASSUME_FUSED_SOFTMAX:
        return attention_scores

    inv_norm_factor = 1.0 / math.sqrt(head_size)
    scaled_attention_scores = ttnn.mul(attention_scores, inv_norm_factor, memory_config=BLOOM_MEMORY_CONFIG)
    ttnn.deallocate(attention_scores)

    scaled_attention_scores_plus_alibi = ttnn.add(scaled_attention_scores, alibi, memory_config=BLOOM_MEMORY_CONFIG)
    ttnn.deallocate(scaled_attention_scores)

    return scaled_attention_scores_plus_alibi


def compute_attention_probs(attention_scores, causal_mask, base_address="", intermediate_outputs=None):
    if ASSUME_FUSED_SOFTMAX:
        attention_weights = attention_scores
    else:
        attention_weights = ttnn.add(attention_scores, causal_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attention_scores)

    if intermediate_outputs is not None:
        torch_attention_weights = torch.load(torch_attn_dumps_path + f"attention_weights.pt")
        torch_attention_weights = torch_attention_weights[:1, 1, :, :].squeeze().squeeze()
        tt_attention_weights = ttnn.to_torch(attention_weights)
        tt_attention_weights = tt_attention_weights[:1, 1, :num_tokens, :num_tokens].squeeze().squeeze()
        write_data_to_csv(
            base_address + "attention_weights",
            golden_tensor=torch_attention_weights,
            computed_tensor=tt_attention_weights,
            is_sliced=True,
        )
        plot_figure(
            key=base_address + "attention_weights",
            golden_tensor=torch_attention_weights,
            computed_tensor=tt_attention_weights,
            is_sliced=True,
        )

    attention_probs = ttnn.softmax(attention_weights, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if not ASSUME_FUSED_SOFTMAX:
        ttnn.deallocate(attention_weights)

    return attention_probs


# Based on transformers/models/bloom/modeling_bloom.py
def merge_heads(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.transformer.concatenate_heads(x, memory_config=BLOOM_MEMORY_CONFIG)


def compute_context_layer(attention_probs, value_layer, base_address="", intermediate_outputs=None):
    context_layer = ttnn.matmul(
        attention_probs,
        value_layer,
        core_grid=ttnn.CoreGrid(y=9, x=12),
        memory_config=BLOOM_MEMORY_CONFIG,
        dtype=BLOOM_DTYPE,
    )

    if intermediate_outputs is not None:
        torch_context_layer = torch.load(torch_attn_dumps_path + f"attention_context_matmul.pt")
        torch_context_layer = torch_context_layer[:1, :, :].squeeze()
        tt_context_layer = ttnn.to_torch(context_layer)
        tt_context_layer = tt_context_layer[:1, 1, :num_tokens, :].squeeze().squeeze()
        write_data_to_csv(
            base_address + "context_matmul",
            golden_tensor=torch_context_layer,
            computed_tensor=tt_context_layer,
            is_sliced=True,
        )
        plot_figure(
            key=base_address + "context_matmul",
            golden_tensor=torch_context_layer,
            computed_tensor=tt_context_layer,
            is_sliced=True,
        )

    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value_layer)
    return merge_heads(context_layer)


def finalize_output(context_layer, *, parameters: ParameterDict):
    output_tensor = ttnn.linear(
        context_layer,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        core_grid=ttnn.CoreGrid(y=9, x=12),
        memory_config=BLOOM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(context_layer)
    return output_tensor


def bloom_attention(
    config: BloomConfig,
    hidden_states: ttnn.Tensor,
    residual: ttnn.Tensor,
    alibi: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    *,
    parameters: ParameterDict,
    base_address="",
    intermediate_outputs=None,
):
    query_layer, key_layer, value_layer = create_query_key_value(
        config,
        hidden_states,
        parameters=parameters,
        base_address=base_address,
        intermediate_outputs=intermediate_outputs,
    )

    attention_scores = compute_attention_scores(query_layer, key_layer, alibi)
    if intermediate_outputs is not None:
        torch_attention_scores = torch.load(torch_attn_dumps_path + f"attention_scores.pt")
        torch_attention_scores = torch_attention_scores[:1, 1, :, :].squeeze().squeeze()
        tt_attention_scores = ttnn.to_torch(attention_scores)
        tt_attention_scores = tt_attention_scores[:1, 1, :num_tokens, :num_tokens].squeeze().squeeze()
        write_data_to_csv(
            base_address + "attention_scores",
            golden_tensor=torch_attention_scores,
            computed_tensor=tt_attention_scores,
            is_sliced=True,
        )
        plot_figure(
            key=base_address + "attention_scores",
            golden_tensor=torch_attention_scores,
            computed_tensor=tt_attention_scores,
            is_sliced=True,
        )

    attention_probs = compute_attention_probs(
        attention_scores, attention_mask, base_address=base_address, intermediate_outputs=intermediate_outputs
    )
    if intermediate_outputs is not None:
        torch_attention_probs = torch.load(torch_attn_dumps_path + f"attention_probs.pt")
        torch_attention_probs = torch_attention_probs[:1, 1, :, :].squeeze().squeeze()
        tt_attention_probs = ttnn.to_torch(attention_probs)
        tt_attention_probs = tt_attention_probs[:1, 1, :num_tokens, :num_tokens].squeeze().squeeze()
        write_data_to_csv(
            base_address + "attention_probs",
            golden_tensor=torch_attention_scores,
            computed_tensor=tt_attention_scores,
            is_sliced=True,
        )
        plot_figure(
            key=base_address + "attention_probs",
            golden_tensor=torch_attention_probs,
            computed_tensor=tt_attention_probs,
            is_sliced=True,
        )

    context_layer = compute_context_layer(
        attention_probs, value_layer, base_address=base_address, intermediate_outputs=intermediate_outputs
    )
    if intermediate_outputs is not None:
        torch_context_layer = torch.load(torch_attn_dumps_path + f"attention_context_layer.pt")
        torch_context_layer = torch_context_layer[:1, :num_tokens, :]
        tt_context_layer = ttnn.to_torch(context_layer)
        tt_context_layer = tt_context_layer[:1, :num_tokens, :]
        write_data_to_csv(
            base_address + ".context_layer",
            golden_tensor=torch_attention_scores,
            computed_tensor=tt_attention_scores,
            is_sliced=True,
        )
        plot_figure(
            key=base_address + ".context_layer",
            golden_tensor=torch_context_layer,
            computed_tensor=tt_context_layer,
            is_sliced=True,
        )

    output_tensor = finalize_output(context_layer, parameters=parameters)
    if intermediate_outputs is not None:
        key = base_address + "dense"
        golden_tensor = intermediate_outputs[key]
        computed_tensor = ttnn.to_torch(output_tensor)
        write_data_to_csv(key, golden_tensor, computed_tensor)
        plot_figure(key, golden_tensor, computed_tensor)

    attention_output = ttnn.add(output_tensor, residual, memory_config=BLOOM_MEMORY_CONFIG)
    return attention_output


def bloom_mlp(
    hidden_states,
    residual: torch.Tensor,
    *,
    parameters: ParameterDict,
    base_address="",
    intermediate_outputs=None,
):
    ff1_output = ttnn.linear(
        hidden_states,
        parameters.dense_h_to_4h.weight,
        bias=parameters.dense_h_to_4h.bias,
        core_grid=ttnn.CoreGrid(y=9, x=12),
        activation="gelu",
        memory_config=BLOOM_MEMORY_CONFIG,
        dtype=BLOOM_DTYPE,
    )
    ttnn.deallocate(hidden_states)

    if intermediate_outputs is not None:
        key = base_address + "gelu_impl"
        golden_tensor = intermediate_outputs[key]
        computed_tensor = ttnn.to_torch(ff1_output)
        write_data_to_csv(key, golden_tensor, computed_tensor)
        plot_figure(key, golden_tensor, computed_tensor)

    ff2_output = ttnn.linear(
        ff1_output,
        parameters.dense_4h_to_h.weight,
        bias=parameters.dense_4h_to_h.bias,
        core_grid=ttnn.CoreGrid(y=9, x=12),
        memory_config=BLOOM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    if intermediate_outputs is not None:
        key = base_address + "dense_4h_to_h"
        golden_tensor = intermediate_outputs[key]
        computed_tensor = ttnn.to_torch(ff2_output)
        write_data_to_csv(key, golden_tensor, computed_tensor)
        plot_figure(key, golden_tensor, computed_tensor)

    ttnn.deallocate(ff1_output)
    mlp_output = ttnn.add(ff2_output, residual, memory_config=BLOOM_MEMORY_CONFIG)

    return mlp_output


def bloom_block(
    config: BloomConfig,
    hidden_states: ttnn.Tensor,
    alibi: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    *,
    parameters: ParameterDict,
    base_address="",
    intermediate_outputs=None,
) -> ttnn.Tensor:
    global iter
    path = "tests/ttnn/integration_tests/bloom/inputs_bb_analysis/"
    torch.save(ttnn.to_torch(hidden_states), path + f"hidden_states_{iter}.pt")
    torch.save(ttnn.to_torch(alibi), path + f"alibi_{iter}.pt")
    torch.save(ttnn.to_torch(attention_mask), path + f"attention_mask_{iter}.pt")

    normalized_hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.input_layernorm.weight,
        bias=parameters.input_layernorm.bias,
        memory_config=BLOOM_MEMORY_CONFIG,
    )

    if intermediate_outputs is not None:
        write_header()
        global num_tokens
        num_tokens = intermediate_outputs["input_layernorm"].shape[1]
        key = base_address + "input_layernorm"
        golden_tensor = intermediate_outputs[key]
        computed_tensor = ttnn.to_torch(normalized_hidden_states)
        write_data_to_csv(key, golden_tensor, computed_tensor)
        plot_figure(key, golden_tensor, computed_tensor)

    attention_output = bloom_attention(
        config,
        normalized_hidden_states,
        hidden_states,
        alibi,
        attention_mask,
        parameters=parameters.self_attention,
        base_address=base_address + "self_attention.",
        intermediate_outputs=intermediate_outputs,
    )
    ttnn.deallocate(hidden_states)

    if intermediate_outputs is not None:
        key = base_address + "self_attention"
        golden_tensor = intermediate_outputs[key][0]
        computed_tensor = ttnn.to_torch(attention_output)
        write_data_to_csv(key, golden_tensor, computed_tensor)
        plot_figure(key, golden_tensor, computed_tensor)

    normalized_attention_output = ttnn.layer_norm(
        attention_output,
        weight=parameters.post_attention_layernorm.weight,
        bias=parameters.post_attention_layernorm.bias,
        memory_config=BLOOM_MEMORY_CONFIG,
    )

    if intermediate_outputs is not None:
        key = base_address + "post_attention_layernorm"
        golden_tensor = intermediate_outputs[key]
        computed_tensor = ttnn.to_torch(normalized_attention_output)
        write_data_to_csv(key, golden_tensor, computed_tensor)
        plot_figure(key, golden_tensor, computed_tensor)

    mlp_output = bloom_mlp(
        normalized_attention_output,
        residual=attention_output,
        parameters=parameters.mlp,
        base_address=base_address + "mlp.",
        intermediate_outputs=intermediate_outputs,
    )
    ttnn.deallocate(attention_output)

    if intermediate_outputs is not None:
        key = base_address + "mlp"
        golden_tensor = intermediate_outputs[key]
        computed_tensor = ttnn.to_torch(mlp_output)
        write_data_to_csv(key, golden_tensor, computed_tensor)
        plot_figure(key, golden_tensor, computed_tensor)

    hidden_states = mlp_output
    hidden_states = ttnn.reallocate(hidden_states)
    return hidden_states


def bloom(
    config,
    input_ids,
    alibi,
    causal_mask,
    *,
    parameters,
):
    inputs_embeds = ttnn.embedding(
        input_ids,
        parameters.word_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )

    hidden_states = ttnn.layer_norm(
        inputs_embeds,
        weight=parameters.word_embeddings_layernorm.weight,
        bias=parameters.word_embeddings_layernorm.bias,
        memory_config=BLOOM_MEMORY_CONFIG,
    )
    ttnn.deallocate(inputs_embeds)
    global iter
    for layer_parameters in parameters.h:
        hidden_states = bloom_block(
            config,
            hidden_states,
            alibi,
            causal_mask,
            parameters=layer_parameters,
        )
        iter += 1

    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.ln_f.weight,
        bias=parameters.ln_f.bias,
    )
    return hidden_states


def bloom_for_causal_lm(config, input_ids, alibi, causal_mask, *, parameters):
    hidden_states = bloom(config, input_ids, alibi, causal_mask, parameters=parameters.transformer)

    # Unfortunately we do not have the ability to handle large tensors yet. So running final matmul ising torch as a workaround.
    hidden_states = ttnn.from_device(hidden_states)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.to_torch(hidden_states).to(torch.float32)
    output = hidden_states @ parameters.lm_head.weight

    return output


def bloom_for_question_answering(config, input_ids, alibi, casual_mask, *, parameters):
    hidden_states = bloom(config, input_ids, alibi, casual_mask, parameters=parameters.transformer)
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.qa_outputs.weight,
        bias=parameters.qa_outputs.bias,
        memory_config=BLOOM_MEMORY_CONFIG,
    )
    return hidden_states


def preprocess_inputs(
    *,
    input_ids,
    device,
    num_heads,
    max_length=384,
    attention_mask=None,
):
    num_input_tokens = input_ids.shape[-1]
    padding_needed = (max_length - (num_input_tokens % max_length)) % max_length
    padded_input_ids = F.pad(input_ids, (0, padding_needed, 0, 0))
    padded_input_ids = ttnn.from_torch(padded_input_ids, dtype=ttnn.uint32)
    padded_input_ids = ttnn.to_device(padded_input_ids, device)

    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    attention_mask = F.pad(attention_mask, (0, padding_needed, 0, 0))

    alibi = build_alibi_tensor(attention_mask, num_heads, dtype=torch.float)
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16)
    alibi = ttnn.to_layout(alibi, ttnn.TILE_LAYOUT)
    alibi = ttnn.to_device(alibi, device)

    batch_size, padded_seq_length = attention_mask.shape
    mask = torch.empty((padded_seq_length, padded_seq_length), dtype=torch.bool)
    seq_ids = torch.arange(padded_seq_length)
    mask[:, 0:] = seq_ids[:, None] < seq_ids[None, :]
    causal_mask = mask[None, None, :, :].expand(batch_size, num_heads, padded_seq_length, padded_seq_length)
    causal_mask = causal_mask.float()
    causal_mask *= -100

    causal_mask = ttnn.from_torch(causal_mask, dtype=ttnn.bfloat16)
    causal_mask = ttnn.to_layout(causal_mask, ttnn.TILE_LAYOUT)
    causal_mask = ttnn.to_device(causal_mask, device)

    return padded_input_ids, alibi, causal_mask


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.bloom.modeling_bloom.BloomAttention):
        weight = torch_model.query_key_value.weight
        bias = torch_model.query_key_value.bias

        assert weight.shape[-1] == 1024
        num_heads = 16

        three_times_hidden_size, _ = weight.shape
        hidden_size = three_times_hidden_size // 3
        head_size = hidden_size // num_heads

        # Store QKV one after another instead of interleaving heads
        weight = weight.view(num_heads, 3, head_size, hidden_size)
        query, key, value = weight[:, 0], weight[:, 1], weight[:, 2]
        query = torch.reshape(query, (hidden_size, hidden_size))
        key = torch.reshape(key, (hidden_size, hidden_size))
        value = torch.reshape(value, (hidden_size, hidden_size))
        preprocessed_weight = torch.cat([query, key, value], dim=0)

        # Store QKV one after another instead of interleaving heads
        bias = bias.view(num_heads, 3, head_size)
        query, key, value = bias[:, 0], bias[:, 1], bias[:, 2]
        query = torch.reshape(query, (hidden_size,))
        key = torch.reshape(key, (hidden_size,))
        value = torch.reshape(value, (hidden_size,))
        preprocessed_bias = torch.cat([query, key, value], dim=0)

        parameters = {"query_key_value": {}, "dense": {}}

        parameters["query_key_value"]["weight"] = preprocess_linear_weight(preprocessed_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(preprocessed_bias, dtype=ttnn.bfloat16)

        parameters["dense"]["weight"] = preprocess_linear_weight(torch_model.dense.weight, dtype=ttnn.bfloat16)
        parameters["dense"]["bias"] = preprocess_linear_bias(torch_model.dense.bias, dtype=ttnn.bfloat16)
    return parameters
