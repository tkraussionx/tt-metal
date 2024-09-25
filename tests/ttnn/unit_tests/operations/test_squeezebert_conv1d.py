# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import transformers
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters


def preprocess_conv1d_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT):
    weight = ttnn.from_torch(weight, dtype=dtype)
    return weight


def preprocess_conv1d_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT):
    bias = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    bias = ttnn.from_torch(bias, dtype=dtype)
    return bias


def custom_preprocessor(torch_model, name):
    parameters = {}
    for attr in ["query", "key", "value"]:
        if hasattr(torch_model, attr):
            parameters[attr] = {
                "weight": preprocess_conv1d_weight(getattr(torch_model, attr).weight, dtype=ttnn.float32),
                "bias": preprocess_conv1d_bias(getattr(torch_model, attr).bias, dtype=ttnn.float32),
            }

    if hasattr(torch_model, "conv1d"):
        parameters["conv1d"] = {"weight": preprocess_conv1d_weight(torch_model.conv1d.weight, dtype=ttnn.float32)}

    return parameters


def ttnn_conv1d(
    device,
    tt_input_tensor,
    parameters,
    conv_params,
    bias,
    *,
    output_dtype=ttnn.bfloat16,
    weights_dtype=ttnn.bfloat8_b,
    math_fidelity=ttnn.MathFidelity.LoFi,
    deallocate_activation=True,
    act_block_h=None,
    height_sharding=True,
    use_shallow_conv_variant=False,
    fp32_accum=False,
    packer_l1_acc=False,
    debug=False,
    groups=4,
):
    reader_patterns_cache = {}
    conv_config = ttnn.Conv1dConfig(
        dtype=output_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        shard_layout=(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        ),
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        deallocate_activation=deallocate_activation,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
    )

    [tt_output_tensor_on_device, out_length, weights_device, bias_device] = ttnn.Conv1d(
        input_tensor=tt_input_tensor,
        weight_tensor=parameters,
        in_channels=tt_input_tensor.shape[-1],
        out_channels=parameters.shape[0],
        device=device,
        bias_tensor=bias,
        kernel_size=1,
        stride=1,
        padding=0,
        batch_size=tt_input_tensor.shape[0],
        input_length=tt_input_tensor.shape[1],
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
    )
    reader_patterns_cache.clear()
    return tt_output_tensor_on_device, out_length


@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_squeezebert_attention(device, model_name, batch_size, sequence_size, torch_dtype, reset_seeds):
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    model = transformers.models.squeezebert.modeling_squeezebert.SqueezeBertSelfAttention(
        config, cin=config.hidden_size, q_groups=config.q_groups, k_groups=config.k_groups, v_groups=config.v_groups
    ).eval()
    model = model.to(torch_dtype)

    parameters = preprocess_model_parameters(
        model_name=f"ttnn_{model_name}_optimized",
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    hidden_states = torch.randn([batch_size, sequence_size, config.hidden_size], dtype=torch_dtype)
    torch_hidden_states = hidden_states.permute(0, 2, 1)
    torch_weight_tensor = ttnn.to_torch(parameters.query.weight)
    torch_bias_tensor = ttnn.to_torch(parameters.query.bias).squeeze(0).squeeze(0).squeeze(0)
    torch_out_golden_tensor = torch.nn.functional.conv1d(
        torch_hidden_states.float(),
        torch_weight_tensor.float(),
        bias=torch_bias_tensor.reshape(-1).float(),
        stride=1,
        padding=0,
        groups=4,
    )

    hidden_states = ttnn.from_torch(hidden_states, ttnn.bfloat16)

    query_layer, query_length = ttnn_conv1d(
        device,
        ttnn.from_device(hidden_states),
        parameters.query.weight,
        conv_params=[1, 0],
        bias=parameters.query.bias,
        output_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        deallocate_activation=True,
        act_block_h=None,
        height_sharding=True,
        use_shallow_conv_variant=False,
        fp32_accum=False,
        packer_l1_acc=False,
        debug=False,
        groups=4,
    )

    tt_output_tensor = ttnn.from_device(query_layer)
    torch_output_tensor = torch.Tensor(ttnn.to_torch(tt_output_tensor))
    torch_output_tensor = torch_output_tensor.reshape(tt_output_tensor.shape[0], query_length, hidden_states.shape[-1])
    output_tensor = torch.permute(torch_output_tensor, (0, 2, 1))

    assert_with_pcc(torch_out_golden_tensor, output_tensor, 0.99)
