# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers.models import bloom
from models.experimental.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.experimental.functional_bloom.tt import ttnn_functional_bloom
from models.experimental.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import skip_for_wormhole_b0
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import comp_allclose
from loguru import logger


def torch_random(shape, low, high, dtype):
    if dtype in {torch.bool, torch.int64}:
        return torch.randint(low, high, shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype).uniform_(low, high)


intermediate_outputs = {}


# Define a forward hook function to capture intermediate outputs
def capture_intermediate_output(name, module, inputs, outputs):
    intermediate_outputs[name] = outputs


@skip_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_block(device, model_name, batch_size, sequence_size, reset_seeds):
    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomBlock(config).eval()

    path = "tests/ttnn/integration_tests/bloom/inputs_bb_analysis/"
    real_input_test = True
    block_index = 5  # can test any block starting from index 0-23

    if real_input_test:
        torch_hidden_states = torch.load(path + f"torch_hidden_states{block_index}.pt")
        torch_causal_mask = torch.load(path + f"torch_causal_mask{block_index}.pt")
        torch_alibi = torch.load(path + f"torch_alibi{block_index}.pt")

        hidden_states = torch.load(path + f"hidden_states_{block_index}.pt").to(torch.bfloat16)
        hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
        alibi = torch.load(path + f"alibi_{block_index}.pt").to(torch.bfloat16)
        alibi = ttnn.from_torch(alibi, layout=ttnn.TILE_LAYOUT, device=device)
        causal_mask = torch.load(path + f"attention_mask_{block_index}.pt").to(torch.bfloat16)
        causal_mask = ttnn.from_torch(causal_mask, layout=ttnn.TILE_LAYOUT, device=device)

    else:
        torch_hidden_states = torch_random(
            (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32
        )
        torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
        torch_alibi = bloom.modeling_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.float32)

        torch_causal_mask = torch.empty((sequence_size, sequence_size), dtype=torch.bool)
        torch_seq_ids = torch.arange(sequence_size)
        torch_causal_mask[:, 0:] = torch_seq_ids[:, None] < torch_seq_ids[None, :]
        torch_causal_mask = torch_causal_mask[None, None, :, :].expand(
            batch_size, config.n_head, sequence_size, sequence_size
        )

        hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
        causal_mask = ttnn.from_torch(torch_causal_mask.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

        alibi = ttnn_optimized_functional_bloom.build_alibi_tensor(
            torch_attention_mask, config.n_head, dtype=torch.bfloat16
        )
        alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for name, module in model.named_modules():
        module.register_forward_hook(
            lambda module, inputs, outputs, name=name: capture_intermediate_output(name, module, inputs, outputs)
        )

    torch_output, *_ = model(
        torch_hidden_states,
        torch_alibi,
        torch_causal_mask,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
    )

    output = ttnn_optimized_functional_bloom.bloom_block(
        config=config,
        hidden_states=hidden_states,
        alibi=alibi,
        attention_mask=causal_mask,
        parameters=parameters,
        base_address="",
        intermediate_outputs=intermediate_outputs,
    )
    output = ttnn.to_torch(output)

    golden_tensor = torch_output
    num_tokens = torch_output.shape[1]
    computed_tensor = output[:, :num_tokens, :]
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
    logger.info(
        f"Bloom block: PCC:{pcc}, \ntotal_elements:{len(gt)}, tolerance:{tolerance}, count(element > tolerance):{num_values_gt_tolerance},  \nallclose{allclose}, atol_delta: {atol_delta}, rtol_delta: {rtol_delta}, \ntorch.min: {g_min.item()}, torch_max:{g_max.item()}, ttnn_min:{c_min.item()}, ttnn_max:{c_max.item()}"
    )
