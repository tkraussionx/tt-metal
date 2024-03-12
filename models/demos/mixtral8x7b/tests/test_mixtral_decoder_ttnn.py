# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from pathlib import Path
import ttnn
from models.demos.mixtral8x7b.tt.mixtral_common_ttnn import (
    precompute_freqs,
    generate_cos_sin_cache_ttnn,
    prepare_inputs_ttnn,
)
from models.demos.mixtral8x7b.tt.mixtral_decoder_ttnn import TtTransformerBlock
from models.demos.mixtral8x7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mixtral8x7b.reference.model import TransformerBlock
from models.utility_functions import comp_pcc, comp_allclose, get_devices_for_t3000


@pytest.mark.parametrize(
    "iterations",
    ((1),),
)
def test_mixtral_decoder_inference(all_devices, iterations):
    pcc = 0.99
    dtype = ttnn.bfloat8_b
    devices = all_devices
    num_devices = len(devices)
    assert num_devices == 8, "This test requires a T3000 (8 devices)"
    devices = get_devices_for_t3000(devices, num_devices)  # [ttnn.open_device(device_id=i) for i in range(8)]

    model_args = TtModelArgs()
    state_dict = torch.load(model_args.consolidated_weights_path(0), map_location="cpu")

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}
    base_address = "feed_forward."
    partial_state_dict[base_address + "gate.weight"] = partial_state_dict["block_sparse_moe.gate.weight"]
    del partial_state_dict["block_sparse_moe.gate.weight"]

    w1 = partial_state_dict["block_sparse_moe.w1"].view(8, 14336, 4096)
    w2 = partial_state_dict["block_sparse_moe.w2"].view(8, 4096, 14336)
    w3 = partial_state_dict["block_sparse_moe.w3"].view(8, 14336, 4096)
    for i in range(8):
        partial_state_dict[base_address + f"experts.{i}.w1.weight"] = w1[i]
        partial_state_dict[base_address + f"experts.{i}.w2.weight"] = w2[i]
        partial_state_dict[base_address + f"experts.{i}.w3.weight"] = w3[i]
    partial_state_dict.pop("block_sparse_moe.w1")
    partial_state_dict.pop("block_sparse_moe.w2")
    partial_state_dict.pop("block_sparse_moe.w3")

    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache_ttnn(
        devices, model_args.head_dim, model_args.max_seq_len * 2, 10000, dtype
    )
    # Initialize TT model
    tt_model = TtTransformerBlock(
        devices=devices,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtype=dtype,
        tt_cos_cached=tt_cos_cached,
        tt_sin_cached=tt_sin_cached,
    )

    generation_start_pos = 0
    generation_length = iterations
    all_tests_pass = True

    seqlen = 1
    batch = 32

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    # TODO Update start_pos (check llama test for reference)
    for i in range(generation_length):
        print(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        start_pos = generation_start_pos + i

        decode_input, start_pos, attn_mask, current_pos = prepare_inputs_ttnn(
            tt_decode_input,
            start_pos,
            tt_model.hidden_size,
            tt_model.n_local_heads,
            tt_model.sliding_window,
            tt_model.devices,
            tt_model.num_devices,
        )
        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask)
        print("DONE TT OUT")
        tt_output_torch = ttnn.to_torch(tt_out[0]).squeeze(2)  # [batch, seq, hidden_dim]

        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])

        # Reference model
        # mask = tt2torch_tensor(attn_mask[0])
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions, mask=None)  # mask)
        print("REF MODEL DONE", ref_output.shape)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Decoder Block Passed!")
        else:
            logger.warning("Mistral Decoder Block Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
