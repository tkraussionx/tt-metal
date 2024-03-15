# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from pathlib import Path

import ttnn
from models.demos.mixtral8x7b.tt.mixtral_attention_ttnn import TtMixtralAttention
from models.demos.mixtral8x7b.tt.mixtral_common_ttnn import (
    precompute_freqs,
    generate_cos_sin_cache_ttnn,
    prepare_inputs_ttnn,
)
from models.demos.mixtral8x7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mixtral8x7b.reference.model import Attention
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import get_devices_for_t3000


@pytest.mark.parametrize(
    "iterations",
    ((10),),
)
def test_mixtral_attention_inference(all_devices, iterations, reset_seeds):
    pcc = 0.99
    dtype = ttnn.bfloat8_b
    devices = all_devices
    num_devices = len(devices)
    assert num_devices == 8, f"This test requires a T3000 (8 devices), found {num_devices} devices."
    devices = get_devices_for_t3000(devices, num_devices)  # [ttnn.open_device(device_id=i) for i in range(8)]

    model_args = TtModelArgs()
    state_dict = torch.load(model_args.consolidated_weights_path(0), map_location="cpu")

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    batch = 32
    seq_len = 1  # length to generate

    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache_ttnn(
        devices, model_args.head_dim, model_args.max_seq_len * 2, 10000, dtype
    )
    tt_model = TtMixtralAttention(
        devices,
        state_dict,
        args=model_args,
        layer_num=0,
        dtype=dtype,
        tt_cos_cached=tt_cos_cached,
        tt_sin_cached=tt_sin_cached,
    )
    generation_start_pos = 0
    generation_length = iterations
    all_tests_pass = True

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_attention_input = pt_attention_input.clone()
        start_pos = generation_start_pos + i
        attention_input, start_pos, attn_mask, current_pos, rot_mat = prepare_inputs_ttnn(
            tt_attention_input,
            start_pos,
            tt_model.hidden_size,
            tt_model.head_dim,
            tt_model.sliding_window,
            tt_model.max_seq_len,
            tt_model.devices,
        )

        tt_out = tt_model(
            attention_input,
            start_pos,
            current_pos,
            attn_mask,
            rot_mat,
        )
        assert isinstance(tt_out, list)  # tt_out should be replicated on N devices
        tt_out = tt_out[0]
        tt_output_torch = ttnn.to_torch(tt_out).squeeze(2)  # [ batch, seq, hidden_dim]

        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])
        # mask = torch.randn(1, 1)

        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=None)
        print("OUTPUT SHAPES, ", reference_output.shape, tt_output_torch.shape)
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info(f"[start_pos={start_pos}] Mistral_Attention Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] Mistral_Attention Failed!")
            all_tests_pass = False

        # Check kv cache
        # PyTorch output --------------------------------------------------------------------
        pytorch_layer_present = [
            reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
            reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
        ]
        # TT hardware execution -------------------------------------------------------------
        tt_layer_present = []
        for layer_past in tt_model.layer_past_list:
            tt_layer_present.append([ttnn.to_torch(cache) for cache in layer_past])
        # concat the pasts by heads
        if len(devices) > 1:
            tt_layer_present = [
                torch.cat([tt_cache for tt_cache in tt_cache_head], dim=1) for tt_cache_head in zip(*tt_layer_present)
            ]
        else:
            tt_layer_present = tt_layer_present[0]

        for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
            # print("CACHE PT", cache_pt, cache_pt.shape) #[32, 8, 4096, 128]
            # print("CACHE TT", cache_tt, cache_tt.shape) #[32, 8, 4096, 128]

            if i == 0:
                logger.info(
                    f"Skipping K cache comparison, since tt_lib rot_embed op does a different permutation from reference PyTorch code"
                )
                continue

            cache_length_to_check = min(model_args.sliding_window, generation_start_pos + generation_length + 1)
            cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
            cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
            logger.info(f"V cache output: {output_pcc}")

            if does_pass:
                logger.info(f"V Cache Passed!")
            else:
                logger.warning(f"V Cache Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    if all_tests_pass:
        logger.info("Mistral Attention output Passed!")
    else:
        logger.warning("Mistral Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
