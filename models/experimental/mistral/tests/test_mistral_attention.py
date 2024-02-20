# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
import pytest
from loguru import logger
import json

from models.experimental.mistral.tt.mistral_attention import TtMistralAttention
from models.experimental.mistral.tt.mistral_common import precompute_freqs
from models.experimental.mistral.tt.model_config import TtModelArgs, get_model_config
from models.experimental.mistral.reference.model import Attention
from models.utility_functions import torch_to_tt_tensor_rm, tt2torch_tensor
from models.experimental.mistral.mistral_helper_funcs import unpad_from_zero, get_freqs_cis, format_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)

"""
@pytest.mark.parametrize(
    "rotary_embedding_ondevice",
    (True, False),
)
@pytest.mark.parametrize(
    "softmax_ondevice",
    (True, False),
)
@pytest.mark.parametrize(
    "empty_ondevice",
    (True, False),
)
@pytest.mark.parametrize(
    "scatter_ondevice",
    (True, False),
)
@pytest.mark.parametrize(
    "dtype",
    (tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B),
)
"""


@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT16-L1", "BFLOAT8-DRAM", "BFLOAT8-L1"),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.97),),
    # ((0.9793647197892646),),
)
def test_mistral_attention_inference(
    pcc,
    model_config,
    model_location_generator,
    device,
):
    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    state_dict = torch.load(mistral_path / "consolidated.00.pth")

    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    if True:
        state_dict = {k[19:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention."))}

    model_args.max_batch_size = 32
    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(state_dict)

    # TODO Scale the model (mixtral) to multiple devices when T3000 is available
    devices = [
        device,
    ]

    batch = 32
    seq_len = 1

    tt_model = TtMistralAttention(
        devices,
        state_dict,
        base_url=base_address,
        layer_num=0,
        model_config=get_model_config(model_config),
        configuration=model_args,
    )
    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    # TODO Update start_pos (check llama test for reference)
    for i in range(generation_length):
        pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
        tt_attention_input = pt_attention_input.clone()
        start_pos = generation_start_pos + i
        attention_input, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs(tt_attention_input, start_pos)

        tt_out = tt_model(
            attention_input,
            rot_mat,
            start_pos,
            attn_mask,
        )
        assert isinstance(tt_out, list)  # tt_out should be replicated on N devices
        tt_out = tt_out[0]
        tt_output_torch = tt2torch_tensor(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        # empty_tensor = torch.zeros((start_pos+1, 64))
        # cos, sin = precompute_freqs(model_args.head_dim, 1)
        # freqs_cis = torch.complex(cos, sin)
        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])
        # mask = torch.randn(1, 1)

        reference_output = reference_model(pt_attention_input, freqs_cis_i, positions, mask=None)

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
            tt_layer_present.append([tt2torch_tensor(cache) for cache in layer_past])
        # concat the pasts by heads
        if len(devices) > 1:
            tt_layer_present = [
                torch.cat([tt_cache for tt_cache in tt_cache_head], dim=1) for tt_cache_head in zip(*tt_layer_present)
            ]
        else:
            tt_layer_present = tt_layer_present[0]

        for cache_pt, cache_tt in zip(pytorch_layer_present, tt_layer_present):
            # print("CACHE PT", cache_pt, cache_pt.shape) #[32, 8, 4096, 128]
            # print("CACHE TT", cache_tt, cache_tt.shape) #[32, 8, 4096, 128]

            cache_length_to_check = min(model_args.sliding_window, generation_start_pos + generation_length + 1)
            cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
            cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
            logger.info(f"Output: {output_pcc}")

            if does_pass:
                logger.info(f"KV Cache Passed!")
            else:
                logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    if all_tests_pass:
        logger.info("Mistral Attention output Passed!")
    else:
        logger.warning("Mistral Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
