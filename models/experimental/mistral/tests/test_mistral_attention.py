# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
import pytest
from loguru import logger
import json

from models.experimental.mistral.tt.mistral_attention import TtMistralAttention
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
    print("KEYS!", state_dict.keys())
    model_args.max_batch_size = 32
    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(state_dict)

    devices = [
        device,
        device,
        device,
        device,
        device,
        device,
        device,
        device,
    ]

    batch = 32
    seq_len = 1

    pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
    tt_attention_input = pt_attention_input.clone()

    tt_model = TtMistralAttention(
        devices,
        state_dict,
        base_url=base_address,
        layer_num=0,
        model_config=get_model_config(model_config),
        configuration=model_args,
    )

    # TODO Update start_pos (check llama test for reference)
    attention_input, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs(tt_attention_input, start_pos=0)

    tt_out = tt_model(
        attention_input,
        rot_mat,
        start_pos,
        attn_mask,
    )
    assert isinstance(tt_out, list)  # tt_out should be replicated on N devices
    tt_out = tt_out[0]
    tt_output_torch = tt2torch_tensor(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

    # input = torch.randn(1, 11, 4096)
    # seqlen = input.shape[1]
    empty_tensor = torch.zeros((1, 64))
    freqs_cis = torch.complex(empty_tensor, empty_tensor)
    # query_shape = [1, 11, model_args.n_heads, model_args.head_dim // 2]
    # key_shape = [1, 11, model_args.n_kv_heads, model_args.head_dim // 2]
    # bcast_freq_xq, bcast_freq_xk = get_freqs_cis(freqs_cis, query_shape, key_shape, device, output_mem_config)
    positions = torch.arange(0, 1)
    mask = torch.randn(1, 1)

    reference_output = reference_model(pt_attention_input, freqs_cis, positions, mask=mask)
    del reference_model

    """
    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_position = torch_to_tt_tensor_rm(positions, device, put_on_device=False)
    mask = torch_to_tt_tensor_rm(mask, device, put_on_device=False)
    mask = format_tensor(mask, tt_lib.tensor.Layout.TILE, device, output_mem_config, pad_value=-10000)
    tt_input = format_tensor(tt_input, tt_lib.tensor.Layout.TILE, device, output_mem_config)
    tt_output = tt_model(tt_input, bcast_freq_xq, bcast_freq_xk, tt_position, mask, seqlen)
    desired_shape = list(reference_output.shape)
    desired_shape.insert(0, 1)
    tt_output_torch = unpad_from_zero(tt_output, desired_shape).squeeze(0)
    """
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_Attention Passed!")
    else:
        logger.warning("Mistral_Attention Failed!")

    assert passing, f"Mistral_Attention output does not meet PCC requirement {pcc}."
