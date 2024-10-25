# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_attention import TtLlamaAttention
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.llama3.tt.llama_common import (
    get_prefill_rot_mat,
    get_rot_transformation_mat,
)
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Attention, precompute_freqs_cis
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (2048,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_attention_inference(seq_len, mesh_device, use_program_cache, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaAttention", 0) + "."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = Attention(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    batch = 1

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, mesh_device, seq_len=seq_len)
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    tt_model = TtLlamaAttention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        configuration=model_args,
    )

    pt_attention_input = (torch.rand(batch, seq_len, model_args.dim) * 2) - 1
    tt_attention_input = pt_attention_input.clone()
    attention_input = model_args.prepare_inputs_ttnn_prefill(
        tt_attention_input,
        force_replicated=True,
    )

    tt_out = tt_model(attention_input, 0, rot_mats, transformation_mats, user_id=0, mode="prefill")
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[
        0, :, :, : model_args.dim
    ].view(
        batch, seq_len, -1
    )  # [ batch, seq, dim]

    positions = torch.LongTensor(range(seq_len))
    freqs_cis_i = precompute_freqs_cis(
        model_args.head_dim, model_args.max_seq_len * 2, model_args.rope_theta, model_args.use_scaled_rope
    )[positions]
    attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
    attn_mask_torch = torch.triu(attn_mask, diagonal=1)
    reference_output = reference_model(pt_attention_input, positions[0], freqs_cis_i, mask=attn_mask_torch)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info(f"Llama_Attention Passed!")
    else:
        logger.warning(f"Llama_Attention Failed!")
        all_tests_pass = False

    check_kv_cache = True  # May want to disable: Issue #10648
    if check_kv_cache:
        # PyTorch output --------------------------------------------------------------------
        pytorch_layer_present = [
            reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
            reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
        ]
        # TT hardware execution -------------------------------------------------------------
        tt_layer_present = [
            ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
            for cache in tt_model.layer_past
        ]

        for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
            cache_length_to_check = min(model_args.sliding_window, generation_start_pos + generation_length + 1)
            cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
            cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
            if i == 0:
                logger.info(f"K cache output: {output_pcc}")
            else:
                logger.info(f"V cache output: {output_pcc}")

            if does_pass:
                logger.info(f"KV Cache Passed!")
            else:
                logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    if all_tests_pass:
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
