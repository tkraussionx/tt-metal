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
from models.demos.mixtral8x7b.tt.mixtral_model_ttnn import TtTransformer
from models.demos.mixtral8x7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mixtral8x7b.reference.model import Transformer
from models.demos.mixtral8x7b.reference.tokenizer import Tokenizer
from models.utility_functions import comp_pcc, comp_allclose, get_devices_for_t3000


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.parametrize(
    "n_layers",
    (1,),
)
@pytest.mark.parametrize(
    "iterations",
    (1,),
)
def test_mixtral_model_inference(all_devices, iterations, n_layers, reset_seeds):
    pcc = 0.99
    dtype = ttnn.bfloat8_b

    # Can avoid running reference model to speed up the test (unless measuring PCC)
    run_ref_pt = True

    devices = all_devices
    num_devices = len(devices)
    assert num_devices == 8, "This test requires a T3000 (8 devices)"
    devices = get_devices_for_t3000(devices, num_devices)  # [ttnn.open_device(device_id=i) for i in range(8)]

    model_args = TtModelArgs()
    model_args.n_layers = n_layers

    state_dict = {}
    for i in range(1 + (n_layers - 1) // 4):
        state_dict_i = torch.load(model_args.consolidated_weights_path(i), map_location="cpu")
        state_dict.update(state_dict_i)

    partial_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }

    base_address = "feed_forward."
    for l in range(model_args.n_layers):
        pre = f"layers.{l}."
        partial_state_dict[pre + base_address + "gate.weight"] = partial_state_dict[
            pre + "block_sparse_moe.gate.weight"
        ]
        del partial_state_dict[pre + "block_sparse_moe.gate.weight"]

        w1 = partial_state_dict[pre + "block_sparse_moe.w1"].view(8, 14336, 4096)
        w2 = partial_state_dict[pre + "block_sparse_moe.w2"].view(8, 4096, 14336)
        w3 = partial_state_dict[pre + "block_sparse_moe.w3"].view(8, 14336, 4096)
        for i in range(8):
            partial_state_dict[pre + base_address + f"experts.{i}.w1.weight"] = w1[i]
            partial_state_dict[pre + base_address + f"experts.{i}.w2.weight"] = w2[i]
            partial_state_dict[pre + base_address + f"experts.{i}.w3.weight"] = w3[i]
        partial_state_dict.pop(pre + "block_sparse_moe.w1")
        partial_state_dict.pop(pre + "block_sparse_moe.w2")
        partial_state_dict.pop(pre + "block_sparse_moe.w3")

    tokenizer = Tokenizer(model_args.tokenizer_path)

    # TODO Update the prompt
    # prompts = ["It_was_the_best_of_times_"] * 32
    prompts = [""] * 32
    # Space token -> (U+2581) == "▁"

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if run_ref_pt:
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(partial_state_dict)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    # Helper function supports multiple devices but we are only using one in this demo
    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache_ttnn(
        devices, model_args.head_dim, model_args.max_seq_len * 2, 10000, dtype
    )

    # Load TTNN model
    tt_model = TtTransformer(
        devices=devices,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
        tt_cos_cached=tt_cos_cached,
        tt_sin_cached=tt_sin_cached,
    )

    generation_start_pos = 0
    generation_length = iterations
    if run_ref_pt:
        all_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = 32

    if run_ref_pt:
        cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
        freqs_cis = torch.complex(cos, sin)

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    tt_decode_input = pt_decode_input

    # Keep track of generated outputs to print out later
    all_outputs = []
    if run_ref_pt:
        all_outputs_ref = []

    # After loading the model weights, wait for an input to start the generation
    # print("Waiting for an input to start...")
    # input()

    for i in range(generation_length):
        print(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i

        decode_input, start_pos, attn_mask, current_pos, rot_mat = prepare_inputs_ttnn(
            tt_decode_input,
            start_pos,
            model_args.dim,
            model_args.head_dim,
            model_args.sliding_window,
            model_args.max_seq_len,
            tt_model.devices,
        )

        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask, rot_mat)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out[0]).squeeze(1).view(batch, seqlen, -1)  # [seq, batch, hidden_dim]

        if run_ref_pt:  # Run reference model
            freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
            positions = torch.tensor([start_pos])
            # mask = tt2torch_tensor(attn_mask[0])
            ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)  # mask)

        print(f"encoded_prompts[0] = {len(encoded_prompts[0])}")
        if i in range(len(encoded_prompts[0])):
            all_outputs.append(tokenizer.decode([encoded_prompts[0][i]]))
            if run_ref_pt:
                all_outputs_ref.append(tokenizer.decode([encoded_prompts[0][i]]))

            print("Prefilling...")
            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Decode the generated token and save it to print out later
            tt_out_tok = torch.argmax(tt_output_torch, dim=-1).squeeze(1)
            tt_decode_input = embd(tt_out_tok)
            all_outputs.append(tokenizer.decode(tt_out_tok.tolist()[0]))
            if run_ref_pt:
                pt_out_tok = torch.argmax(ref_output, dim=-1).squeeze(1)
                pt_decode_input = embd(pt_out_tok)
                all_outputs_ref.append(tokenizer.decode(pt_out_tok.tolist()[0]))

        # Measure PCC
        if run_ref_pt:
            passing, pcc_message = comp_pcc(
                ref_output.view(batch, seqlen, -1), tt_output_torch.view(batch, seqlen, -1), pcc
            )

            logger.info(comp_allclose(ref_output, tt_output_torch))
            logger.info(pcc_message)

            if passing:
                logger.info("Mistral Model Passed!")
            else:
                logger.warning("Mistral Model Failed!")
                all_tests_pass = False

        # TODO Space decoding is currently not working as expected
        # TODO print All 32 users
        print("[User 0] TT generation: ", "".join(all_outputs))
        if run_ref_pt:
            print("[User 0] Ref generation: ", "".join(all_outputs_ref))

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} Mistral decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Mistral decode Failed!")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
