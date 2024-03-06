# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.mistral7b.tt.mistral_common_ttnn import (
    precompute_freqs,
    generate_cos_sin_cache_ttnn,
    prepare_inputs_ttnn,
)
from models.demos.mistral7b.tt.mistral_model_ttnn import TtTransformer
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mistral7b.reference.model import Transformer
from models.demos.mistral7b.reference.tokenizer import Tokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.parametrize(
    "n_layers",
    (32,),
)
@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT8-DRAM"),
)
@pytest.mark.parametrize(
    "iterations",
    (17),
)
@pytest.mark.parametrize(
    "pcc",
    (0.97,),
)
def test_mistral_model_inference(device, pcc, model_config, iterations, n_layers):
    ttnn.enable_program_cache()

    # Avoid running reference model to speed up the test (unless measuring PCC)
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = False  # Flag to measure KV cache PCC for all layers

    dtype_str, mem_config_str = model_config.split("-")
    if dtype_str == "BFLOAT16":
        dtype = ttnn.bfloat16
    elif dtype_str == "BFLOAT8":
        dtype = ttnn.bfloat8_b
    else:
        raise ValueError(f"Unknown dtype {dtype_str}")

    model_args = TtModelArgs()
    model_args.max_batch_size = 32
    model_args.n_layers = n_layers
    state_dict = torch.load(model_args.consolidated_weights_path)
    tokenizer = Tokenizer(model_args.tokenizer_path)
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }

    prompts = ["This is a test"] * 32

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if run_ref_pt:
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(state_dict)

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    # Helper function supports multiple devices but we are only using one in this demo
    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache_ttnn(
        [device], model_args.head_dim, model_args.max_seq_len * 2, 10000, dtype
    )

    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layers=list(range(model_args.n_layers)),
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
            tt_model.device,
        )

        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask, rot_mat)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        if run_ref_pt:  # Run reference model
            freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
            positions = torch.tensor([start_pos])
            # mask = ttnn.to_torch(attn_mask[0])
            ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            all_outputs.append(encoded_prompts[0][i])  # Update list of TT outputs
            if run_ref_pt:
                all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs

            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode the generated token and save it to print out later
            tt_out_tok = torch.argmax(tt_output_torch, dim=-1)
            tt_decode_input = embd(tt_out_tok)
            all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of TT outputs

            if run_ref_pt:
                pt_out_tok = torch.argmax(ref_output, dim=-1)
                pt_decode_input = embd(pt_out_tok)
                all_outputs_ref.append(
                    pt_out_tok.squeeze(1).tolist()[0]
                )  # Update generated token to list of ref outputs

        # Measure PCC if also running reference model
        if run_ref_pt:
            passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

            logger.info(comp_allclose(ref_output, tt_output_torch))
            logger.info(f"Model output: {pcc_message}")

            if passing:
                logger.info("Mistral Model Passed!")
            else:
                logger.warning("Mistral Model Failed!")
            if not passing:
                all_tests_pass = False

            # Compare V caches
            if cache_pcc:
                for i in range(n_layers):
                    pytorch_layer_present = [
                        reference_model.layers[i]
                        .attention.cache_k.clone()
                        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                        reference_model.layers[i]
                        .attention.cache_v.clone()
                        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                    ]

                    tt_layer_present = []
                    for layer_past in tt_model.layers[i].attention.layer_past_list[0]:
                        tt_layer_present.append(ttnn.to_torch(layer_past))

                    for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                        cache_length_to_check = min(
                            model_args.sliding_window, generation_start_pos + generation_length + 1
                        )
                        cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                        cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                        does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                        if i == 0:
                            logger.info(f"K cache output: {output_pcc}")
                        else:
                            logger.info(f"V cache output: {output_pcc}")

                        if does_pass:
                            logger.info(f"V Cache Passed!")
                        else:
                            logger.warning(f"V Cache Failed! PCC value is lower than {pcc}")
                        # if not does_pass:
                        # all_tests_pass = False

        # TODO print all 32 users
        print("[TT generation User 0] ", "".join(tokenizer.decode(all_outputs)))
        if run_ref_pt:
            print("[Ref generation User 0] ", "".join(tokenizer.decode(all_outputs_ref)))

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} Mistral decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Mistral decode had bad PCC")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
