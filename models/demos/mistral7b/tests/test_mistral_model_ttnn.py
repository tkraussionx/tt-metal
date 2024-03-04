# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from pathlib import Path
import ttnn
from models.demos.mistral7b.tt.mistral_common_ttnn import (
    precompute_freqs,
    generate_cos_sin_cache_ttnn,
    prepare_inputs_ttnn,
)
from models.demos.mistral7b.tt.mistral_model_ttnn import TtTransformer
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs, get_model_config
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
    (1, 3, 16, 32),
)
@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT8-DRAM"),
)
@pytest.mark.parametrize(
    "iterations",
    (1, 20, 127),
)
@pytest.mark.parametrize(
    "pcc",
    (0.99,),
)
def test_mistral_model_inference(pcc, model_config, model_location_generator, device, iterations, n_layers):
    ttnn.enable_program_cache()

    # Avoid running reference model to speed up the test (unless measuring PCC)
    run_ref_pt = False

    dtype_str, mem_config_str = model_config.split("-")
    if dtype_str == "BFLOAT16":
        dtype = ttnn.bfloat16
    elif dtype_str == "BFLOAT8":
        dtype = ttnn.bfloat8_b
    else:
        raise ValueError(f"Unknown dtype {dtype_str}")
    model_config = get_model_config(model_config)

    mistral_path = Path(model_location_generator(model_config["DEFAULT_CACHE_PATH"], model_subdir="mistral"))
    tokenizer = Tokenizer(str(Path(mistral_path) / "tokenizer.model"))

    # TODO Update the prompt
    # prompts = ["It_was_the_best_of_times_"] * 32
    prompts = [""] * 32
    # Space token -> (U+2581) == "▁"

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))
    model_args.max_batch_size = 32
    model_args.n_layers = n_layers

    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
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
        weight_cache_path=Path(model_config["DEFAULT_WEIGHT_PATH"]),
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

    # After loading the model weights, wait for an input to start the generation
    # print("Waiting for an input to start...")
    # input()

    for i in range(generation_length):
        print(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i

        decode_input, start_pos, attn_mask, current_pos = prepare_inputs_ttnn(
            tt_decode_input,
            start_pos,
            model_args.dim,
            model_args.sliding_window,
            tt_model.device,
        )

        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

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
            passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

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
