# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.mistral7b.tt.mistral_common import (
    precompute_freqs,
    generate_cos_sin_cache_ttnn,
    prepare_inputs_ttnn,
    sample,
)
from models.demos.mistral7b.tt.mistral_model import TtTransformer
from models.demos.mistral7b.tt.model_config import TtModelArgs
from models.demos.mistral7b.reference.model import Transformer
from models.demos.mistral7b.reference.tokenizer import Tokenizer

from models.perf.perf_utils import prep_perf_report
from models.utility_functions import profiler, enable_persistent_kernel_cache


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "iterations",
    (12,),
)
def test_mistral_model_inference(device, iterations, use_program_cache):
    dtype = ttnn.bfloat8_b

    run_ref_pt = True

    model_args = TtModelArgs(device)
    model_args.max_batch_size = 32
    model_args.n_layers = 32  # Full model
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("weight_loading")
    state_dict = torch.load(model_args.consolidated_weights_path)
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }
    profiler.end("weight_loading")

    prompts = ["This is a test"] * 32

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if run_ref_pt:
        profiler.start("Mistral_pytorch_ref_model_setup")
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(state_dict)

        cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
        freqs_cis = torch.complex(cos, sin)
        profiler.end("Mistral_pytorch_ref_model_setup")

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    profiler.start("TtMistral_model_setup")
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
    profiler.end("TtMistral_model_setup")

    generation_start_pos = 0
    generation_length = iterations
    seqlen = 1  # Generating one token per user at a time
    batch = 32

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    tt_decode_input = pt_decode_input

    profiler.disable()  # Disable profiler for first 10 iterations
    for i in range(generation_length):
        start_pos = generation_start_pos + i

        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.enable()
            enable_persistent_kernel_cache()
            profiler.start(f"input_processing_{i}")
        decode_input, start_pos, attn_mask, current_pos, rot_mat = prepare_inputs_ttnn(
            tt_decode_input,
            start_pos,
            model_args.dim,
            model_args.head_dim,
            model_args.sliding_window,
            model_args.max_seq_len,
            tt_model.device,
        )
        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.end(f"input_processing_{i}")
            profiler.start(f"model_run_for_inference_{i}")

        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask, rot_mat)

        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.end(f"model_run_for_inference_{i}")

        if run_ref_pt:  # Run reference model
            if i == 0:  # Skip the first few iterations to warm up
                profiler.start(f"ref_model_run_for_inference_{i}")

            freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
            positions = torch.tensor([start_pos])
            ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)

            if i == 0:  # Skip the first few iterations to warm up
                profiler.end(f"ref_model_run_for_inference_{i}")

        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode (temperature = 0) the generated token and save it to print out later
            tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)
            tt_decode_input = embd(tt_out_tok)
            if run_ref_pt:
                pt_out_tok = sample(ref_output, temperature=0, top_p=0.8)
                pt_decode_input = embd(pt_out_tok)

    profiler.print()
    comment = f"num_layers={model_args.n_layers}"
    weight_loading = profiler.get("weight_loading")
    input_processing = profiler.get("input_processing")
    ref_model_run_for_inference = profiler.get("ref_model_run_for_inference_0")
    first_iter_time = profiler.get("model_run_for_inference_0")
    second_iter_time = profiler.get("model_run_for_inference_10")

    prep_perf_report(
        model_name=f"Mistral7B",
        batch_size=model_args.max_batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=155,
        expected_inference_time=0.085,
        inference_time_cpu=ref_model_run_for_inference,
        comments=comment,
    )
