# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn,
    sample,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.utility_functions import comp_pcc, comp_allclose, get_devices_for_t3000


from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.utility_functions import profiler, enable_persistent_kernel_cache, skip_for_grayskull


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch, iterations, expected_compile_time, expected_inference_time",
    ((32, 12, 155, 0.16),),
)
def test_mistral_model_perf(
    all_devices, batch, iterations, expected_compile_time, expected_inference_time, use_program_cache
):
    devices = all_devices
    num_devices = len(devices)
    assert num_devices == 8, "This test requires a T3000 (8 devices)"
    devices = get_devices_for_t3000(devices, num_devices)  # [ttnn.open_device(device_id=i) for i in range(8)]

    dtype = ttnn.bfloat8_b

    run_ref_pt = True

    model_args = TtModelArgs(devices[0])
    model_args.max_batch_size = batch
    model_args.n_layers = 1
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("weight_loading")
    state_dict = torch.load(model_args.state_dict_path)
    keys_dict = list(state_dict.keys())[:]
    remv = [f"layers.{i}" for i in range(model_args.n_layers, 32)]
    for k in keys_dict:
        if any([r in k for r in remv]):
            state_dict.pop(k)
    profiler.end("weight_loading")

    prompts = [""] * 32

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if run_ref_pt:
        profiler.start("Mistral_pytorch_ref_model_setup")
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(state_dict)
        profiler.end("Mistral_pytorch_ref_model_setup")

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    # TODO Add argmax + embedding on device, same as the demo.py code

    generation_start_pos = 0
    generation_length = iterations
    seqlen = 1  # Generating one token per user at a time
    batch = 32

    profiler.start("TtMistral_model_setup")

    # Load TTNN model
    tt_model = TtTransformer(
        devices=devices,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )
    # Load TTNN embedding module
    # tt_embd = TtMistralEmbedding(
    #     device=device,
    #     args=model_args,
    #     weight_cache_path=model_args.weight_cache_path(dtype),
    #     state_dict=state_dict,
    #     dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    # )
    profiler.end("TtMistral_model_setup")

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    tt_decode_input = pt_decode_input

    profiler.disable()  # Disable profiler for first 10 iterations
    for i in range(generation_length):
        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.enable()
            enable_persistent_kernel_cache()
            profiler.start(f"input_processing_{i}")

        start_pos = generation_start_pos + i
        current_pos = start_pos % model_args.sliding_window

        decode_input, rot_mat = prepare_inputs_ttnn(
            tt_decode_input,
            model_args.dim,
            model_args.head_dim,
            model_args.max_seq_len,
            tt_model.devices,
        )
        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.end(f"input_processing_{i}")
            profiler.start(f"model_run_for_inference_{i}")

        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, rot_mat)
        # Convert ttnn tensor to torch tensor
        # tt_output_torch = ttnn.to_torch(tt_out[0]).view(
        #     32, 1, -1
        # )  # permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        if i == 0 or i == 10:  # Skip the first few iterations to warm up
            profiler.end(f"model_run_for_inference_{i}")

        if run_ref_pt:  # Run reference model
            if i == 0:  # Skip the first few iterations to warm up
                profiler.start(f"ref_model_run_for_inference_{i}")

            positions = torch.LongTensor([start_pos])
            ref_output = reference_model(pt_decode_input, positions)

            if i == 0:  # Skip the first few iterations to warm up
                profiler.end(f"ref_model_run_for_inference_{i}")

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            # tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            tt_decode_input = embd(encoded_prompts_tensor[:, i].unsqueeze(-1))  # Embedding on device
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode (temperature = 0) the generated token and save it to print out later
            tt_out_tok = sample(ref_output, temperature=0, top_p=0.8)
            tt_decode_input = embd(tt_out_tok)  # Embedding on device
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
        model_name=f"Mixtral8x7B",
        batch_size=batch,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        inference_time_cpu=ref_model_run_for_inference,
        comments=comment,
    )
