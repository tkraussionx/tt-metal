# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from time import time
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
    (1, 3, 11),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mistral_model_inference(pcc, model_config, model_location_generator, device, iterations, n_layers):
    ttnn.enable_program_cache()
    prompts = [
        "This is a sample text for single layer execution ",
    ]
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

    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)

    # TODO Scale the model (mixtral) to multiple devices when T3000 is available
    devices = [
        device,
    ]

    tt_cos_cached, tt_sin_cached = generate_cos_sin_cache_ttnn(
        devices, model_args.head_dim, "", model_args.max_seq_len * 2, 10000, dtype
    )
    tt_model = TtTransformer(
        args=model_args,
        devices=devices,
        dtype=dtype,
        state_dict=state_dict,
        model_config=model_config,
        layers=list(range(model_args.n_layers)),
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
    times = []
    for i in range(generation_length):
        start = time()
        print(f"[Model] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        start_pos = generation_start_pos + i

        decode_input, start_pos, attn_mask, current_pos = prepare_inputs_ttnn(
            tt_decode_input,
            start_pos,
            model_args.dim,
            model_args.n_heads // len(devices),
            model_args.sliding_window,
            tt_model.devices,
            tt_model.num_devices,
        )
        # Run TT model
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask)
        # tt_output = tt_model(tt_input, bcast_freq_xq, bcast_freq_xk, tt_position, mask, seqlen)

        tt_output_torch = ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        freqs_cis_i = freqs_cis[start_pos, :].unsqueeze(0)
        positions = torch.tensor([start_pos])

        # Reference model
        # mask = tt2torch_tensor(attn_mask[0])
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions)  # mask)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        if passing:
            logger.info("Mistral Model Block Passed!")
        else:
            logger.warning("Mistral Model Block Failed!")
            all_tests_pass = False
        duration = time() - start
        if i > 0:  # skip initial step
            times.append(duration)
        logger.info(f"Time taken for token {i}: {duration:.2f} seconds")

    if times:
        logger.info(f"Average tokens/s/user: {1/(sum(times)/len(times)):.2f}")

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
