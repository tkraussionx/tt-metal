# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os

# Set Llama flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["LLAMA31_8B_CKPT_DIR"] = "/mnt/MLPerf/ttnn/models/demos/llama31_8b/"
    os.environ["LLAMA31_8B_TOKENIZER_PATH"] = "/mnt/MLPerf/ttnn/models/demos/llama31_8b/"
    os.environ["LLAMA31_8B_CACHE_PATH"] = "/mnt/MLPerf/ttnn/models/demos/llama31_8b/"

import ttnn
from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs
from models.demos.wormhole.llama31_8b.tt.llama_mlp import TtLlamaMLP
from transformers.models.llama.modeling_llama import LlamaMLP as RefLlamaMLP
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from safetensors.torch import load_file


@skip_for_grayskull("Requires wormhole_b0 to run")
def test_llama_mlp_inference(device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b
    model_args = TtModelArgs(device=device)
    state_dict = load_file(model_args.weights_index_path)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    prefix = "model.layers.0.mlp."
    partial_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if (k.startswith(prefix))}

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = RefLlamaMLP(config=model_args)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtLlamaMLP(
        device=device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
    )
    torch_input = torch.randn(1, 1, 17, 4096)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )

    logger.info("Compilation pass for Llama_MLP")
    tt_output = tt_model(tt_input)

    logger.info("Performance pass for Llama_MLP")
    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    pcc_required = 0.985  # TODO: why not .99?
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Llama_MLP Passed!")
    else:
        logger.warning("Llama_MLP Failed!")

    assert passing, f"Llama_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
