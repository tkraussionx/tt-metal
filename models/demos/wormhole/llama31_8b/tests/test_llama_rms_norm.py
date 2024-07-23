# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os

# Set Mistral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["LLAMA31_8B_CKPT_DIR"] = "/mnt/MLPerf/ttnn/models/demos/llama31_8b/"
    os.environ["LLAMA31_8B_TOKENIZER_PATH"] = "/mnt/MLPerf/ttnn/models/demos/llama31_8b/"
    os.environ["LLAMA31_8B_CACHE_PATH"] = "/mnt/MLPerf/ttnn/models/demos/llama31_8b/"

import ttnn
from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs
from models.common.rmsnorm import RMSNorm as TtRMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RefRMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from safetensors.torch import load_file


@skip_for_grayskull("Requires wormhole_b0 to run")
def test_llama_rms_norm_inference(device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(device)
    state_dict = load_file(model_args.weights_index_path)
    state_dict = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in state_dict.items()}

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    prefix = "layers.0.post_attention_layernorm."
    partial_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if (k.startswith(prefix))}
    reference_model = RefRMSNorm(hidden_size=model_args.dim)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtRMSNorm(
        device=device,
        dim=model_args.dim,
        state_dict=state_dict,
        layer_num=0,
        weight_key="post_attention_layernorm",
        weight_dtype=dtype,
    )
    input = torch.rand(1, 32, 4096)
    reference_output = reference_model(input)

    tt_input = ttnn.from_torch(
        input, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT
    )  # , device, put_on_device=False)

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_rms_norm Passed!")
    else:
        logger.warning("Mistral_rms_norm Failed!")

    assert passing, f"Mistral_rms_norm output does not meet PCC requirement {0.99}."
