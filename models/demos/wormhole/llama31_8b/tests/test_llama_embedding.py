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
from models.demos.wormhole.llama31_8b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.wormhole.llama31_8b.reference.tokenizer import Tokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


def test_llama_embedding(device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16

    model_args = TtModelArgs(device)
    state_dict = torch.load(model_args.consolidated_weights_path)
    tokenizer = Tokenizer(model_args.tokenizer_path)

    reference_emb = Emb()
    reference_emb.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    tt_emb = TtLlamaEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=dtype,
    )

    prompts = ["Joy"] * 32
    pt_input = torch.tensor([tokenizer.encode(prompt) for prompt in prompts])
    reference_output = reference_emb(pt_input)
    logger.info(f"reference_output: {reference_output.shape}")

    tt_input = ttnn.from_torch(pt_input, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_output = tt_emb(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    logger.info(f"tt_output_torch: {tt_output_torch.shape}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Llama_embedding Passed!")
    else:
        logger.warning("Llama_embedding Failed!")

    assert passing, f"Llama_embedding output does not meet PCC requirement {0.99}."
