# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.mixtral8x7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mixtral8x7b.tt.mixtral_mlp_ttnn import TtMixtralMLP
from models.demos.mixtral8x7b.reference.model import FeedForward, RMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


def test_mistral_mlp_inference(device, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs()
    state_dict = torch.load(model_args.consolidated_weights_path(0))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[9:]: v
        for k, v in state_dict.items()
        if (k.startswith("layers.0.") and "attention" not in k and "norm" not in k)
    }

    partial_state_dict["gate.weight"] = partial_state_dict["block_sparse_moe.gate.weight"]
    del partial_state_dict["block_sparse_moe.gate.weight"]

    w1 = partial_state_dict["block_sparse_moe.w1"].view(8, 14336, 4096)
    w2 = partial_state_dict["block_sparse_moe.w2"].view(8, 4096, 14336)
    w3 = partial_state_dict["block_sparse_moe.w3"].view(8, 14336, 4096)
    for i in range(8):
        partial_state_dict[f"experts.{i}.w1.weight"] = w1[i]
        partial_state_dict[f"experts.{i}.w2.weight"] = w2[i]
        partial_state_dict[f"experts.{i}.w3.weight"] = w3[i]
    partial_state_dict.pop("block_sparse_moe.w1")
    partial_state_dict.pop("block_sparse_moe.w2")
    partial_state_dict.pop("block_sparse_moe.w3")

    partial_state_dict = {k[10:]: v for k, v in partial_state_dict.items() if k.startswith("experts.0.")}

    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtMixtralMLP(
        device=device,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        expert_num=0,
        dtype=dtype,
    )

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    rms_state_dict = {k[18:]: v for k, v in state_dict.items() if (k.startswith("layers.0.ffn_norm."))}
    rms = RMSNorm(dim=model_args.dim)
    rms.load_state_dict(rms_state_dict)

    torch_input = (torch.rand(1, 1, 32, model_args.dim) * 2) - 1
    torch_input = rms(torch_input)  # apply rmsnorm to input
    torch.save(torch_input, "ff_norm_input.pt")

    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )

    logger.info("Compilation pass for Mistral_MLP")
    tt_output = tt_model(tt_input)

    logger.info("Performance pass for Mistral_MLP")
    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_MLP Passed!")
    else:
        logger.warning("Mistral_MLP Failed!")

    assert passing, f"Mistral_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
