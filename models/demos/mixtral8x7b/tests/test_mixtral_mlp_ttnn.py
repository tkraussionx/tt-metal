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


def test_mixtral_mlp_inference(device, reset_seeds):
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs()
    state_dict = torch.load(model_args.state_dict_path)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k: v for k, v in state_dict.items() if (k.startswith("layers.0.") and "attention" not in k and "norm" not in k)
    }

    partial_state_dict_ref = {k[32:]: v for k, v in partial_state_dict.items() if "experts.0" in k}
    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(partial_state_dict_ref)

    tt_model = TtMixtralMLP(
        device=device,
        state_dict=partial_state_dict,
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
