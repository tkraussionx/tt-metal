# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
import tt_lib
from loguru import logger

from transformers import AutoModelForCausalLM
from models.experimental.llama2.reference.generation import Llama

from models.experimental.llama2.tt.llama2_configuration import ModelArgs
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_pcc,
    comp_allclose,
)
from models.experimental.llama2.tt.llama2_transformer_block import TtTransformerBlock


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_llama2_transformer_block(device, pcc, model_location_generator, reset_seeds):
    llama2_path = str(model_location_generator("llama-2-7b", model_subdir="llama-2"))
    facebook_research_reference_model = Llama.build(llama2_path, llama2_path, 50, 1)
    torch_model = facebook_research_reference_model.model.layers[0]
    state_dict = facebook_research_reference_model.model.state_dict()

    base_address = "layers.0"

    layer_id = 0
    input = torch.rand((1, 9, 4096))
    start_pos = 0
    empty_tensor = torch.empty((9, 64))
    freqs_cis = torch.complex(empty_tensor, empty_tensor)
    mask = torch.rand((1, 1, 9, 9))

    torch_output = torch_model(input, start_pos, freqs_cis, mask)

    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
    tt_mask = torch_to_tt_tensor_rm(mask, device, put_on_device=False)

    Tt_model = TtTransformerBlock(ModelArgs, layer_id, state_dict, base_address, device)

    tt_output = Tt_model(tt_input, start_pos, freqs_cis, tt_mask)
    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("Llama2Transformerblock Passed!")

    assert does_pass, f"Llama2Transformerblock does not meet PCC requirement {pcc}."
