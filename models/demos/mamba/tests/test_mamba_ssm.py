# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.tt.mamba_one_step_ssm import TtMambaSSM
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


class PytorchMambaSSM(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.block = hf_reference_model.layers[layer_num].mixer
        self.block.eval()

    def forward(self, x):
        result = self.block.ssm(x)
        return result


@pytest.mark.parametrize(
    "model_version, batch, pcc",
    (
        (
            "state-spaces/mamba-370m",
            1,
            0.98,
        ),
    ),
)
def test_mamba_ssm_inference(
    model_version: MambaPretrainedModelName,
    batch,
    pcc: float,
    device: tt_lib.device,
):
    torch.manual_seed(0)

    LAYER_NUM = 0

    reference_model = MambaDecode.from_pretrained(model_version)

    d_in = reference_model.args.d_model * reference_model.args.expand
    input = torch.ones(batch, 1, d_in)

    reference_output = PytorchMambaSSM(reference_model, LAYER_NUM)(input)

    mamba_block = reference_model.layers[LAYER_NUM].mixer
    model = TtMambaSSM(reference_model.args, device, mamba_block.state_dict())
    model_output = model(input)

    logger.info(comp_allclose(reference_output, model_output))

    does_pass, output_pcc = comp_pcc(reference_output, model_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
