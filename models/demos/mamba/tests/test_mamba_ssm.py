# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.mamba.reference.decode_model import (
    MambaDecode,
)
from models.demos.mamba.tt.mamba_one_step_ssm import TtMambaSSM
from models.demos.mamba.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchMambaOneStepSSMModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mamba_block = hf_reference_model.layers[layer_num].mixer
        # Disable dropout
        self.mamba_block.eval()

    def forward(self, x):
        result = self.mamba_block.ssm(x)
        return result


def run_test_MambaSSM_inference(device, model_version, batch, seq_len, pcc, model_config, tt_cache_path):
    # Load the model
    pytorch_model = MambaDecode.from_pretrained(model_version)
    d_in = pytorch_model.args.d_model * pytorch_model.args.expand
    # Get reference to Mamba block
    layer_num = 0
    mamba_block = pytorch_model.layers[layer_num].mixer

    # Prepare input
    torch.manual_seed(0)
    ssm_input = torch.rand(batch, d_in)

    # PyTorch output --------------------------------------------------------------------
    pytorch_out = pytorch_model(ssm_input)
    tt_out = torch.rand(batch, d_in)
    # # TT hardware execution -------------------------------------------------------------
    # tt_FalconMLP_model = TtFalconMLP(
    #     device,
    #     state_dict,
    #     base_url,
    #     layer_num,
    #     configuration.hidden_size,
    #     model_config,
    #     tt_cache_path,
    # )

    # tt_mlp_input = torch2tt_tensor(mlp_input, device)

    # tt_out = tt_FalconMLP_model(tt_mlp_input)
    # tt_out = tt2torch_tensor(tt_out)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "state-spaces/mamba-370m",
            1,
            128,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_MambaSSM_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    device,
):
    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    run_test_MambaSSM_inference(device, model_version, batch, seq_len, pcc, model_config, tt_cache_path)
