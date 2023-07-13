from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib
import pytest

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.codegen.tt.codegen_fixed_pos_embed as codegen_fixed_pos_embed
from transformers import CodeGenConfig, CodeGenModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

def run_codegen_fixed_pos_embed_test(device, pcc):

    model_hf = CodeGenModel.from_pretrained('Salesforce/codegen-350M-mono')
    sd = model_hf.state_dict()
    model_hf.eval()
    block = 0

    test_in = torch.rand(1, 1, 256, 256)

    tt_test_in = torch2tt_tensor(test_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

    pt_out = codegen_fixed_pos_embed.pt_fixed_pos_embed(test_in, seq_dim=1, seq_len=None)

    tt_out = codegen_fixed_pos_embed.tt_fixed_pos_embed(tt_test_in, device, seq_dim=1, seq_len=None)

    tt_out_converted = tt2torch_tensor(tt_out[0])

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("codegen_fixed_pos_embed: Passed!")
    else:
        logger.warning("codegen_fixed_pos_embed: Failed!")

    assert does_pass

@pytest.mark.parametrize(
    "pcc",
    (
        (
            0.99,
        ),
    ),
)
def test_codegen_merge_heads(pcc):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_codegen_fixed_pos_embed_test(device, pcc)
    tt_lib.device.CloseDevice(device)
