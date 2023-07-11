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
import python_api_testing.models.codegen.tt.codegen_split_heads as codegen_split_heads
from transformers import CodeGenConfig, CodeGenModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

def run_codegen_split_heads_test(device, pcc):

    model_hf = CodeGenModel.from_pretrained('Salesforce/codegen-350M-mono')
    sd = model_hf.state_dict()
    model_hf.eval()
    block = 0

    test_in = torch.rand(1, 1, 1024)

    tt_test_in = torch2tt_tensor(test_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

    config = CodeGenConfig('Salesforce/codegen-350M-mono')

    embed_dim = config.hidden_size
    num_attention_heads = config.num_attention_heads
    head_dim = embed_dim // num_attention_heads
    mp_num = 4


    pt_out = codegen_split_heads.pt_split_heads(test_in, num_attention_heads, head_dim, mp_num=mp_num)
    print(pt_out.shape)

    tt_out = codegen_split_heads.tt_split_heads(tt_test_in, num_attention_heads, head_dim, mp_num=mp_num)

    tt_out_converted = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("codegen_split_heads: Passed!")
    else:
        logger.warning("codegen_split_heads: Failed!")

    assert does_pass

@pytest.mark.parametrize(
    "pcc",
    (
        (
            0.99,
        ),
    ),
)
def test_codegen_split_heads(pcc):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    run_codegen_split_heads_test(device, pcc)
    tt_lib.device.CloseDevice(device)
