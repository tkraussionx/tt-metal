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
import python_api_testing.models.codegen.tt.codegen_rotate_every_two as codegen_rotate_every_two
from transformers import CodeGenConfig, CodeGenModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

def run_codegen_rotate_every_two_test(device, pcc):

    test = torch.arange(1, 32*32+1)
    test_in = test.reshape(1, 1, 32, 32)


    tt_test_in = torch2tt_tensor(test_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

    pt_out = codegen_rotate_every_two.pt_rotate_every_two(test_in)

    tt_out = codegen_rotate_every_two.tt_rotate_every_two(tt_test_in, device)

    tt_out_converted = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("codegen_rotate_every_two: Passed!")
    else:
        logger.warning("codegen_rotate_every_two: Failed!")

    assert does_pass

@pytest.mark.parametrize(
    "pcc",
    (
        (
            0.99,
        ),
    ),
)
def test_codegen_rotate_every_two(pcc):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    run_codegen_rotate_every_two_test(device, pcc)
    tt_lib.device.CloseDevice(device)
