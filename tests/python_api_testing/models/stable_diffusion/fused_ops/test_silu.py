import numpy as np
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch.nn import functional as F
from loguru import logger

from libs import tt_lib as ttl
from utility_functions import print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from python_api_testing.fused_ops.silu import SiLU as TtSiLU
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def torch_silu(x):
    return F.silu(x)


def run_test_silu_inference(device, host):
    input_shape =  [1, 1, 32, 32]
    input = torch.randn(input_shape) + 10

    torch_out = torch_silu(input)
    tt_input = torch_to_tt_tensor(input, device)

    tt_out = TtSiLU(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)
    print_diff_argmax(tt_out, torch_out)

    does_pass, pcc_message = comp_pcc(torch_out, tt_out, 0.99)

    print(comp_allclose(torch_out, tt_out))
    print(pcc_message)


    if does_pass:
        logger.info("test_silu_inference Passed!")
    else:
        logger.warning("test_silu_inference Failed!")
    assert does_pass


def test_silu_inference():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_test_silu_inference(device, host)
    ttl.device.CloseDevice(device)
