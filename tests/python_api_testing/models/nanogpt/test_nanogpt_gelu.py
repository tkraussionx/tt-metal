from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib

from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils
import python_api_testing.models.nanogpt.nanogpt_gelu as nanogpt_gelu


def run_nanogpt_gelu_test(device):

    # Prepare input
    torch.manual_seed(0)
    test_in = torch.rand(1, 1, 61, 1024) / 1024

    pt_out = nangpt_gelu.new_gelu(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)
    tt_out = nangpt_gelu.tt_new_gelu(tt_test_in, device)
    tt_out_converted = nangpt_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("nanogpt_gelu: Passed!")
    else:
        logger.warning("nanogpt_gelu: Failed!")

    assert does_pass


def test_nanogpt_gelu():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_nangpt_gelu_test(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_gelu_forward()
