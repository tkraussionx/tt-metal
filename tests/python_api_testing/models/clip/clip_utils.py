from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")

import torch
from loguru import logger

from helper_funcs import Linear as linear
from utility_functions_new import torch_to_tt_tensor_rm, comp_pcc, comp_allclose_and_pcc
import tt_lib


def make_address(base_address, op_name):
    return op_name if base_address == "" else f"{base_address}.{op_name}"


def make_linear(in_feature, out_feature, op_name, state_dict, base_address, device):
    q_weight = state_dict[make_address(base_address, f"{op_name}.weight")]
    q_weight = torch_to_tt_tensor_rm(q_weight, device)
    if make_address(base_address, f"{op_name}.bias") in state_dict:
        q_bias = state_dict[make_address(base_address, f"{op_name}.bias")]
        q_bias = torch_to_tt_tensor_rm(q_bias, device)
    else:
        q_bias = None
    return linear(in_feature, out_feature, weight=q_weight, bias=q_bias)

def setup_device_and_host():
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()
    return device, host

def compare_and_display(golden_output: torch.Tensor, tt_output: torch.Tensor, name: str, pcc: float) -> None:
    pcc_passing, _ = comp_pcc(golden_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(golden_output, tt_output, pcc)
    logger.info(f"{name}: {pcc_output}")
    assert(
        pcc_passing
    ), f"Model output does not meet PCC requirement {pcc}."
    logger.info(f"{name} test passed!")
