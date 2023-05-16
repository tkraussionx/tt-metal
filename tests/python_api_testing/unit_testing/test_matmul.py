import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import numpy as np
import torch

from libs import tt_lib as ttl
from libs.tt_lib import tensor

from python_api_testing.models.utility_functions import untilize, tilize, tilize_to_list, torch_to_tt_tensor_rm, comp_pcc, tt_to_torch_tensor

def test_run_fused_linear(input_size=1280):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    ttl.device.SetDefaultDevice(device)

    inputs = torch.randn(input_size, input_size)
    weights = torch.randn(input_size, input_size)
    input_tt = torch_to_tt_tensor_rm(inputs, device,[1, 1, input_size, input_size], put_on_device=True)
    weights_tt = torch_to_tt_tensor_rm(weights, device,[1, 1, input_size, input_size], put_on_device=True)


    for i in range(0, 10):
        input_tt = tensor.matmul(input_tt, weights_tt)
        inputs = torch.matmul(inputs, weights)
        passing_pcc, output_pcc = comp_pcc(inputs, tt_to_torch_tensor(input_tt, host), 0.99)
        print(i)
        print("Passing=", passing_pcc)
        print("Output pcc=", output_pcc)


    ttl.device.CloseDevice(device)
