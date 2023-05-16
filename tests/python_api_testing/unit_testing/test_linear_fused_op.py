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

def Linear(in_features: int, out_features: int, weight, bias):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be the weight as a tilized list of values.
    """

    weight = weight
    bias = bias

    def linear_(activation):
        weight_T = tensor.transpose(weight)
        output = tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tensor.bcast(output, bias, tensor.BcastOpMath.ADD, tensor.BcastOpDim.H)
            return output_plus_bias

        return output

    return linear_

def make_linear(in_features: int, out_features: int, weights, bias, device):
    weights = torch_to_tt_tensor_rm(weights, device, shape=[1, 1, out_features, in_features], put_on_device=False)
    bias = torch_to_tt_tensor_rm(bias, device, shape=[1, 1, 1, out_features], put_on_device=False) if bias is not None else None
    output = Linear(in_features, out_features, weights, bias)
    return output


def test_run_fused_linear(input_size=1280):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    ttl.device.SetDefaultDevice(device)

    inputs = torch.randn(input_size, input_size)
    weights = torch.randn(input_size, input_size)
    biases = torch.randn(1, input_size)
    input_tt = torch_to_tt_tensor_rm(inputs, device,[1, 1, input_size, input_size], put_on_device=True)
    linear_func = make_linear(input_size,input_size,weights, biases, host)
    torch_linear_func = torch.nn.Linear(input_size, input_size)
    torch_linear_func.weight = torch.nn.Parameter(weights)
    torch_linear_func.bias = torch.nn.Parameter(biases)
    torch_linear_func.eval()

    for i in range(0, 10):
        input_tt = linear_func(input_tt)
        inputs = torch_linear_func(inputs)
        # print(i, comp_pcc(inputs, tt_to_torch_tensor(input_tt, host)))
        passing_pcc, output_pcc = comp_pcc(inputs, tt_to_torch_tensor(input_tt, host), 0.99)
        print(i)
        print("Passing=", passing_pcc)
        print("Output pcc=", output_pcc)


    ttl.device.CloseDevice(device)
