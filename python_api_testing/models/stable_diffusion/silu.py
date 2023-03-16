import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from pymetal import ttmetal as ttm
from utility_functions import tilize_to_list, print_diff_argmax, untilize, tilize, tilize_to_list

def torch_silu(x):
    return F.silu(x)


def TtSiLU(x):
    xs = ttm.tensor.sigmoid(x)
    xs = ttm.tensor.mul(xs, x)
    return xs



def run_silu_inference(device):

    input_shape =  [1, 1, 32, 32]
    input = torch.randn(input_shape) + 10

    torch_out = torch_silu(input)

    tt_input = ttm.tensor.Tensor(tilize_to_list(input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_out = TtSiLU(tt_input).to(host).data()
    tt_out = torch.Tensor(tt_out).reshape(torch_out.shape)
    tt_untilized_output = untilize(tt_out)
    print_diff_argmax(tt_untilized_output, torch_out)

    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)






if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_silu_inference(device)
    ttm.device.CloseDevice(device)
