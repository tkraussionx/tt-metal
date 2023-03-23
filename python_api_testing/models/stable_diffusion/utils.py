
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch

from pymetal import ttlib as ttl
from utility_functions import untilize, tilize, tilize_to_list



def move_to_cpu(x, host):
    x_shape = x.shape()
    x = x.to(host).data()
    x = torch.tensor(x).reshape(x_shape)
    return untilize(x)


def move_to_device(x, device):
    x_shape = x.shape
    x = ttl.tensor.Tensor(tilize_to_list(x), x_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    return x
