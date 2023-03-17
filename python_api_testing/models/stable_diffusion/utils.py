
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch

from pymetal import ttmetal as ttm
from utility_functions import untilize, tilize, tilize_to_list



def move_to_cpu(x, host):
    x_shape = x.shape()
    x = x.to(host).data()
    x = torch.tensor(x).reshape(x_shape)
    return untilize(x)


def move_to_device(x, device):
    x_shape = x.shape
    x = ttm.tensor.Tensor(tilize_to_list(x), x_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    return x
