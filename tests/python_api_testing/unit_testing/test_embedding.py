import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    pad_activation,
    pad_weight,
    tilize,
    untilize,
    tilize_to_list,
    print_diff_argmax,
    pad_weight,
)


def test_embedding():
    host = ttl.device.GetHost()
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    # Power of 2 reshape
    N = 1
    C = 1
    H = 128
    W = 128
    x = torch.rand(N * C * H * W).reshape(N, C, H, W).float()
    xp = pad_activation(x).view(-1).tolist()
    xtt = ttl.tensor.Tensor(
        xp,
        [N, C, H, W],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device,
    )

    reshaped = ttl.tensor.reshape(xtt, 1, 128, 2, 64)
    reshaped = torch.Tensor(reshaped.to(host).data()).reshape(reshaped.shape())
    torch_reshaped = torch.Tensor(x).reshape(1, 128, 2, 64)
    assert (abs(torch_reshaped - reshaped) < 0.02).all().item(), "Failure"
    ttl.device.CloseDevice(device)
