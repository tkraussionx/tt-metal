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
from utils import move_to_device, move_to_cpu



class TtUpsampledNearest2d(nn.Module):
    def __init__(self, scale_factor=2.0, device=None, host=None):
        super().__init__()

        assert scale_factor % 1 == 0 and scale_factor > 0, "We only support scaling by positive integer values"
        self.scale_factor = int(scale_factor)
        self.device = device
        self.host = host


    def forward(self, input):
        input_shape = input.shape()
        output_shape = list(input.shape())
        output_shape[-1] *= 2
        output_shape[-2] *= 2

        input = move_to_cpu(input, self.host)

        input = torch.repeat_interleave(input, repeats= self.scale_factor, dim=-1)
        input = torch.repeat_interleave(input, repeats=self.scale_factor, dim=-2)

        input = move_to_device(input, self.device)

        return input






def run_upsample_nearest_inference(device, host):
    input_shape =  [1, 1, 32, 32]
    input = torch.randn(input_shape)

    torch_out = F.interpolate(input, scale_factor=2.0, mode="nearest")

    tt_input = ttm.tensor.Tensor(tilize_to_list(input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_up = TtUpsampledNearest2d(scale_factor=2.0, device=device, host=host)
    tt_out = tt_up(tt_input).to(host).data()
    tt_out = torch.Tensor(tt_out).reshape(torch_out.shape)
    tt_untilized_output = untilize(tt_out)
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)


if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_upsample_nearest_inference(device, host)
    ttm.device.CloseDevice(device)
