import math
import numpy as np
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pymetal import ttmetal as ttm
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize
from pymetal.ttmetal.utils import print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as TtLinear


class CLIPMLP(nn.Module):
    def __init__(self, config=None, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size if config else hidden_size
        intermediate_size = config.intermediate_size if config else intermediate_size
        # self.activation_fn = ACT2FN[config.hidden_act] # this is gelu
        self.activation_fn = torch.nn.functional.gelu
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc1.weight.data.fill_(1)
        self.fc1.bias.data.fill_(0)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.fc2.weight.data.fill_(1)
        self.fc2.bias.data.fill_(0)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states



class TtCLIPMLP(torch.nn.Module):
    def __init__(self,  device, config=None, hidden_size=None, intermediate_size=None, state_dict=None):
        super().__init__()
        # Note: Load Weights
        self.config = config
        hidden_size = config.hidden_size if config else hidden_size
        intermediate_size = config.intermediate_size if config else intermediate_size
        weight1_shape = [1, 1, hidden_size, intermediate_size]
        self.linear1_weight = torch.ones(weight1_shape).flatten().tolist()
        self.linear1_bias = torch.zeros(weight1_shape).flatten().tolist()

        self.linear_1 = TtLinear(hidden_size, intermediate_size, self.linear1_weight, bias=self.linear1_bias, device=device)

        weight2_shape = [1, 1, intermediate_size, hidden_size]
        self.linear2_weight = torch.ones(weight2_shape).flatten().tolist()
        self.linear2_bias = torch.zeros(weight2_shape).flatten().tolist()
        self.linear_2 = TtLinear(intermediate_size, hidden_size, self.linear2_weight, bias=self.linear2_bias, device=device)



    def forward(self, x):

        x = self.linear_1(x)
        x = ttm.tensor.gelu(x)
        x = self.linear_2(x)
        return x



def run_clip_mlp_inference(device):
    hidden_size = 64
    intermediate_size=32
    input_shape = [1, 2, 32, hidden_size]
    input = torch.randn(input_shape)

    torch_mlp = CLIPMLP(hidden_size = hidden_size, intermediate_size=intermediate_size)
    torch_out = torch_mlp(input)

    tt_input = ttm.tensor.Tensor(tilize_to_list(input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    tt_mlp = TtCLIPMLP(device, hidden_size=hidden_size, intermediate_size=intermediate_size)

    tt_out = tt_mlp(tt_input).to(host).data()

    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)





if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_clip_mlp_inference(device)
    ttm.device.CloseDevice(device)
