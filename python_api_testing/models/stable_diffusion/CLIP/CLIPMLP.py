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
from python_api_testing.fused_ops.linear import Linear as tt_linear


class CLIPMLP(nn.Module):
    def __init__(self, state_dict, config=None, hidden_size=None, intermediate_size=None):
        super().__init__()

        fc1_weight = state_dict["text_model.encoder.layers.10.mlp.fc1.weight"]
        fc1_bias = state_dict["text_model.encoder.layers.10.mlp.fc1.bias"]
        fc2_weight = state_dict["text_model.encoder.layers.10.mlp.fc2.weight"]
        fc2_bias = state_dict["text_model.encoder.layers.10.mlp.fc2.bias"]


        self.config = config
        hidden_size = config.hidden_size if config else hidden_size
        intermediate_size = config.intermediate_size if config else intermediate_size
        # self.activation_fn = ACT2FN[config.hidden_act] # this is gelu
        self.activation_fn = torch.nn.functional.gelu
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc1.weight = nn.Parameter(fc1_weight)
        self.fc1.bias = nn.Parameter(fc1_bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.fc2.weight = nn.Parameter(fc2_weight)
        self.fc2.bias = nn.Parameter(fc2_bias)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states



class TtCLIPMLP(torch.nn.Module):
    def __init__(self,  device, state_dict, config=None, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.linear1_weight = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.mlp.fc1.weight"]))
        self.linear1_bias = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.mlp.fc1.bias"]))
        self.linear2_weight = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.mlp.fc2.weight"]))
        self.linear2_bias = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.mlp.fc2.bias"]))

        # Note: Load Weights
        self.config = config
        hidden_size = config.hidden_size if config else hidden_size
        intermediate_size = config.intermediate_size if config else intermediate_size

        self.linear_1 = tt_linear(hidden_size, intermediate_size, self.linear1_weight, bias=self.linear1_bias, device=device)

        self.linear_2 = tt_linear(intermediate_size, hidden_size, self.linear2_weight, bias=self.linear2_bias, device=device)



    def forward(self, x):

        x = self.linear_1(x)
        x = ttm.tensor.gelu(x)
        x = self.linear_2(x)
        return x



def run_clip_mlp_inference(device):

    from transformers import CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    state_dict = model.state_dict()
    config = model.config.text_config

    hidden_size = config.hidden_size
    intermediate_size= config.intermediate_size
    input_shape = [1, 2, 32, hidden_size]
    input = torch.randn(input_shape)

    # torch_mlp = CLIPMLP(hidden_size = hidden_size, intermediate_size=intermediate_size)
    torch_mlp = CLIPMLP(config=config, state_dict=state_dict)
    torch_out = torch_mlp(input)

    tt_input = ttm.tensor.Tensor(tilize_to_list(input), input_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    # tt_mlp = TtCLIPMLP(device, hidden_size=hidden_size, intermediate_size=intermediate_size)
    tt_mlp = TtCLIPMLP(device, config=config, state_dict=state_dict)

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
