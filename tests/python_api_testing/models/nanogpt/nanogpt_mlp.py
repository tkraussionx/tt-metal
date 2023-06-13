import torch
from torch.nn import functional as F

import tt_lib
import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils

from transformers import GPT2LMHeadModel

class TtMLP(torch.nn.Module):
    def __init__(self, base_address, state_dict, device):
        super().__init__()
        print(base_address)
        # Get the weights
        self.tt_weight_c_fc = state_dict[f"{base_address}.c_fc.weight"]
        self.tt_weight_c_proj = state_dict[f"{base_address}.c_proj.weight"]


        # Transpose the weights
        #self.tt_weight_c_fc = torch.transpose(self.tt_weight_c_fc, -1, -2)
        #self.tt_weight_c_proj = torch.transpose(self.tt_weight_c_proj, -1, -2)

        # Push weights to Tt device
        self.tt_weight_c_fc = nanogpt_utils.torch2tt_tensor(
            self.tt_weight_c_fc, device
        )
        self.tt_weight_c_proj = nanogpt_utils.torch2tt_tensor(
            self.tt_weight_c_proj, device
        )

        print(self.tt_weight_c_fc.shape())
        print(self.tt_weight_c_proj.shape())
        # Load biases
        self.tt_bias_c_fc = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.c_fc.bias"], device
        )
        self.tt_bias_c_proj = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.c_proj.bias"], device
        )




    def forward(self, x, device):

        x1 = nanogpt_utils.tt_linear(x, self.tt_weight_c_fc, self.tt_bias_c_fc)

        print(x1.shape())

        x2 = tt_lib.tensor.gelu(x1)

        x3 = nanogpt_utils.tt_linear(x2, self.tt_weight_c_proj, self.tt_bias_c_proj)

        return x3
