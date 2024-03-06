# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.tt_opt.residual_block import TtResidualBlock

class MambaTT(torch.nn.Module):
    def __init__(
        self,
        reference_model,
        num_layers,
        device,
        num_users,
        hidden_size,
        
    ):
        super().__init__()
        print(f"Initalizing MambaTT with {num_layers} layers")
        self.args = reference_model.args
        self.device = device
        self.layers = [TtResidualBlock(self.args, device, reference_model.layers[i].state_dict(), num_users, hidden_size) for i in range(num_layers)]
        

    def forward(self, x):
       for layer in self.layers:
            x = layer(x)

       return x