# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.experimental.mamba.reference.args import ModelArgs
from models.experimental.mamba.tt_opt.mamba_one_step_ssm import TtMambaSSM

class TtMambaBlock(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device,
        state_dict,
        num_users,
        hidden_size,
        configs
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args
        self.num_users = num_users
        self.hidden_size = hidden_size
        self.configs = configs
        
        # mlp wt
        self.mlp_proj = ttnn.from_torch(torch.rand(1,1,self.hidden_size,2*self.hidden_size), layout=ttnn.TILE_LAYOUT, device=self.device, 
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


        self.tt_ssm = TtMambaSSM(self.args,self.device,self.state_dict, num_users, hidden_size, configs)

    def forward(self, x):

        x0 = ttnn.linear(x, self.mlp_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x)
        x = self.tt_ssm(x0)
        return x
