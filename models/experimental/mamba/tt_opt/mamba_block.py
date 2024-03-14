# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from typing import Callable

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.experimental.mamba.reference.args import ModelArgs
from models.experimental.mamba.tt_opt.mamba_one_step_ssm import TtMambaSSM

class TtMambaBlock(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device,
        load_fn: Callable,
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
        
        # ssm wt
        if self.args.d_model == self.hidden_size:
            print('**********using ssm proj wts')
            in_proj_weight_name = "mixer.in_proj.weight"
            ssm_proj = torch.transpose(self.state_dict[in_proj_weight_name][: self.args.d_inner, :], -1, -2)
            self.ssm_proj = ttnn.from_torch(ssm_proj, layout=ttnn.TILE_LAYOUT, device=self.device, 
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        else:
            self.ssm_proj = ttnn.from_torch(torch.rand(1,1,self.hidden_size,2*self.hidden_size), layout=ttnn.TILE_LAYOUT, device=self.device, 
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        
        # mlp wt
        self.mlp_proj = ttnn.from_torch(torch.rand(1,1,self.hidden_size,2*self.hidden_size), layout=ttnn.TILE_LAYOUT, device=self.device, 
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        
        # down proj wt
        self.down_proj = ttnn.from_torch(torch.rand(1,1,self.hidden_size*2,self.hidden_size), layout=ttnn.TILE_LAYOUT, device=self.device, 
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        
        # conv states
        self.conv_states = []
        for i in range(4):
            self.conv_states.append(ttnn.zeros((1,1,self.num_users,self.hidden_size*2), layout=ttnn.TILE_LAYOUT, device=self.device, 
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16))
        self.conv_wts = [] 
        for i in range(4):
            self.conv_wts.append(ttnn.from_torch(torch.rand(1,1,1,self.hidden_size*2,), layout=ttnn.TILE_LAYOUT, device=self.device, 
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16))
        self.conv_bias = ttnn.from_torch(torch.rand(1,1,1,self.hidden_size*2), layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


        self.tt_ssm = TtMambaSSM(self.args,self.device,load_fn,self.state_dict, num_users, hidden_size, configs)

    def forward(self, x):
        x_input = x # b, e=d_model
        x = ttnn.linear(x, self.ssm_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
     
        # left shift conv states
        ttnn.deallocate(self.conv_states[0])
        for i in range(3):
            self.conv_states[i] = self.conv_states[i + 1]
        self.conv_states[3] = x
        
        # do the convolution
        conv_wts = ttnn.repeat_interleave(self.conv_wts[0], self.num_users, dim=2)
        x = ttnn.mul(conv_wts, self.conv_states[0])        
        for i in range(1,4):
            print('**********', self.conv_wts[i].shape, self.conv_states[i].shape)
            conv_wts = ttnn.repeat_interleave(self.conv_wts[i], self.num_users, dim=2)
            prod = ttnn.mul(conv_wts, self.conv_states[i])
            x = ttnn.add(x, prod)
        conv_bias = ttnn.repeat_interleave(self.conv_bias, self.num_users, dim=2)
        x = ttnn.add(x, conv_bias)
        
        x = ttnn.silu(x)
        print('**********', x.shape)
        x = self.tt_ssm(x)
        res = ttnn.linear(x_input, self.mlp_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.mul(x, res)
        ttnn.deallocate(res)
        
        x = ttnn.linear(x, self.down_proj, memory_config=ttnn.L1_MEMORY_CONFIG)

        return x
