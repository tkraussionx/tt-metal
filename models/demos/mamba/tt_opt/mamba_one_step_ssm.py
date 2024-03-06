# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib as ttl
import ttnn

from models.utility_functions import torch2tt_tensor
from models.helper_funcs import Linear
from models.demos.mamba.reference.args import ModelArgs


class TtMambaSSM(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device,
        state_dict,
        num_users,
        hidden_size,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args

        """
        We need to split up the x_proj weights because in the reference
        implementation they perform the linear operation for dt, B, and C in a
        single step. Here we can't do that because it would involve fallback op
        slicing, so we break up the weights ahead of time and do the linear ops
        separately.
        """
        self.num_users = num_users
        self.hidden_size = hidden_size
        self.tt_hidden_state = ttnn.zeros((1,1,self.num_users,self.hidden_size), layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
    def forward(self, x):

        abar = torch.rand((1,1,self.num_users,self.hidden_size), dtype=torch.bfloat16)
        cfg = ttnn.create_sharded_memory_config(shape=(1,1,self.num_users,self.hidden_size), core_grid=ttnn.CoreGrid(y=self.num_users//32, x=8), strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=False)

        abar = ttnn.from_torch(abar, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=cfg)
        hidden_state = ttnn.to_memory_config(self.tt_hidden_state, cfg)
        self.tt_hidden_state = ttnn.mul(abar, hidden_state, memory_config=cfg)

        self.output = self.tt_hidden_state
        return self.output
