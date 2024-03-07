# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import tt_lib as ttl

from models.utility_functions import torch2tt_tensor
from models.helper_funcs import Linear
from models.demos.mamba.reference.args import ModelArgs


class TtMambaSSM(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, state_dict, num_users, hidden_size, configs):
        super().__init__()
        self.state_dict = state_dict
        self.device = device
        self.args = args

        # hidden state
        self.num_users = num_users
        self.hidden_size = hidden_size
        self.configs = configs
        self.tt_hidden_state = ttnn.zeros(
            (1, 1, self.num_users, self.hidden_size * self.args.d_state),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        """
        We need to split up the x_proj weights because in the reference
        implementation they perform the linear operation for dt, B, and C in a
        single step. Here we can't do that because it would involve fallback op
        slicing, so we break up the weights ahead of time and do the linear ops
        separately.
        """

        # delta rank weight
        if self.hidden_size == self.args.d_inner:
            x_proj_weight_name = "mixer.x_proj.weight"
            self.delta_t_proj = ttnn.from_torch(
                self.state_dict[x_proj_weight_name][: self.args.dt_rank, :],
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        else:
            self.delta_t_proj = ttnn.from_torch(
                torch.rand(1, 1, self.hidden_size, self.hidden_size // 16),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )

        # delta full weight
        if self.hidden_size == self.args.d_inner:
            dt_proj_weight_name = "mixer.dt_proj.weight"
            dt_proj_bias_name = "mixer.dt_proj.bias"
            self.dt_proj_weights = ttnn.from_torch(
                self.state_dict[dt_proj_weight_name],
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            self.dt_proj_bias = ttnn.from_torch(
                self.state_dict[dt_proj_bias_name],
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        else:
            self.dt_proj_weights = ttnn.from_torch(
                torch.rand(1, 1, self.hidden_size // 16, self.hidden_size),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            self.dt_proj_bias = ttnn.from_torch(
                torch.rand(1, 1, 1, self.hidden_size),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )

        # B
        if self.hidden_size == self.args.d_inner:
            self.B_proj_weights = ttnn.from_torch(
                self.state_dict[x_proj_weight_name][self.args.dt_rank : (self.args.dt_rank + self.args.d_state), :],
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        else:
            self.B_proj_weights = ttnn.from_torch(
                torch.rand(
                    1,
                    1,
                    self.hidden_size,
                    self.args.d_state,
                ),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )

        # zeros for broadcasting B
        self.B_zeros = ttnn.zeros(
            (1, 1, self.num_users, self.hidden_size * self.args.d_state),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=self.device,
        )

    def forward(self, x):
        # delta
        delta_t_proj = ttnn.to_memory_config(self.delta_t_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        delta_t = ttnn.linear(x, delta_t_proj, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(delta_t_proj)

        dt_proj_weights = ttnn.to_memory_config(self.dt_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        delta_t = ttnn.linear(
            delta_t, self.dt_proj_weights, bias=self.dt_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(dt_proj_weights)
        delta_t = ttnn.softplus(delta_t, memory_config=ttnn.L1_MEMORY_CONFIG)
        delta_t = ttnn.to_memory_config(delta_t, memory_config=self.configs["sharded"])

        # B
        B_proj_weights = ttnn.to_memory_config(self.B_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        B = ttnn.linear(x, B_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(B_proj_weights)
        B_zeros = ttnn.to_memory_config(self.B_zeros, memory_config=ttnn.L1_MEMORY_CONFIG)
        B = ttnn.Tensor(
            ttl.tensor.bcast(B_zeros.value, B.value, math_op=ttl.tensor.BcastOpMath.ADD, dim=ttl.tensor.BcastOpDim.W)
        )
        print(f"***************shape: {B_zeros.shape}, {B.shape}, {self.args.d_inner}, {delta_t.shape}***************")
        # B = ttnn.to_memory_config(B, self.configs['sharded_large'])
        ttnn.deallocate(B_zeros)

        bbar = ttnn.Tensor(
            ttl.tensor.bcast(B.value, delta_t.value, math_op=ttl.tensor.BcastOpMath.MUL, dim=ttl.tensor.BcastOpDim.W)
        )
        bbar = ttnn.to_memory_config(bbar, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(B)

        # allocate abar
        abar = torch.rand((1, 1, self.num_users, self.hidden_size * self.args.d_state), dtype=torch.bfloat16)
        abar = ttnn.from_torch(
            abar, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=self.configs["sharded_large"]
        )

        # multiply abar and hidden_state
        hidden_state = ttnn.to_memory_config(self.tt_hidden_state, self.configs["sharded_large"])
        amulh = ttnn.mul(abar, hidden_state, memory_config=self.configs["sharded_large"])

        # deallocate abar and hidden_state
        ttnn.deallocate(abar)
        ttnn.deallocate(hidden_state)

        # multiply bbar and x
        bmulx = ttnn.mul(bbar, x, memory_config=self.configs["sharded_large"])

        # deallocate bbar and x
        ttnn.deallocate(bbar)
        ttnn.deallocate(x)

        self.tt_hidden_state = ttnn.add(amulh, bmulx, memory_config=self.configs["sharded_large"])
        return self.tt_hidden_state
