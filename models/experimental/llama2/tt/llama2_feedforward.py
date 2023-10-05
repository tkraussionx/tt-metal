# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import Optional
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm

from models.helper_funcs import Linear as TtLinear


class TtFeedForward(nn.Module):
    def __init__(
        self,
        config,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
        state_dict=None,
        base_address="",
        device=None,
    ):
        super().__init__()
        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1_weight = torch_to_tt_tensor_rm(state_dict[f"{self.base_address}.feed_forward.w1.weight"], self.device)
        self.w1 = TtLinear(
            self.w1_weight.shape()[-1],
            self.w1_weight.shape()[-2],
            self.w1_weight,
            None,
        )

        self.w2_weight = torch_to_tt_tensor_rm(state_dict[f"{self.base_address}.feed_forward.w2.weight"], self.device)
        self.w2 = TtLinear(
            self.w2_weight.shape()[-1],
            self.w2_weight.shape()[-2],
            self.w2_weight,
            None,
        )

        self.w3_weight = torch_to_tt_tensor_rm(state_dict[f"{self.base_address}.feed_forward.w3.weight"], self.device)
        self.w3 = TtLinear(
            self.w3_weight.shape()[-1],
            self.w3_weight.shape()[-2],
            self.w3_weight,
            None,
        )

    def forward(self, input: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        linear_1 = self.w1(input)
        linear_2 = self.w3(input)

        linear = tt_lib.tensor.mul(linear_1, linear_2)
        act = tt_lib.tensor.silu(linear)

        return self.w2(act)
