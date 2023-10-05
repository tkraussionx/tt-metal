# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch.nn as nn
from models.experimental.llama2.tt.llama2_attention import TtAttention
from models.experimental.llama2.tt.llama2_feedforward import TtFeedForward
from models.experimental.llama2.tt.llama2_rmsnorm import TtRMSNorm
import torch
import tt_lib


class TtTransformerBlock(nn.Module):
    def __init__(self, config, layer_id: int, state_dict=None, base_address="", device=None):
        super().__init__()

        self.config = config
        self.base_address = base_address
        self.state_dict = state_dict
        self.device = device

        self.n_heads = self.config.n_heads
        self.dim = self.config.dim
        self.head_dim = self.config.dim // self.config.n_heads
        self.attention = TtAttention(self.config, self.state_dict, self.base_address, self.device)
        self.feed_forward = TtFeedForward(
            self.config,
            self.config.dim,
            4 * self.config.dim,
            self.config.multiple_of,
            self.config.ffn_dim_multiplier,
            self.state_dict,
            self.base_address,
            self.device,
        )
        self.layer_id = layer_id

        self.attention_norm = TtRMSNorm(
            self.config,
            self.config.dim,
            self.config.norm_eps,
            state_dict,
            f"{self.base_address}.attention_norm",
            self.device,
        )
        self.ffn_norm = TtRMSNorm(
            self.config,
            self.config.dim,
            self.config.norm_eps,
            self.state_dict,
            f"{self.base_address}.ffn_norm",
            self.device,
        )

    def forward(
        self,
        x: tt_lib.tensor.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[tt_lib.tensor.Tensor],
    ) -> tt_lib.tensor.Tensor:
        attn_norm = self.attention_norm(x)
        attn = self.attention(attn_norm, start_pos, freqs_cis, mask)
        h = tt_lib.tensor.add(x, attn)

        out = self.feed_forward(self.ffn_norm(h))
        out = tt_lib.tensor.add(h, out)

        return out
