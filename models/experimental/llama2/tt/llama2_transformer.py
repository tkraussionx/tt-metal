# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List
import torch.nn as nn
from models.experimental.llama2.tt.llama2_transformer_block import TtTransformerBlock
from models.experimental.llama2.tt.llama2_rmsnorm import TtRMSNorm

from models.helper_funcs import Linear as TtLinear

import tt_lib
import torch

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtTransformer(nn.Module):
    def __init__(self, config, start_layer: int, end_layer: int, state_dict=None, base_address="", device=None) -> None:
        super().__init__()
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        if start_layer == 0:
            self.tok_embeddings_weight = state_dict["tok_embeddings.weight"]
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim, _weight=self.tok_embeddings_weight)
            self.freqs_cis = self.precompute_freqs_cis(
                self.config.dim // self.config.n_heads,
                self.config.max_seq_len * 2,
            )

        self.layers = nn.ModuleList()
        for layer_id in range(start_layer, end_layer):
            self.layers.append(
                TtTransformerBlock(config, layer_id, self.state_dict, f"{self.base_address}.{layer_id}", self.device)
            )

        if end_layer == config.n_layers:
            self.norm = TtRMSNorm(
                config,
                dim=config.dim,
                eps=config.norm_eps,
                state_dict=self.state_dict,
                base_address="norm",
                device=self.device,
            )

            self.output_weight = torch_to_tt_tensor_rm(state_dict["output.weight"], self.device)
            self.output = TtLinear(
                self.output_weight.shape()[-1], self.output_weight.shape()[-2], self.output_weight, None
            )

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def const_tensor(self, shape: List[int], value: float) -> tt_lib.tensor.Tensor:
        return tt_lib.tensor.full(shape, value)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        start_idx: int,
        end_idx: int,
        freqs_cis: Optional[tt_lib.tensor.Tensor] = None,
        mask: Optional[tt_lib.tensor.Tensor] = None,
    ):
        if start_idx == 0:
            _bsz, seqlen = tokens.shape
            tokens = self.tok_embeddings(tokens)
            tokens = torch_to_tt_tensor_rm(tokens, self.device, put_on_device=False)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
            if seqlen > 1:
                mask = self.const_tensor([1, 1, seqlen, seqlen], float("-inf"))
                mask = tt_to_torch_tensor(mask)
                mask = torch.triu(mask, diagonal=start_pos + 1)

                mask = torch_to_tt_tensor_rm(mask, self.device, put_on_device=False)

        for index, module in enumerate(self.layers):
            tokens = module(tokens, start_pos, freqs_cis, mask)

        if end_idx != self.config.n_layers:
            return tokens, freqs_cis, mask
        tokens = self.norm(tokens)
        output = self.output(tokens)
        return output, freqs_cis, mask
