# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import tt_lib
from models.helper_funcs import Linear

import models.experimental.nanogpt.tt.nanogpt_block as nanogpt_block
from models.experimental.nanogpt.nanogpt_helper_funcs import format_tensor, unpad_from_zero

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


class TtGPT(nn.Module):
    def __init__(self, config, device, tt_cache_path, dtype):
        super().__init__()

        assert config.vocab_size is not None

        self.config = config
        self.config.block_size = 1024
        base_address = f"transformer"
        self.device = device
        self.output_mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        )

        self.beta = tt_lib.tensor.load_tensor(tt_cache_path + base_address + ".ln_f.bias" + str(dtype) + ".bin")

        self.gamma = tt_lib.tensor.load_tensor(tt_cache_path + base_address + ".ln_f.weight" + str(dtype) + ".bin")

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(self.config.block_size, config.n_embd)

        self.wte.weight = torch.nn.Parameter(torch.load(tt_cache_path + "transformer.wte.weight.pt"))

        self.wpe.weight = torch.nn.Parameter(torch.load(tt_cache_path + "transformer.wpe.weight.pt"))

        blocks = []

        for i in range(config.n_layer):
            block = nanogpt_block.TtBlock(self.config, f"{base_address}.h.{i}", self.device, tt_cache_path, dtype)
            blocks.append(block)

        self.h = nn.ModuleList(blocks)

        self.ln_f = tt_lib.tensor.layernorm

        tt_lm_weight = tt_lib.tensor.load_tensor(tt_cache_path + "lm_head.weight" + str(dtype) + ".bin")

        weight = unpad_from_zero(tt_lm_weight, (1, 1, self.config.vocab_size, self.config.n_embd))
        weight_torch = weight
        weight = torch_to_tt_tensor_rm(weight, device=self.device)

        desired_lm_head_vocab_shape = tt_lm_weight.shape()[-2]
        self.lm_head = Linear(self.config.n_embd, desired_lm_head_vocab_shape, tt_lm_weight)

        self.wte.weight = nn.Parameter(weight_torch.squeeze())  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx: torch.Tensor) -> tt_lib.tensor.Tensor:
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = tt_lib.tensor.arange(0, t, 1)
        pos = tt_to_torch_tensor(pos)
        pos = pos.squeeze(0).squeeze(0)
        pos = pos.to(dtype=torch.int64)

        # forward the GPT model itself
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)

        tt_tok_emb = torch_to_tt_tensor_rm(tok_emb, self.device)
        tt_pos_emb = torch_to_tt_tensor_rm(pos_emb, self.device)

        desired_tt_x_shape = tt_tok_emb.shape()
        tt_tok_emb = format_tensor(tt_tok_emb, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        tt_pos_emb = format_tensor(tt_pos_emb, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)

        tt_tok_emb = tt_lib.tensor.permute(tt_tok_emb, (0, 2, 1, 3))
        tt_pos_emb = tt_lib.tensor.permute(tt_pos_emb, (0, 2, 1, 3))

        tt_x = tt_lib.tensor.bcast(tt_tok_emb, tt_pos_emb, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H)
        tt_tok_emb.deallocate()
        tt_pos_emb.deallocate()

        tt_x = tt_lib.tensor.permute(tt_x, (0, 2, 1, 3))

        for block in self.h:
            tt_x = block.forward(tt_x)

        tt_x = format_tensor(tt_x, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        tt_x = self.ln_f(tt_x, eps=1e-5, gamma=self.gamma, beta=self.beta)

        logits = self.lm_head(tt_x)

        desired_logits_shape = desired_tt_x_shape.copy()
        desired_logits_shape[-1] = self.config.vocab_size

        logits = unpad_from_zero(logits, desired_logits_shape)
        logits = torch_to_tt_tensor_rm(logits, self.device, put_on_device=False)

        return logits
