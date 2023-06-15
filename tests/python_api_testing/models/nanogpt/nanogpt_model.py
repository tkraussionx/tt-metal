import torch
from torch.nn import functional as F
import torch.nn as nn
import tt_lib
import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils
import python_api_testing.models.nanogpt.nanogpt_mlp as nanogpt_mlp
import python_api_testing.models.nanogpt.nanogpt_attention as nanogpt_attention

import python_api_testing.models.nanogpt.nanogpt_block as nanogpt_block
from tt_lib.fallback_ops import fallback_ops

from dataclasses import dataclass
import math

from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


"""
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
"""
class TtGPT(nn.Module):
    def __init__(self, config, state_dict, device):
        super().__init__()

        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        base_address = f"transformer"

        self.beta = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.ln_f.bias"], device
        )
        self.gamma = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.ln_f.weight"], device
        )

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        blocks = []

        for i in range(config.n_layer):
            block = nanogpt_block.TtBlock(self.config, state_dict, f"{base_address}.h.{i}", device)
            blocks.append(block)

        self.h = torch.nn.ModuleList(blocks)


        self.ln_f = fallback_ops.LayerNorm(
            self.gamma,
            self.beta,
            eps=1e-5,
            normalized_shape=config.n_embd,
        )

        self.lm_weight = state_dict["lm_head.weight"]

        # Push weights to Tt device
        self.tt_weight_lm_head = nanogpt_utils.torch2tt_tensor(
            self.lm_weight, device
        )


    def forward(self, idx, device):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)

        tt_tok_emb = nanogpt_utils.torch2tt_tensor(tok_emb, device)
        tt_pos_emb = nanogpt_utils.torch2tt_tensor(pos_emb, device)

        tt_sum = tt_lib.tensor.add(tt_tok_emb, tt_pos_emb)

        sum = nanogpt_utils.tt2torch_tensor(tt_sum)

        x = self.drop(sum)
        tt_x = nanogpt_utils.torch2tt_tensor(x, device)

        for block in self.h:
            tt_x = block.forward(tt_x, device)

        tt_x = self.ln_f(tt_x)
        x = nanogpt_utils.tt2torch_tensor(tt_x)
        x = x[:, [-1], :]
        tt_x = nanogpt_utils.torch2tt_tensor(x, device)
        print('-------------sgapes')
        print(tt_x.shape())
        print(self.tt_weight_lm_head.shape())

        logits = nanogpt_utils.tt_linear(tt_x, self.tt_weight_lm_head)
        loss = None

        return logits, loss
