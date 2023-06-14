import torch
from torch.nn import functional as F
import torch.nn as nn
import tt_lib
import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils
import python_api_testing.models.nanogpt.nanogpt_mlp as nanogpt_mlp
import python_api_testing.models.nanogpt.nanogpt_attention as nanogpt_attention

from tt_lib.fallback_ops import fallback_ops

from dataclasses import dataclass
import math

from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class TtBlock(nn.Module):

    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.beta_1 = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.ln_1.bias"], device
        )
        self.gamma_1 = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.ln_1.weight"], device
        )

        self.ln_1 = fallback_ops.LayerNorm(
            biases = self.gamma_1,
            weights = self.beta_1,
            normalized_shape = config.n_embd
        )

        base_address_attn = f"{base_address}.attn"

        self.attn = nanogpt_attention.TtCausalSelfAttention(config, state_dict, base_address_attn, device)

        self.beta_2 = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.ln_2.bias"], device
        )
        self.gamma_2= nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.ln_2.weight"], device
        )

        self.ln_2 = fallback_ops.LayerNorm(
            biases = self.gamma_2,
            weights = self.beta_2,
            normalized_shape = config.n_embd
        )

        base_address_mlp = f"{base_address}.mlp"

        self.mlp = nanogpt_mlp.TtMLP(base_address_mlp, state_dict, device)


    def forward(self, x, device):
        y = self.ln_1(x)
        res1 = self.ln_1(x)
        res2 = self.attn.forward(res1, device)

        x = tt_lib.tensor.add(x, res2)

        y = self.ln_2(x)
        res3 = self.ln_2(x)
        res4 = self.attn.forward(res3, device)
        x = tt_lib.tensor.add(x, res4)

        return x
