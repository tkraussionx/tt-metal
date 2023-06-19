import torch
from torch.nn import functional as F
import torch.nn as nn
import tt_lib
import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils
from dataclasses import dataclass
import math
from tt_lib.fallback_ops import fallback_ops

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

class TtCausalSelfAttention(nn.Module):

    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Get the weights
        self.tt_weight_c_attn = state_dict[f"{base_address}.c_attn.weight"]
        self.tt_weight_c_proj = state_dict[f"{base_address}.c_proj.weight"]

        # Push weights to Ttp device
        self.tt_weight_c_attn = nanogpt_utils.torch2tt_tensor(
            self.tt_weight_c_attn, device
        )
        self.tt_weight_c_proj = nanogpt_utils.torch2tt_tensor(
            self.tt_weight_c_proj, device
        )

        # Load biases
        self.tt_bias_c_attn = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.c_attn.bias"], device
        )
        self.tt_bias_c_proj = nanogpt_utils.torch2tt_tensor(
            state_dict[f"{base_address}.c_proj.bias"], device
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x, device):

        _, B, T, C = x.shape() # batch size, sequence length, embedding dimensionality (n_embd)

        x1 = nanogpt_utils.tt_linear(x, self.tt_weight_c_attn, self.tt_bias_c_attn)
        pt_x1 = nanogpt_utils.tt2torch_tensor(x1)
        pt_x1 = pt_x1.squeeze(0)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = pt_x1.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        print('ATT----------')
        print(att.shape)
        tt_att = nanogpt_utils.torch2tt_tensor(att, device)

        tt_att = fallback_ops.softmax(tt_att, dim=-1)

        att = nanogpt_utils.tt2torch_tensor(tt_att)
        att = self.attn_dropout(att)

        tt_att = nanogpt_utils.torch2tt_tensor(att, device)
        tt_v = nanogpt_utils.torch2tt_tensor(v, device)

        #y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        tt_y = nanogpt_utils.tt_matmul(tt_att, tt_v, device)

        #y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        tt_y = tt_lib.tensor.transpose_hc(tt_y)
        tt_y = fallback_ops.reshape(tt_y, 1, B, T, C)


        # output projection
        x2 = nanogpt_utils.tt_linear(tt_y, self.tt_weight_c_proj, self.tt_bias_c_proj)
        pt_x2 = nanogpt_utils.tt2torch_tensor(x2)

        y = self.resid_dropout(pt_x2)
        y = nanogpt_utils.torch2tt_tensor(y, device)
        return y
