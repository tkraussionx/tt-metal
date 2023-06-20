import torch
from torch.nn import functional as F
import torch.nn as nn
import tt_lib
import python_api_testing.models.nanogpt.nanogpt_utils as nanogpt_utils
import python_api_testing.models.nanogpt.nanogpt_mlp as nanogpt_mlp
import python_api_testing.models.nanogpt.nanogpt_attention as nanogpt_attention

import python_api_testing.models.nanogpt.nanogpt_block as nanogpt_block
from tt_lib.fallback_ops import fallback_ops

import numpy as np

from dataclasses import dataclass
import math

from transformers import GPT2LMHeadModel


def pad_input_tensor(tensor, value, multiple):
    print('inside---')
    tensor = torch.transpose(tensor, 0, 1)
    print(tensor.shape)
    len = tensor.shape[1]

    if len % multiple == 0:
        tensor = torch.transpose(tensor, 0, 1)

        return tensor

    padded_len = ((len // multiple) + 1) * multiple

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)
    print('DONE---')
    print(tensor.shape)

    tensor = torch.transpose(tensor, 0, 1)

    return tensor

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

def pad_2(tensor, value):
    print('Shape')
    print(tensor.shape)
    len = tensor.shape[2]

    if len % 2 == 0:
        return tensor

    padded_len = len + 1

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor

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


        self.wte.weight = torch.nn.Parameter(
            state_dict[f"{base_address}.wte.weight"]
        )

        self.wpe.weight = torch.nn.Parameter(
            state_dict[f"{base_address}.wpe.weight"]
        )

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

        self.tt_weight_lm_head = tt_lib.tensor.transpose(self.tt_weight_lm_head)

        self.wte.weight = nn.Parameter(self.lm_weight) # https://paperswithcode.com/method/weight-tying


    def forward(self, idx, device, mask=None):
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
        print('UNPADDED!!!-----')
        print(x.shape)
        x = x.squeeze(0)
        x = x.squeeze(0)
        a = pad_input_tensor(x, 0, 2)

        print('PADDED______X')
        print(a.shape)

        tt_x = nanogpt_utils.torch2tt_tensor(a, device)

        for block in self.h:
            tt_x = block.forward(tt_x, device, mask)

        tt_x = self.ln_f(tt_x)
        x = nanogpt_utils.tt2torch_tensor(tt_x)
        x = x[:, [-1], :]
        tt_x = nanogpt_utils.torch2tt_tensor(x, device)

        logits = nanogpt_utils.tt_linear(tt_x, weight=self.tt_weight_lm_head, bias=None)
        loss = None

        return logits, loss




    def generate(self, idx, device, max_new_tokens, temperature=1.0, top_k=None, mask=None):
            """
            Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
            the sequence max_new_tokens times, feeding the predictions back into the model each time.
            Most likely you'll want to make sure to be in model.eval() mode of operation for this.
            """
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                # forward the model to get the logits for the index in the sequence


                #mask = pad_2(torch.zeros(idx.shape()))

                tt_logits, _ = self.forward(idx_cond, device, mask)
                # pluck the logits at the final step and scale by desired temperature
                print(tt_logits.shape())
                logits = nanogpt_utils.tt2torch_tensor(tt_logits)
                print('Logits')
                print(logits.shape)
                #logits = logits.squeeze(0)

                #logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                #tt_logits = nanogpt_utils.torch2tt_tensor(logits, device)
            # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, :, -1, :] / temperature
            # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                print('P-SHAPEEE')
                print(probs.shape)
                probs = probs.squeeze(0)
                print(probs)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
                print(idx.shape)
                print(idx_next.shape)
                idx = torch.cat((idx, idx_next), dim=1)

            return idx
