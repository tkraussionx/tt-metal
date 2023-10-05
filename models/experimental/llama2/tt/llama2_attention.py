# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import Tuple, Optional

from models.helper_funcs import Linear as TtLinear
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, torch_to_tt_tensor
import tt_lib
import math
import torch


class TtAttention(nn.Module):
    def __init__(self, config, state_dict=None, base_address="", device=None):
        super().__init__()
        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device
        self.n_kv_heads = self.config.n_heads if self.config.n_kv_heads is None else self.config.n_kv_heads
        self.n_local_heads = self.config.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = self.config.dim // self.config.n_heads

        self.wq_weight = torch_to_tt_tensor_rm(state_dict[f"{self.base_address}.attention.wq.weight"], self.device)
        self.wq = TtLinear(
            self.wq_weight.shape()[-1],
            self.wq_weight.shape()[-2],
            self.wq_weight,
            None,
        )

        self.wk_weight = torch_to_tt_tensor_rm(state_dict[f"{self.base_address}.attention.wk.weight"], self.device)
        self.wk = TtLinear(
            self.wk_weight.shape()[-1],
            self.wk_weight.shape()[-2],
            self.wk_weight,
            None,
        )

        self.wv_weight = torch_to_tt_tensor_rm(state_dict[f"{self.base_address}.attention.wv.weight"], self.device)
        self.wv = TtLinear(
            self.wv_weight.shape()[-1],
            self.wv_weight.shape()[-2],
            self.wv_weight,
            None,
        )

        self.wo_weight = torch_to_tt_tensor_rm(state_dict[f"{self.base_address}.attention.wo.weight"], self.device)
        self.wo = TtLinear(
            self.wo_weight.shape()[-1],
            self.wo_weight.shape()[-2],
            self.wo_weight,
            None,
        )

        self.cache_k = tt_lib.tensor.zeros(
            [
                self.config.max_batch_size,
                self.config.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ]
        )
        self.cache_v = tt_lib.tensor.zeros(
            [
                self.config.max_batch_size,
                self.config.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ]
        )

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = len(x.shape)
        assert 0 <= 1 <= ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_embedding(
        self, xq: tt_lib.tensor.Tensor, xk: tt_lib.tensor.Tensor, freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq = tt_to_torch_tensor(xq)
        xk = tt_to_torch_tensor(xk)

        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x

        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(
        self,
        input: tt_lib.tensor.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,  # complex64
        mask: Optional[tt_lib.tensor.Tensor],
    ) -> tt_lib.tensor.Tensor:
        _, bsz, seqlen, _ = input.shape()
        xq = self.wq(input)
        xk = self.wk(input)
        xv = self.wv(input)

        xq = fallback_ops.reshape(xq, bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = fallback_ops.reshape(xk, bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = fallback_ops.reshape(xv, bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = self.apply_rotary_embedding(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = tt_to_torch_tensor(self.cache_k)
        self.cache_v = tt_to_torch_tensor(self.cache_v)

        xv = tt_to_torch_tensor(xv)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = self.repeat_kv(keys, self.n_rep)
        values = self.repeat_kv(values, self.n_rep)

        keys = torch_to_tt_tensor(keys, self.device)
        values = torch_to_tt_tensor(values, self.device)

        xq = torch_to_tt_tensor(xq, self.device)

        xq = tt_lib.tensor.transpose(xq, 1, 2)
        keys = tt_lib.tensor.transpose(keys, 1, 2)
        values = tt_lib.tensor.transpose(values, 1, 2)

        scores = tt_lib.tensor.bmm(xq, tt_lib.tensor.transpose(keys, 2, 3))
        head_dim_sqrt = math.sqrt(self.head_dim)
        head_dim_sqrt = tt_lib.tensor.full(scores.shape(), head_dim_sqrt)
        head_dim_sqrt = tt_lib.tensor.recip(head_dim_sqrt)
        scores = tt_lib.tensor.mul(scores, head_dim_sqrt)

        if mask is not None:
            scores = tt_lib.tensor.transpose(scores, 1, 3)
            # mask tensor is not in device
            mask = tt_to_torch_tensor(mask)
            mask = torch.transpose(mask, 1, 3)
            mask = torch_to_tt_tensor_rm(mask, self.device, put_on_device=False)
            scores = tt_lib.tensor.bcast(scores, mask, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.W)

            scores = tt_lib.tensor.transpose(scores, 1, 3)

        scores = fallback_ops.softmax(scores, dim=-1)

        output = tt_lib.tensor.bmm(scores, values)
        output = tt_lib.tensor.transpose(output, 1, 2)
        output = fallback_ops.reshape(output, 1, bsz, seqlen, -1)
        return self.wo(output)
