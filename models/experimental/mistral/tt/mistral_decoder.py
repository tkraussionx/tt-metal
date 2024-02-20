# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import tt_lib
from typing import Optional
from models.experimental.mistral.tt.mistral_attention import TtMistralAttention
from models.experimental.mistral.tt.mistral_mlp import TtMistralMLP
from models.experimental.mistral.tt.mistral_rms_norm import TtRMSNorm

from models.utility_functions import (
    torch2tt_tensor,
    nearest_32,
)

from models.experimental.mistral.tt.mistral_common import (
    precompute_freqs as tt_precompute_freqs,
    freqs_to_rotation_matrix,
    gather_rotary_emb as tt_gather_rotary_emb,
    tt_all_reduce,
)


class TtTransformerBlock(nn.Module):
    def __init__(
        self,
        args=None,
        devices=None,
        state_dict=None,
        base_address=None,
        layer_num=None,
        model_config=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.sliding_window = args.sliding_window

        self.layer_num = layer_num
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.model_config = model_config

        self.oldest = 0

        self.attention = TtMistralAttention(
            devices=devices,
            state_dict=state_dict,
            base_url=f"{base_address}attention.",
            layer_num=layer_num,  # TODO double check the logic for layer_num when scaling for all layers
            model_config=model_config,
            configuration=args,
        )
        self.feed_forward = TtMistralMLP(
            device=devices[0],  # TODO Should we update MLP code to support multiple devices when scaling up?
            state_dict=state_dict,
            base_address=f"{base_address}feed_forward.",
            model_config=model_config,
        )
        self.attention_norm = TtRMSNorm(
            device=devices[0],
            base_address=f"{base_address}attention_norm.",
            state_dict=state_dict,
            model_config=model_config,
        )
        self.ffn_norm = TtRMSNorm(
            device=devices[0],
            base_address=f"{base_address}ffn_norm.",
            state_dict=state_dict,
            model_config=model_config,
        )

    # TODO function taken from mistral_attention.py. Move to mistral_common.py
    def get_rotation_mat(self, cos, sin, start_pos, seqlen, batch):
        # cos, sin = tt_precompute_freqs(dhead, end)
        rot_mat = freqs_to_rotation_matrix(cos, sin)
        position_ids = torch.ones(batch, seqlen, dtype=torch.long) * start_pos
        rot_emb = tt_gather_rotary_emb(rot_mat, position_ids)
        return rot_emb

    # TODO function taken from mistral_attention.py. Move to mistral_common.py
    def prepare_inputs(self, x, start_pos, cos, sin):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        x: (batch, seq, hidden_dim)
        start_pos: int
        """
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
        rot_mat = self.get_rotation_mat(cos, sin, start_pos=start_pos, seqlen=seq_len, batch=batch)

        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)
        self.current = self.current % self.sliding_window
        attn_mask = torch.zeros(seq_len, 1, batch, padded_layer_past_len)

        if start_pos < self.sliding_window:
            attn_mask[:, :, :, self.current + 1 :] = torch.finfo(attn_mask.dtype).min
        else:
            attn_mask[:, :, :, : self.current] = torch.finfo(attn_mask.dtype).min
            attn_mask[:, :, :, self.sliding_window - self.current :] = torch.finfo(attn_mask.dtype).min
        attn_mask = attn_mask.expand(-1, self.n_local_heads, -1, -1)

        # TODO: mask out >sliding_window prev tokens

        # expected shapes:
        # x: (seq_len, 1, batch, hidden_dim)
        # start_pos: int
        # rot_mat: [1, bsz, head_dim, head_dim]
        # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
        assert x.size() == (seq_len, 1, batch, self.hidden_size)
        assert rot_mat.size() == (1, batch, self.head_dim, self.head_dim)
        assert attn_mask.size() == (seq_len, self.n_local_heads, batch, padded_layer_past_len)

        xs, rot_mats, attn_masks = [], [], []
        for i in range(self.num_devices):
            device = self.devices[i]
            xs.append(torch2tt_tensor(x.clone(), device))
            rot_mats.append(torch2tt_tensor(rot_mat.clone(), device))
            attn_masks.append(torch2tt_tensor(attn_mask.clone(), device))

        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def forward(
        self,
        # x: tt_lib.tensor.Tensor,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_masks: Optional[tt_lib.tensor.Tensor],
        # bcast_freq_xq: tt_lib.tensor.complex_tensor,
        # bcast_freq_xk: tt_lib.tensor.complex_tensor,
        # positions: tt_lib.tensor.Tensor,
        # mask: Optional[torch.Tensor],
        # seqlen: int,
    ) -> tt_lib.tensor.Tensor:
        # TODO We're passign a list of inputs + rot_mat + start_pos + attn mask (for each device)
        attn_norm = [self.attention_norm(xs[0])]
        r = self.attention.forward(
            attn_norm,
            rot_mats,
            start_pos,
            attn_masks,
        )
        # Attn takes a list of inputs (assuming multiple devices) and returns multiple outputs
        h = tt_lib.tensor.add(xs[0], r[0])
        xs[0].deallocate()
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = tt_lib.tensor.add(h, r)
        h.deallocate()
        return out
