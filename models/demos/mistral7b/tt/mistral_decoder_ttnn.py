# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import Optional
from models.demos.mistral7b.tt.mistral_attention_ttnn import TtMistralAttention
from models.demos.mistral7b.tt.mistral_mlp_ttnn import TtMistralMLP
from models.demos.mistral7b.tt.mistral_rms_norm_ttnn import TtRMSNorm


class TtTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        args,
        device,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        tt_cos_cached,
        tt_sin_cached,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.num_devices = 1

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

        self.attention = TtMistralAttention(
            devices=[device],
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            configuration=args,
            tt_cos_cached=tt_cos_cached,
            tt_sin_cached=tt_sin_cached,
        )
        self.feed_forward = TtMistralMLP(
            device=device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
        )
        self.attention_norm = TtRMSNorm(
            device=device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            weight_key="attention_norm",
        )
        self.ffn_norm = TtRMSNorm(
            device=device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            weight_key="ffn_norm",
        )

    def forward(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        current_pos: int,
        attn_masks: Optional[ttnn.Tensor],
        rot_mat: ttnn.Tensor,
    ) -> ttnn.Tensor:
        attn_norm = self.attention_norm(x)
        # Attention module expects a list of inputs, attn masks, rot_mat (multi-device support)
        r = self.attention.forward(
            [attn_norm],
            start_pos,
            current_pos,
            [attn_masks],
            [rot_mat],
        )
        # Attention also returns multiple outputs (multi-device support)
        assert len(r) == 1, "Multiple devices not yet supported"
        r = r[0]
        r = ttnn.reshape(r, (1, 1, 32, 4096))
        h = ttnn.experimental.tensor.add(x, r)
        ttnn.deallocate(x)
        ttnn.deallocate(r)
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = ttnn.experimental.tensor.add(h, r)
        ttnn.deallocate(h)
        return out
