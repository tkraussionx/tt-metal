# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import List
from models.demos.mixtral8x7b.tt.mixtral_attention_ttnn import TtMixtralAttention
from models.demos.mixtral8x7b.tt.mixtral_mlp_ttnn import TtMixtralMLP
from models.demos.mixtral8x7b.tt.mixtral_rms_norm_ttnn import TtRMSNorm
from models.demos.mixtral8x7b.tt.mixtral_moe_ttnn_new import TtMoeLayer


class TtTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        args,
        layer_num,
        dtype,
        tt_cos_cached,
        tt_sin_cached,
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
        self.sliding_window = args.sliding_window

        self.layer_num = layer_num
        self.n_local_heads = self.n_heads // len(devices)
        self.n_local_kv_heads = self.n_kv_heads // len(devices)

        self.attention = TtMixtralAttention(
            devices=devices,
            state_dict=state_dict,
            args=args,
            layer_num=layer_num,
            dtype=dtype,
            tt_cos_cached=tt_cos_cached,
            tt_sin_cached=tt_sin_cached,
        )

        self.feed_forward = TtMoeLayer(
            devices=devices,
            state_dict=state_dict,
            experts=[
                TtMixtralMLP(
                    device=devices[i],
                    state_dict=state_dict,
                    args=args,
                    layer_num=layer_num,
                    expert_num=i,
                    dtype=dtype,
                )
                for i in range(args.num_experts)
            ],
            args=args,
            layer_num=layer_num,
            dtype=dtype,
        )
        self.attention_norm = [
            TtRMSNorm(
                device=dev,
                state_dict=state_dict,
                args=args,
                dtype=dtype,
                layer_num=layer_num,
                weight_key="attention_norm",
            )
            for dev in self.devices
        ]
        self.ffn_norm = [
            TtRMSNorm(
                device=dev,
                state_dict=state_dict,
                args=args,
                dtype=dtype,
                layer_num=layer_num,
                weight_key="ffn_norm",
            )
            for dev in self.devices
        ]

    def forward(
        self,
        xs: ttnn.Tensor,
        start_pos: int,
        current_pos: int,
        attn_masks: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
    ) -> ttnn.Tensor:
        assert isinstance(xs, list)

        # Attention module expects a list of inputs, start_pos, attn mask (multi-device support)
        attn_norm = []
        for i in range(self.num_devices):
            attn_norm.append(self.attention_norm[i](xs[i]))

        r = self.attention(
            attn_norm,
            start_pos,
            current_pos,
            attn_masks,
            rot_mats,
        )
        h = []
        # Attention also returns multiple outputs (multi-device support)
        for i in range(self.num_devices):
            r[i] = ttnn.permute(r[i], (2, 1, 0, 3))
            h_i = self.ffn_norm[i](ttnn.experimental.tensor.add(xs[i], r[i]))
            h.append(h_i)
            ttnn.deallocate(xs[i])
            ttnn.deallocate(r[i])
        r = self.feed_forward(h)

        out = []
        for i in range(self.num_devices):
            h[i] = ttnn.permute(h[i], (2, 1, 0, 3))
            out.append(ttnn.experimental.tensor.add(h[i], r[i]))
            ttnn.deallocate(h[i])
            ttnn.deallocate(r[i])
        return out
