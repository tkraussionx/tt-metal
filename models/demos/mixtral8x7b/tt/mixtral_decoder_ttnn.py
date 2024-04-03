# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import List
from models.demos.mixtral8x7b.tt.mixtral_attention_ttnn import TtMixtralAttention
from models.demos.mixtral8x7b.tt.mixtral_mlp_ttnn import TtMixtralMLP
from models.demos.mixtral8x7b.tt.mixtral_rms_norm_ttnn import TtRMSNorm
from models.demos.mixtral8x7b.tt.mixtral_moe_ttnn import TtMoeLayer


class TtTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        args,
        layer_num,
        dtype,
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

        self.comps = []

    def forward(
        self,
        xs_b1sh: ttnn.Tensor,
        start_pos: int,
        current_pos: int,
        rot_mats: List[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """
        b: batch dim
        s: seq dim
        1: unary dim
        h: hidden dim
        """
        assert isinstance(xs_b1sh, list)
        deallocate = lambda ls: [ttnn.deallocate(l) for l in ls]

        # Attention module expects a list of inputs, start_pos, attn mask (multi-device support)
        attn_norm_s1bh = [self.attention_norm[i](xs_b1sh[i]) for i in range(self.num_devices)]
        # self.comps.append(ttnn.to_torch(attn_norm_s1bh[0]))
        attn_b1sh = self.attention(
            attn_norm_s1bh,
            start_pos,
            current_pos,
            rot_mats,
        )
        # self.comps.append(ttnn.to_torch(attn_b1sh[0]))
        hs_b1sh = [ttnn.add(xs_b1sh[i], attn_b1sh[i]) for i in range(self.num_devices)]
        # self.comps.append(ttnn.to_torch(hs_b1sh[0]))
        # deallocate(attn_b1sh)

        ffn_norm_s1bh = [self.ffn_norm[i](hs_b1sh[i]) for i in range(self.num_devices)]
        # self.comps.append(ttnn.to_torch(ffn_norm_s1bh[0]))
        ffn_b1sh = self.feed_forward(ffn_norm_s1bh)
        # self.comps.append(ttnn.to_torch(ffn_b1sh[0]))
        out_b1sh = [ttnn.add(hs_b1sh[i], ffn_b1sh[i]) for i in range(self.num_devices)]
        # self.comps.append(ttnn.to_torch(out_b1sh[0]))
        # deallocate(ffn_b1sh)
        # deallocate(hs_b1sh)
        # deallocate(attn_b1sh)
        return out_b1sh
