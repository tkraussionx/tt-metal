# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import Optional
from models.demos.mixtral8x7b.tt.mixtral_attention_ttnn import TtMixtralAttention
from models.demos.mixtral8x7b.tt.mixtral_mlp_ttnn import TtMixtralMLP
from models.demos.mixtral8x7b.tt.mixtral_rms_norm_ttnn import TtRMSNorm
from models.demos.mixtral8x7b.tt.mixtral_moe_ttnn import TtMoeLayer


class TtTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        args=None,
        devices=None,
        dtype=None,
        state_dict=None,
        layer_num=None,
        model_config=None,
        tt_cos_cached=None,
        tt_sin_cached=None,
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

        self.attention = TtMixtralAttention(
            devices=devices,
            state_dict=state_dict,
            model_config=model_config,
            layer_num=layer_num,  # TODO double check the logic for layer_num when scaling for all layers
            dtype=dtype,
            configuration=args,
            tt_cos_cached=tt_cos_cached,
            tt_sin_cached=tt_sin_cached,
        )

        self.feed_forward = TtMoeLayer(
            experts=[
                TtMixtralMLP(
                    device=devices[i],  # TODO Consider updating MLP code to support multiple devices when scaling up
                    state_dict=state_dict,
                    model_config=model_config,
                    layer_num=layer_num,
                )
                for i in range(args.moe.num_experts)
            ],
            state_dict=state_dict,
            layer_num=layer_num,
            moe_args=args.moe,
            device=devices,
        )
        self.attention_norm = TtRMSNorm(
            device=devices[0],
            state_dict=state_dict,
            model_config=model_config,
            layer_num=layer_num,  # TODO double check the logic for layer_num when scaling for all layers
            weight_key="attention_norm",
        )
        self.ffn_norm = TtRMSNorm(
            device=devices[0],
            state_dict=state_dict,
            model_config=model_config,
            layer_num=layer_num,  # TODO double check the logic for layer_num when scaling for all layers
            weight_key="ffn_norm",
        )

    def forward(
        self,
        xs: ttnn.Tensor,
        start_pos: int,
        current_pos: int,
        attn_masks: Optional[ttnn.Tensor],
    ) -> ttnn.Tensor:
        # TODO Consider updating the remaining rms_norm and MLP modules to support multi-device
        if not isinstance(xs, list):
            xs = [xs]

        # Attention module expects a list of inputs, start_pos, attn mask (multi-device support)
        attn_norm = [self.attention_norm(xs[0])]
        r = self.attention.forward(
            attn_norm,
            start_pos,
            current_pos,
            attn_masks,
        )
        # Attention also returns multiple outputs (multi-device support)
        r[0] = ttnn.reshape(r[0], (1, 1, 32, 4096))
        h = ttnn.experimental.tensor.add(xs[0], r[0])
        ttnn.deallocate(xs[0])
        ttnn.deallocate(r[0])
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = ttnn.experimental.tensor.add(h, r)
        ttnn.deallocate(h)
        return out
