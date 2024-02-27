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
        args=None,
        devices=None,
        dtype=None,
        state_dict=None,
        base_address=None,
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

        self.attention = TtMistralAttention(
            devices=devices,
            state_dict=state_dict,
            base_url=f"{base_address}attention.",
            layer_num=layer_num,  # TODO double check the logic for layer_num when scaling for all layers
            dtype=dtype,
            configuration=args,
            tt_cos_cached=tt_cos_cached,
            tt_sin_cached=tt_sin_cached,
        )
        self.feed_forward = TtMistralMLP(
            device=devices[0],  # TODO Consider updating MLP code to support multiple devices when scaling up
            state_dict=state_dict,
            base_address=f"{base_address}feed_forward.",
            model_config=model_config,
        )
        self.attention_norm = TtRMSNorm(
            device=devices[0],
            base_address=f"{base_address}attention_norm.",
            state_dict=state_dict,
        )
        self.ffn_norm = TtRMSNorm(
            device=devices[0],
            base_address=f"{base_address}ffn_norm.",
            state_dict=state_dict,
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
