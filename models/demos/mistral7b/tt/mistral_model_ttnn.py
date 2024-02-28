# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs
from models.demos.mistral7b.tt.mistral_decoder_ttnn import TtTransformerBlock
from models.demos.mistral7b.tt.mistral_rms_norm_ttnn import TtRMSNorm
import ttnn
from typing import Optional


class TtTransformer(nn.Module):
    def __init__(
        self,
        args=None,
        dtype=None,
        devices=None,
        state_dict=None,
        base_address=None,
        model_config=None,
        tt_cos_cached=None,
        tt_sin_cached=None,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.devices = devices
        self.num_devices = len(devices)
        self.base_address = base_address
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    args=args,
                    devices=self.devices,
                    dtype=dtype,
                    state_dict=state_dict,
                    base_address=f"layers.{i}.",
                    model_config=model_config,
                    tt_cos_cached=tt_cos_cached,
                    tt_sin_cached=tt_sin_cached,
                )
                for i in range(args.n_layers)
            ]
        )
        self.norm = TtRMSNorm(
            device=devices[0],
            base_address=f"norm.",
            state_dict=state_dict,
        )
        self.state_dict = state_dict

        self.output_weight = ttnn.from_torch(
            self.state_dict["output.weight"].permute(1, 0), device=devices[0], layout=ttnn.TILE_LAYOUT
        )

    def forward(
        self,
        xs: ttnn.Tensor,
        start_pos: int,
        current_pos: int,
        attn_masks: Optional[ttnn.Tensor],
    ):
        for layer in self.layers:
            xs = layer(xs, start_pos, current_pos, attn_masks)

        # output = self.output(self.norm(xs))
        xs = self.norm(xs)
        print("SHAPES!", xs.shape, self.output_weight.shape)
        output = ttnn.linear(xs, self.output_weight)
        ttnn.deallocate(xs)
        return output
