# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
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
        args,
        dtype,
        devices,
        state_dict,
        model_config,
        layers,
        tt_cos_cached,
        tt_sin_cached,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.devices = devices
        self.num_devices = len(devices)
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    args=args,
                    devices=self.devices,
                    dtype=dtype,
                    state_dict=state_dict,
                    model_config=model_config,
                    layer_num=i,
                    tt_cos_cached=tt_cos_cached,
                    tt_sin_cached=tt_sin_cached,
                )
                for i in layers
            ]
        )
        self.norm = TtRMSNorm(
            device=devices[0],
            state_dict=state_dict,
            model_config=model_config,
            layer_num=None,
            weight_key="norm",
        )
        self.state_dict = state_dict

        self.output_weight = ttnn.as_tensor(
            self.state_dict["output.weight"].permute(1, 0),
            device=devices[0],
            layout=ttnn.TILE_LAYOUT,
            dtype=model_config["LM_HEAD_MM_WEIGHTS_DTYPE"],
            memory_config=model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"],
            cache_file_name=Path(model_config["DEFAULT_WEIGHT_PATH"]) / "output.weight",
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
