# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.demos.mixtral8x7b.tt.mixtral_decoder_ttnn import TtTransformerBlock
from models.demos.mixtral8x7b.tt.mixtral_rms_norm_ttnn import TtRMSNorm
import ttnn
from typing import Optional, List


class TtTransformer(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        args,
        dtype,
        layers,
        tt_cos_cached,
        tt_sin_cached,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.devices = devices
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    devices=devices,
                    state_dict=state_dict,
                    args=args,
                    dtype=dtype,
                    layer_num=i,
                    tt_cos_cached=tt_cos_cached,
                    tt_sin_cached=tt_sin_cached,
                )
                for i in layers
            ]
        )
        self.norm = [
            TtRMSNorm(
                device=dev,
                state_dict=state_dict,
                args=args,
                dtype=dtype,
                layer_num=None,
                weight_key="norm",
            )
            for dev in self.devices
        ]
        self.state_dict = state_dict

        self.output_weight = [
            ttnn.as_tensor(
                self.state_dict["output.weight"].permute(1, 0),
                device=dev,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=args.weight_cache_path(dtype) / "output.weight",
            )
            for dev in self.devices
        ]

        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        start_pos: int,
        current_pos: int,
        attn_masks: Optional[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
    ):
        for i, layer in enumerate(self.layers):
            x = layer(x, start_pos, current_pos, attn_masks, rot_mats)

        outputs = []
        x_norm = []
        for i in range(len(self.devices)):
            x_norm.append(self.norm[i](x[i]))
            ttnn.deallocate(x[i])
            # x_norm[i] = ttnn.permute(x_norm[i], (2, 1, 0, 3))
            output_i = ttnn.linear(
                x_norm[i],
                self.output_weight[i],
                core_grid=ttnn.CoreGrid(y=7, x=8),
                use_1d_systolic_array=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel,
            )
            outputs.append(output_i)
            ttnn.deallocate(x_norm[i])

        return outputs
