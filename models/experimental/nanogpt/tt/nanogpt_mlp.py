# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from models.experimental.nanogpt.nanogpt_helper_funcs import format_tensor
from models.helper_funcs import Linear


class TtMLP(torch.nn.Module):
    def __init__(self, base_address, config, device, tt_cache_path, dtype):
        super().__init__()
        # Get the weights
        self.tt_weight_c_fc = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_fc.weight" + str(dtype) + ".bin"
        )
        self.tt_weight_c_proj = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_proj.weight" + str(dtype) + ".bin"
        )

        self.config = config
        self.device = device

        self.output_mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        )
        # Load biases
        self.tt_bias_c_fc = tt_lib.tensor.load_tensor(tt_cache_path + base_address + ".c_fc.bias" + str(dtype) + ".bin")

        self.tt_bias_c_proj = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_proj.bias" + str(dtype) + ".bin"
        )

        self.tt_weight_c_fc = tt_lib.tensor.transpose(self.tt_weight_c_fc, -2, -1)
        self.tt_weight_c_proj = tt_lib.tensor.transpose(self.tt_weight_c_proj, -2, -1)

        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, self.tt_weight_c_fc, self.tt_bias_c_fc)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, self.tt_weight_c_proj, self.tt_bias_c_proj)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x = format_tensor(x, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        x1 = self.c_fc(x)
        x2 = tt_lib.tensor.gelu(x1)
        x3 = self.c_proj(x2)
        return x3
