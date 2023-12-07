# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from models.helper_funcs import Linear


class TtMLP(torch.nn.Module):
    def __init__(self, base_address, config, device, tt_cache_path, dtype):
        super().__init__()

        self.out_mem_config_l1 = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
        )
        self.device = device

        # Get the weights
        self.tt_weight_c_fc = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_fc.weight" + str(dtype) + ".bin"
        ).to(device, self.out_mem_config_l1)

        self.tt_weight_c_proj = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_proj.weight" + str(dtype) + ".bin"
        ).to(device, self.out_mem_config_l1)

        self.config = config

        # Load biases
        self.tt_bias_c_fc = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_fc.bias" + str(dtype) + ".bin"
        ).to(device, self.out_mem_config_l1)

        self.tt_bias_c_proj = tt_lib.tensor.load_tensor(
            tt_cache_path + base_address + ".c_proj.bias" + str(dtype) + ".bin"
        ).to(device, self.out_mem_config_l1)

        self.tt_weight_c_fc = tt_lib.tensor.transpose(self.tt_weight_c_fc, -2, -1)
        self.tt_weight_c_proj = tt_lib.tensor.transpose(self.tt_weight_c_proj, -2, -1)

        self.c_fc = Linear(
            config.n_embd,
            4 * config.n_embd,
            self.tt_weight_c_fc,
            self.tt_bias_c_fc,
            output_mem_config=self.out_mem_config_l1,
        )
        self.c_proj = Linear(
            4 * config.n_embd,
            config.n_embd,
            self.tt_weight_c_proj,
            self.tt_bias_c_proj,
            output_mem_config=self.out_mem_config_l1,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        x1 = self.c_fc(x)
        x2 = tt_lib.tensor.gelu(x1, output_mem_config=self.out_mem_config_l1)
        x3 = self.c_proj(x2)
        return x3
