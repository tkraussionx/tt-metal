# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from typing import Callable

from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.tt.mamba_block import TtMambaBlock
from models.demos.mamba.reference.decode_model import RMSNorm


class TtResidualBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args
        self.configs = configs

        rms_norm_weight_name = "norm.weight"
        self.rms_norm_weights = load_fn(rms_norm_weight_name, return_as_torch=True)
        self.torch_rms_norm = RMSNorm(args.d_model)
        self.torch_rms_norm.weight = torch.nn.Parameter(self.rms_norm_weights)

        self.tt_mamba_block = TtMambaBlock(self.args, self.device, configs, load_fn)

    def forward(self, x):
        assert len(x.shape) == 4, "Mamba residual block expects inputs to be rank 4"

        residual = x
        # rms_norm_weights = ttnn.to_memory_config(self.rms_norm_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        # x = ttnn.experimental.tensor.interleaved_to_sharded(x, sharded_mem_config=self.configs["sharded_h"])
        # mamba_x = ttnn.experimental.operations.primary.rmsnorm(
        #     x,
        #     self.args.eps,
        #     rms_norm_weights,
        #     program_config=self.configs["SHARDED_NORM_PRGM_CFG"],
        #     output_mem_config=self.configs["sharded_h"],
        # )
        mamba_x = ttnn.to_torch(x)
        mamba_x = self.torch_rms_norm(mamba_x)
        mamba_x = ttnn.from_torch(
            mamba_x,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.configs["dtype"]["activations"],
        )
        # mamba_x = ttnn.to_memory_config(mamba_x, memory_config=ttnn.L1_MEMORY_CONFIG)
        # ttnn.deallocate(rms_norm_weights)
        mamba_x = self.tt_mamba_block(mamba_x)

        return ttnn.add(residual, mamba_x, dtype=self.configs["dtype"]["activations"])
