# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.tt.residual_block import TtResidualBlock

class MambaTT(torch.nn.Module):
    def __init__(
        self,
        reference_model,
        num_layers,
        device: tt_lib.device
    ):
        super().__init__()
        print(f"Initalizing MambaTT with {num_layers} layers")
        self.embedding = reference_model.embedding
        self.args = reference_model.args
        self.device = device
        self.layers = [TtResidualBlock(self.args, device, reference_model.layers[i].state_dict()) for i in range(num_layers)]

        self.norm_f = reference_model.norm_f

        self.lm_head = reference_model.lm_head

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) #(BS, 1, 1, E)
        x = torch2tt_tensor(
            x,
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        for layer in self.layers:
            x = layer(x)

        x = tt2torch_tensor(x).squeeze(1)
        x = self.norm_f(x)
        x = self.lm_head(x)
        return x