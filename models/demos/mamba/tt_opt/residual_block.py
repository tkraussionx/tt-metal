# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

#import tt_lib
import ttnn

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.tt_opt.mamba_block import TtMambaBlock

class TtResidualBlock(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device,
        state_dict,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args

        '''
        rms_norm_weight_name = "norm.weight"
        self.rms_norm_weights = torch2tt_tensor(
            self.state_dict[rms_norm_weight_name],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        '''
        self.tt_mamba_block = TtMambaBlock(self.args,self.device,self.state_dict)

    def forward(self, x):
        #mamba_input = tt_lib.tensor.rmsnorm(x, self.args.eps, self.rms_norm_weights)
        xt = self.tt_mamba_block(x)
        #x = tt_lib.tensor.add(x, mamba_input)
        return x
