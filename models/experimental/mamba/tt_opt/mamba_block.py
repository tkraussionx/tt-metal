# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

#import tt_lib

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.tt_opt.mamba_one_step_ssm import TtMambaSSM

class TtMambaBlock(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device,
        state_dict,
        num_users,
        hidden_size,
        configs
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args


        self.tt_ssm = TtMambaSSM(self.args,self.device,self.state_dict, num_users, hidden_size, configs)

    def forward(self, x):

        x = self.tt_ssm(x)

        return x
