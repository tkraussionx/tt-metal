# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor


class TtRMSNorm(nn.Module):
    def __init__(self, config, dim: int, eps: float, state_dict=None, base_address="", device=None):
        super().__init__()
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device
        self.eps = eps
        self.dim = dim

        self.weight = torch_to_tt_tensor_rm(state_dict[f"{self.base_address}.weight"], self.device)

    def _norm(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        pow_tensor = tt_lib.tensor.power(x, 2)
        pow_tensor = tt_to_torch_tensor(pow_tensor).mean(-1, keepdim=True) + self.eps
        pow_tensor = torch_to_tt_tensor_rm(pow_tensor, self.device, put_on_device=False)
        pow_tensor = tt_lib.tensor.sqrt(pow_tensor)
        pow_tensor = tt_lib.tensor.recip(pow_tensor)
        return tt_lib.tensor.bcast(x, pow_tensor, tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.W)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        output = self._norm(x)
        return tt_lib.tensor.bcast(output, self.weight, tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.H)
