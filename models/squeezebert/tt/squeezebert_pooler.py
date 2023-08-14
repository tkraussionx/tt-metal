import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import tt_lib
from models.helper_funcs import Linear as TtLinear


class TtSqueezeBert_Pooler(nn.Module):
    def __init__(self, config, base_address="", state_dict=None, device=None) -> None:
        super().__init__()
        self.config = config
        self.base_address = base_address
        self.device = device
        self.dense_weight = torch_to_tt_tensor_rm(
            state_dict[f"{self.base_address}.dense.weight"], self.device
        )
        self.dense_bias = torch_to_tt_tensor_rm(
            state_dict[f"{self.base_address}.dense.bias"], self.device
        )
        self.dense = TtLinear(
            self.dense_weight.shape()[-1],
            self.dense_weight.shape()[-2],
            self.dense_weight,
            self.dense_bias,
        )

        self.activation = tt_lib.tensor.tanh

    def forward(self, hidden_states: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        first_token_tensor = tt_to_torch_tensor(hidden_states).squeeze(0)[:, 0]
        first_token_tensor = torch_to_tt_tensor_rm(
            first_token_tensor, self.device, put_on_device=True
        )

        pooled_output = self.dense(first_token_tensor)

        pooled_output = self.activation(pooled_output)
        return pooled_output
