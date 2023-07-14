import torch.nn as nn

from models.trocr.tt.trocr_vit_configuration import TtViTConfig
from models.helper_funcs import Linear

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
import tt_lib


class TtViTPooler(nn.Module):
    def __init__(
        self, config: TtViTConfig, base_address: str, state_dict, device, host
    ):
        super().__init__()
        self.host = host
        self.device = device

        self.weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense.weight"], self.host
        )
        self.bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense.bias"], self.host
        )
        self.dense = Linear(
            config.hidden_size, config.hidden_size, self.weight, self.bias
        )
        self.activation = tt_lib.tensor.tanh

    def forward(self, hidden_states) -> tt_lib.tensor.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = tt_to_torch_tensor(hidden_states, self.host)
        first_token_tensor = hidden_states[:, 0]
        first_token_tensor = torch_to_tt_tensor_rm(first_token_tensor, self.host)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
