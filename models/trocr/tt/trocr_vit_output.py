import torch.nn as nn
from models.trocr.tt.trocr_vit_configuration import TtViTConfig
from models.helper_funcs import Linear
from models.utility_functions import torch_to_tt_tensor_rm
import tt_lib


class TtViTOutput(nn.Module):
    def __init__(
        self, config: TtViTConfig, base_address, state_dict, device, host
    ) -> None:
        super().__init__()

        self.host = host

        self.weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense.weight"], self.host
        )
        self.bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense.bias"], self.host
        )
        self.dense = Linear(
            config.intermediate_size, config.hidden_size, self.weight, self.bias
        )

    def forward(
        self, hidden_states: tt_lib.tensor.Tensor, input_tensor: tt_lib.tensor.Tensor
    ) -> tt_lib.tensor.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = tt_lib.tensor.add(hidden_states, input_tensor)
        return hidden_states
