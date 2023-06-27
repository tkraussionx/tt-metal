from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


from torch import nn

from activations import ACT2FN
from tt_lib.tensor import Tensor as tt_tensor
from clip_utils import make_linear


class TtCLIPMLP(nn.Module):
    def __init__(self, config, base_address, state_dict, device) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]

        self.fc1 = make_linear(
            config.hidden_size,
            config.intermediate_size,
            "fc1",
            state_dict,
            base_address,
            device,
        )
        self.fc2 = make_linear(
            config.intermediate_size,
            config.hidden_size,
            "fc2",
            state_dict,
            base_address,
            device,
        )
        # self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: tt_tensor) -> tt_tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
