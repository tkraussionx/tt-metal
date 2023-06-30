from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import pickle
import tt_lib

from python_api_testing.models.swin.tt.swin_self_attention import (
    TtSwinSelfAttention,
)

from python_api_testing.models.utility_functions_new import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from python_api_testing.models.swin.tt.swin_self_output import TtSwinSelfOutput


class TtSwinAttention(nn.Module):
    def __init__(
        self,
        config,
        dim,
        num_heads,
        window_size,
        state_dict,
        base_address,
        device,
        host,
        layer_index,
        index,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.index = index
        self.host = host
        self.self = TtSwinSelfAttention(
            config,
            dim,
            num_heads,
            window_size,
            state_dict,
            base_address=f"{base_address}.self",
            device=device,
            host=host,
        )
        self.output = TtSwinSelfOutput(
            config, dim, state_dict, f"{base_address}.output", device, host
        )
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[tt_lib.tensor.Tensor]:
        pt_self_input = tt_to_torch_tensor(hidden_states, self.host).squeeze(0)
        name = (
            "layer_"
            + str(self.layer_index)
            + "_tt_self_attention_input_"
            + str(self.index)
            + ".pkl"
        )

        with open(name, "wb") as file:
            pickle.dump(pt_self_input, file)

        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, output_attentions
        )

        pt_self_outputs = tt_to_torch_tensor(self_outputs[0], self.host).squeeze(0)
        name = (
            "layer_"
            + str(self.layer_index)
            + "_tt_self_attention_output_"
            + str(self.index)
            + ".pkl"
        )

        with open(name, "wb") as file:
            pickle.dump(pt_self_outputs, file)

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs
