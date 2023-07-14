import torch.nn as nn
from models.trocr.tt.trocr_vit_configuration import TtViTConfig
import tt_lib
from typing import Optional, Union, Tuple, Dict

from models.trocr.tt.trocr_vit_self_attention import TtViTSelfAttention
from models.trocr.tt.trocr_vit_self_output import TtViTSelfOutput


class TtViTAttention(nn.Module):
    def __init__(
        self, config: TtViTConfig, base_address: str, state_dict: Dict, device, host
    ) -> None:
        super().__init__()
        self.attention = TtViTSelfAttention(
            config, f"{base_address}.attention", state_dict, device, host
        )
        self.output = TtViTSelfOutput(
            config, f"{base_address}.output", state_dict, device, host
        )

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[
        Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor], Tuple[tt_lib.tensor.Tensor]
    ]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs
