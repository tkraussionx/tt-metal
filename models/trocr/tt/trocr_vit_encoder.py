import torch.nn as nn

from models.trocr.tt.trocr_vit_configuration import TtViTConfig
from models.trocr.tt.trocr_vit_layer import TtViTLayer

from typing import Dict, Optional, Union

import tt_lib


class TtViTEncoder(nn.Module):
    def __init__(
        self, config: TtViTConfig, base_address: str, state_dict: Dict, device, host
    ) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                TtViTLayer(
                    config, f"{base_address}.layer.{_}", state_dict, device, host
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, tt_lib.tensor.Tensor]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                assert False, "TT does not support training yet"
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict or True:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
