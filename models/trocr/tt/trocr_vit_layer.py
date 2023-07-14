import torch.nn as nn
from typing import Dict, Optional, Union, Tuple

from models.trocr.tt.trocr_vit_configuration import TtViTConfig
from models.trocr.tt.trocr_vit_attention import TtViTAttention
from models.trocr.tt.trocr_vit_output import TtViTOutput
from models.trocr.tt.trocr_vit_intermediate import TtViTIntermediate

import tt_lib
from tt_lib import fallback_ops


class TtViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(
        self, config: TtViTConfig, base_address: str, state_dict: Dict, device, host
    ) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TtViTAttention(
            config, f"{base_address}.attention", state_dict, device, host
        )
        self.intermediate = TtViTIntermediate(
            config, f"{base_address}.intermediate", state_dict, device, host
        )
        self.output = TtViTOutput(
            config, f"{base_address}.output", state_dict, device, host
        )

        lbw = state_dict[f"{base_address}.layernorm_before.weight"]
        lbb = state_dict[f"{base_address}.layernorm_before.bias"]
        self.layernorm_before = fallback_ops.LayerNorm(
            normalized_shape=config.hidden_size,
            weights=lbw,
            biases=lbb,
            eps=config.layer_norm_eps,
        )

        law = state_dict[f"{base_address}.layernorm_after.weight"]
        lab = state_dict[f"{base_address}.layernorm_after.bias"]
        self.layernorm_after = fallback_ops.LayerNorm(
            normalized_shape=config.hidden_size,
            weights=law,
            biases=lab,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[
        Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor], Tuple[tt_lib.tensor.Tensor]
    ]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = tt_lib.tensor.add(attention_output, hidden_states)

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
