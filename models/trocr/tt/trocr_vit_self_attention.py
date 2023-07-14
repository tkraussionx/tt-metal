import torch.nn as nn
from models.trocr.tt.trocr_vit_configuration import TtViTConfig
from models.helper_funcs import Linear
import math
import tt_lib
from tt_lib import fallback_ops
from typing import Optional, Union, Tuple
from models.utility_functions import torch_to_tt_tensor_rm


class TtViTSelfAttention(nn.Module):
    def __init__(
        self, config: TtViTConfig, base_address: str, state_dict, device, host
    ) -> None:
        super().__init__()
        self.device = device
        self.host = host
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.recip_sqrt_attention_head_size_tensor = tt_lib.fallback_ops.full(
            (1, 1, 32, 32), 1 / math.sqrt(self.attention_head_size)
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.query.weight"], self.host
        )
        self.query = Linear(
            config.hidden_size, self.all_head_size, self.query_weight, bias=None
        )

        self.key_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.key.weight"], self.host
        )
        self.key = Linear(
            config.hidden_size, self.all_head_size, self.key_weight, bias=None
        )

        self.value_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.value.weight"], self.host
        )
        self.value = Linear(
            config.hidden_size, self.all_head_size, self.value_weight, bias=None
        )

    def transpose_for_scores(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        new_x_shape = (x.shape()[0], x.shape()[2]) + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = fallback_ops.reshape(x, *new_x_shape)
        return tt_lib.tensor.permute(x, 0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[
        Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor], Tuple[tt_lib.tensor.Tensor]
    ]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer_T = tt_lib.tensor.transpose(key_layer)
        attention_scores = tt_lib.tensor.bmm(query_layer, key_layer_T)

        attention_scores = tt_lib.tensor.bcast(
            attention_scores,
            self.recip_sqrt_attention_head_size_tensor,
            tt_lib.tensor.BcastOpMath.MUL,
            tt_lib.tensor.BcastOpDim.HW,
        )

        # Normalize the attention scores to probabilities.
        attention_probs = fallback_ops.softmax(attention_scores, dim=-1)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tt_lib.tensor.mul(attention_probs, head_mask)

        context_layer = tt_lib.tensor.bmm(attention_probs, value_layer)

        context_layer = tt_lib.tensor.permute(context_layer, 0, 2, 1, 3)
        new_context_layer_shape = (
            (1,) + tuple(context_layer.shape()[:-2]) + (self.all_head_size,)
        )
        context_layer = fallback_ops.reshape(context_layer, *new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs
