from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from typing import Dict, Tuple, Optional

from tests.python_api_testing.models.clip.clip_utils import make_address, make_linear
from torch import nn
import torch

from clip_utils import make_linear, make_address
from tt_lib.tensor import Tensor as tt_tensor
import tt_lib
from tt_lib import fallback_ops


class TtCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = make_linear(
            self.embed_dim, self.embed_dim, "k_proj", state_dict, base_address, device
        )
        self.v_proj = make_linear(
            self.embed_dim, self.embed_dim, "v_proj", state_dict, base_address, device
        )
        self.q_proj = make_linear(
            self.embed_dim, self.embed_dim, "q_proj", state_dict, base_address, device
        )
        self.out_proj = make_linear(
            self.embed_dim, self.embed_dim, "out_proj", state_dict, base_address, device
        )
        # self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: tt_tensor, seq_len: int, bsz: int):
        # reshape = tt_lib.tensor.reshape
        reshape = fallback_ops.reshape
        transpose_hc = tt_lib.tensor.transpose_hc
        tensor = reshape(tensor, bsz, seq_len, self.num_heads, self.head_dim)
        return transpose_hc(tensor)

    def forward(
        self,
        hidden_states: tt_tensor,
        attention_mask: Optional[tt_tensor] = None,
        causal_attention_mask: Optional[tt_tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[tt_tensor], Optional[Tuple[tt_tensor]]]:
        """Input shape: Batch x Time x Channel"""

        _, bsz, tgt_len, embed_dim = hidden_states.shape()
        # NOTE:
        # TODO:
        # get query proj
        # query_states = self.q_proj(hidden_states) * self.scale
        query_states = self.q_proj(hidden_states)
        # TODO: constant prop scale
        scale_tensor = fallback_ops.full(query_states.shape(), self.scale)
        query_states = tt_lib.tensor.mul(query_states, scale_tensor)

        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, 1, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        # query_states = tt_lib.tensor.reshape(query_states, *proj_shape)
        query_states = fallback_ops.reshape(query_states, *proj_shape)
        # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # key_states = key_states.view(*proj_shape)
        # key_states = tt_lib.tensor.reshape(key_states, *proj_shape)
        key_states = fallback_ops.reshape(key_states, *proj_shape)
        # value_states = value_states.view(*proj_shape)
        # value_states = tt_lib.tensor.reshape(value_states, *proj_shape)
        value_states = fallback_ops.reshape(value_states, *proj_shape)
        src_len = key_states.shape()[2]  # two since, tt adds batch dim
        t_key_states = tt_lib.tensor.transpose_hc(key_states)
        # attn_weights = torch.bmm(query_states, t_key_states)
        attn_weights = tt_lib.tensor.matmul(query_states, t_key_states)

        if attn_weights.shape() != (bsz * self.num_heads, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, 1, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            assert False, f"this is not implemented! {causal_attention_mask.shape}"
            if causal_attention_mask.shape() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + causal_attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            assert False, "this is not implemented!"
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = fallback_ops.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            assert False, "not used in TT"
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = attn_weights

        attn_output = tt_lib.tensor.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = tt_lib.tensor.reshape(
            attn_output, bsz, self.num_heads, tgt_len, self.head_dim
        )
        # attn_output = attn_output.transpose(1, 2)
        attn_output = tt_lib.tensor.transpose_hc(attn_output)
        # attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = tt_lib.tensor.reshape(attn_output, 1, bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped
