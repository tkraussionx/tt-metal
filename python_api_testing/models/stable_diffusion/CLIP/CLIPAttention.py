import math
import numpy as np
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pymetal import ttmetal as ttm
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize
from pymetal.ttmetal.utils import print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as tt_linear
from utils import move_to_device, move_to_cpu
from typing import Optional, Tuple, Union



class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, state_dict, config=None, hidden_size=None, num_attention_heads=None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size if config else hidden_size
        self.num_heads = config.num_attention_heads if config else num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        # self.dropout = config.attention_dropout # skipping since not important

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


        self.k_proj.weight = nn.Parameter(state_dict['text_model.encoder.layers.10.self_attn.k_proj.weight'])
        self.k_proj.bias = nn.Parameter(state_dict['text_model.encoder.layers.10.self_attn.k_proj.bias'])

        self.v_proj.weight = nn.Parameter(state_dict['text_model.encoder.layers.10.self_attn.v_proj.weight'])
        self.v_proj.bias = nn.Parameter(state_dict['text_model.encoder.layers.10.self_attn.v_proj.bias'])

        self.q_proj.weight = nn.Parameter(state_dict['text_model.encoder.layers.10.self_attn.q_proj.weight'])
        self.q_proj.bias = nn.Parameter(state_dict['text_model.encoder.layers.10.self_attn.q_proj.bias'])

        self.out_proj.weight = nn.Parameter(state_dict['text_model.encoder.layers.10.self_attn.out_proj.weight'])
        self.out_proj.bias = nn.Parameter(state_dict['text_model.encoder.layers.10.self_attn.out_proj.bias'])

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()



    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        N, bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)


        # continue here!
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = attn_weights
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class TtCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, device, state_dict, config=None, hidden_size=None, num_attention_heads=None):
        super().__init__()
        self.config = config
        self.device = device
        self.embed_dim = config.hidden_size if config else hidden_size
        self.num_heads = config.num_attention_heads if config else num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        # self.dropout = config.attention_dropout

        self.k_proj_weights = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.self_attn.k_proj.weight"]))
        self.k_proj_bias = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.self_attn.k_proj.bias"]))
        self.k_proj = tt_linear(self.embed_dim, self.embed_dim, self.k_proj_weights, self.k_proj_bias, device)

        self.v_proj_weights = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.self_attn.v_proj.weight"]))
        self.v_proj_bias = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.self_attn.v_proj.bias"]))
        self.v_proj = tt_linear(self.embed_dim, self.embed_dim, self.v_proj_weights, self.v_proj_bias, device)


        self.q_proj_weights = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.self_attn.q_proj.weight"]))
        self.q_proj_bias = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.self_attn.q_proj.bias"]))
        self.q_proj = tt_linear(self.embed_dim, self.embed_dim, self.q_proj_weights, self.q_proj_bias, device)

        self.out_proj_weights = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.self_attn.out_proj.weight"]))
        self.out_proj_bias = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.self_attn.out_proj.bias"]))
        self.out_proj = tt_linear(self.embed_dim, self.embed_dim, self.out_proj_weights, self.out_proj_bias, device)


    def _shape(self, tensor, seq_len: int, bsz: int):
        t = ttm.tensor.reshape(tensor, bsz, seq_len, self.num_heads, self.head_dim) .transpose(1, 2).contiguous()
        tt = ttm.tensor.transpose_hc(t)
        return t


    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,):
        """Input shape: 1 x Batch x Time x Channel"""

        N, bsz, tgt_len, embed_dim = hidden_states.shape()

        scale = torch.full(hidden_states.shape(), self.scale)
        scale = move_to_device(scale, self.device)
        query_states = self.q_proj(hidden_states)
        query_states = ttm.tensor.mul(scale, query_states)

        t_k_proj = self.k_proj(hidden_states)
        key_states = self._shape(t_k_proj, -1, bsz)

        t_v_proj = self.v_proj(hidden_states)
        value_states = self._shape(t_v_proj, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = ttm.tensor.reshape(query_states, *proj_shape)

        key_states = ttm.tensor.reshape(key_states, *proj_shape)
        value_states = ttm.tensor.reshape(value_states, *proj_shape)

        src_len = key_states.shape()[1]

        T_key_states = ttm.tensor.transpose(key_states)
        attn_weights = ttm.tensor.bmm(query_states, key_states)

        if attn_weights.shape() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if causal_attention_mask is not None:
            if causal_attention_mask.shape() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = ttm.tensor.reshape(attn_weights, bsz, self.num_heads, tgt_len, src_len)
            attn_weights = ttm.tensor.add(attn_weights, causal_attention_mask)

            attn_weights = ttm.tensor.reshape(attn_weights, 1, bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.shape() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )

            attn_weights = ttm.tensor.reshape(attn_weights, bsz, self.num_heads, tgt_len, src_len)
            attn_weights = ttm.tensor.add(attn_weights, attention_mask)

            attn_weights = ttm.tensor.reshape(attn_weights, 1, bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ttm.fused_ops.softmax(attn_weights)

        # intentionally ignoring output_attention line 103 to 111 and return arg
        # since it is not used

        # intentionally ignoring dropout since it does nothing in inference
        attn_probls = attn_weights # dropout

        attn_output = ttm.tensor.bmm(attn_probs, value_states)

        if attn_output.shape() != (1, bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = ttm.tensor.reshape(attn_output, bsz, self.num_heads, tgt_len, self.head_dim)

        attn_output = ttm.tensor.transpose_hc(attn_output)
        attn_output = ttm.tensor.reshape(1, bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped # output_attention is ignored since it is false and makes no difference in our case


def run_clip_attention_inference(device):

#         self.embed_dim = config.hidden_size
#         self.num_heads = config.num_attention_heads

    # hidden_states = 1, 77, 768
    #     attention_mask = None,
    #     causal_attention_mask  = (1, 1, 77, 77)
    #     output_attentions = False
    # change all 77 to 96
    from transformers import CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    state_dict = model.state_dict()
    config = model.config.text_config

    num_attention_heads = 1
    hidden_size = config.hidden_size
    D = 96 # real value is 77
    embed_dim = config.hidden_size

    hidden_state_shape = [1, 1, D, config.hidden_size]
    causal_attention_mask_shape = (1, 1, D, D)



    hidden_states = torch.randn(hidden_state_shape)
    attention_mask = None
    causal_attention_mask = torch.randn(causal_attention_mask_shape)
    output_attentions = False

    # hidden_states,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     causal_attention_mask: Optional[torch.Tensor] = None,
    #     output_attentions: Optional[bool] = False,):

    # torch_ca = CLIPAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)
    torch_ca = CLIPAttention(config=config, state_dict=state_dict)
    torch_out = torch_ca(hidden_states=hidden_states, causal_attention_mask=causal_attention_mask)

    tt_hidden_states = ttm.tensor.Tensor(tilize_to_list(hidden_states), hidden_state_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    tt_causal_attention_mask = ttm.tensor.Tensor(tilize_to_list(causal_attention_mask), causal_attention_mask_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    # tt_ca = TtCLIPAttention(device, hidden_size=hidden_size, num_attention_heads=num_attention_heads)

    tt_ca = TtCLIPAttention(device=device, config=config, state_dict=state_dict)

    tt_out = tt_ca(hidden_states=tt_hidden_states, causal_attention_mask=tt_causal_attention_mask).to(host).data()

    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)





if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_clip_attention_inference(device)
    ttm.device.CloseDevice(device)
