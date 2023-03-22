import math
import numpy as np
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pymetal import ttmetal as ttm
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, untilize
from pymetal.ttmetal.utils import print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as tt_linear
from python_api_testing.fused_ops.layernorm import Layernorm as tt_layernorm
from CLIPAttention import TtCLIPAttention, CLIPAttention
from CLIPMLP import TtCLIPMLP, CLIPMLP
from transformers import CLIPModel, CLIPConfig


class CLIPEncoderLayer(nn.Module):
    def __init__(self, state_dict, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = CLIPAttention(config=config, state_dict=state_dict)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm1.weight = nn.Parameter(state_dict['text_model.encoder.layers.10.layer_norm1.weight'])
        self.layer_norm1.bias = nn.Parameter(state_dict['text_model.encoder.layers.10.layer_norm1.bias'])

        self.mlp = CLIPMLP(config=config, state_dict=state_dict)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2.weight = nn.Parameter(state_dict['text_model.encoder.layers.10.layer_norm2.weight'])
        self.layer_norm2.bias = nn.Parameter(state_dict['text_model.encoder.layers.10.layer_norm2.bias'])


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:

        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs



class TtCLIPEncoderLayer(nn.Module):
    def __init__(self, device, state_dict, config=None, hidden_size=None):
        super().__init__()
        self.device = device
        self.embed_dim = config.hidden_size if config else hidden_size
        self.self_attn = TtCLIPAttention(device=device, config=config, state_dict=state_dict)


        self.layer_norm1_weight = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.layer_norm1.weight"]))
        self.layer_norm1_bias = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.layer_norm1.bias"]))

# def Layernorm(gamma: float, beta: float, epsilon: float, H, W, device, num_dims = 2):
        H = self.embed_dim
        W = -1
        self.layer_norm1 = tt_layernorm(gamma=self.layer_norm1_weight, beta=self.layer_norm1_bias, epsilon=config.layer_norm_eps, H=H, W=W, device=device)


        self.mlp = TtCLIPMLP(device=device, config=config, state_dict=state_dict)

        self.layer_norm2_weight = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.layer_norm2.weight"]))
        self.layer_norm2_bias = tilize_to_list(pad_weight(state_dict["text_model.encoder.layers.10.layer_norm2.bias"]))

        self.layer_norm2 = tt_layernorm(gamma=self.layer_norm2_weight, beta=self.layer_norm2_bias, epsilon=config.layer_norm_eps, H=H, W=W, device=device)


    def forward(
        self,
        hidden_states,
        attention_mask,
        causal_attention_mask,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:

        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = ttm.tensor.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = ttm.tensor.add(residual, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


def run_clip_encoder_layer_inference(device):


    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    state_dict = model.state_dict()
    config = model.config.text_config

    D = 96 # should be 77
    hidden_states_shape = [1, 1, D, 512]
    attention_mask = None
    causal_attention_mask_shape = [1, 1, D, D]
    output_attentions = False

    hidden_states = torch.randn(hidden_states_shape)
    causal_attention_mask = torch.randn(causal_attention_mask_shape)


    torch_encoder = CLIPEncoderLayer(config=config, state_dict=state_dict)
    torch_out = torch_encoder(hidden_states=hidden_states, attention_mask=attention_mask, causal_attention_mask= causal_attention_mask, output_attentions=output_attentions)

    tt_hidden_states = ttm.tensor.Tensor(tilize_to_list(hidden_states), hidden_states_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    tt_causal_attention_mask = ttm.tensor.Tensor(tilize_to_list(causal_attention_mask), causal_attention_mask_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)


    tt_encoder = TtCLIPEncoderLayer(device=device, config=config, state_dict=state_dict)

    tt_out = tt_encoder(hidden_states=tt_hidden_states, causal_attention_mask=tt_causal_attention_mask, attention_mask=attention_mask, output_attentions=output_attentions).to(host).data()

    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)






if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_clip_encoder_layer_inference(device)
    ttm.device.CloseDevice(device)
