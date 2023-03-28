import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from typing import Optional
from pymetal import ttlib as ttl
from utility_functions import pad_activation, pad_weight, tilize_to_list, get_oom_of_float, print_diff_argmax
from python_api_testing.fused_ops.linear import Linear as TtLinear
from pymetal.ttlib.fused_ops.softmax import softmax as TtSoftmax
from diffusers import StableDiffusionPipeline
from utils import move_to_cpu, move_to_device

'''
class AttentionBlock(torch.nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)

        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.linear_geglu_2.weight.data.fill_(1)
        self.linear_geglu_2.bias.data.fill_(0)

        self.linear_geglu_2.weight.data.fill_(1)
        self.linear_geglu_1.bias.data.fill_(0)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))   # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)  # (n, c, hw)
        x = x.view((n, c, h, w))    # (n, c, h, w)

        return self.conv_output(x) + residue_long


class TtAttentionBlock(torch.nn.Module):
    def __init__(self,  n_head: int, n_embd: int, d_context=768, state_dict=None):
        super().__init__()

        channels = n_head * n_embd

        self.torch_groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.torch_conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.torch_layernorm_1 = nn.LayerNorm(channels)
        self.torch_attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.torch_layernorm_2 = nn.LayerNorm(channels)
        self.torch_attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.torch_layernorm_3 = nn.LayerNorm(channels)
        # in_features, out_features, weight, bias, device
        # TODO: fill in weights and bias
        self.linear_geglu_1 = TtLinear(channels, 4 * channels * 2)

        self.linear_geglu_2 = TtLinear(4 * channels, channels)


        self.torch_conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)


    def forward(self, x, context):
        residue_long = x

        input_shape = [2, 320, 64, 64]
        x = torch.Tensor(x.to(host).data()).reshape(input_shape)

        x = untilize(x)
        return self.torch_groupnorm(x)

'''


class TtAttentionBlock(nn.Module):
    """

    Parameters:
        channels (`int`): The number of channels in the input and output.
        num_head_channels (`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        rescale_output_factor (`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
        eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """
    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
        state_dict=None,
        base_address='encoder.mid_block.attentions.0',
        device=None,
        host=None
    ):
        super().__init__()
        self.device = device
        self.host = host
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.rescale_output_factor = rescale_output_factor
        self._attention_op = None

        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)
        self.group_norm.weight = nn.Parameter(state_dict[f"{base_address}.group_norm.weight"])
        self.group_norm.bias = nn.Parameter(state_dict[f"{base_address}.group_norm.bias"])

        q_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.query.weight"]))
        q_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.query.bias"]))

        k_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.key.weight"]))
        k_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.key.bias"]))

        v_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.value.weight"]))
        v_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.value.bias"]))

        proj_weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj_attn.weight"]))
        proj_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj_attn.bias"]))

        self.query = TtLinear(channels, channels, q_weights, q_bias, device)
        self.key = TtLinear(channels, channels, k_weights, k_bias, device)
        self.value = TtLinear(channels, channels, v_weights, v_bias, device)
        self.proj_attn = TtLinear(channels, channels, proj_weights, proj_bias, device)

        self._attention_op = None


    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape()
        head_size = self.num_heads
        tensor = ttl.tensor.reshape(tensor, batch_size, seq_len, head_size, dim // head_size)

        tensor = move_to_cpu(tensor, self.host)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        tensor = move_to_device(tensor, self.device)

        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape()
        head_size = self.num_heads
        tensor = ttl.tensor.reshape(tensor, batch_size // head_size, head_size, seq_len, dim)
        tensor = move_to_cpu(tensor, self.host)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        tensor = move_to_device(tensor, self.device)
        return tensor

    def _baddbmm(self, query_proj, key_proj, value_proj, scale):
        input1 = torch.empty(1, query_proj.shape()[0], query_proj.shape()[1], key_proj.shape()[1])
        input1 = move_to_device(input1, self.device)
        key_proj_T = ttl.tensor.transpose(key_proj)
        beta = 0
        alpha = scale
        _bmm = ttl.tensor.bmm(key_proj_T, query_proj)
        _scale = torch.full(scale, _bmm.shape())
        _scale = move_to_device(_scale, self.device)
        return ttl.tensor.mul(_scale, _bmm)
        # _bmm = ttl.tensor.add(input1, _bmm)
        # since beta=0


    def _attention_score(self, query_proj, key_proj, value_proj, scale):

        attention_scores = self._baddbmm(query_proj, key_proj, value_proj, scale)
        attention_probs = TtSoftmax(attention_scores)
        return ttl.tensor.bmm(attention_probs, value_proj)



    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape()

        # norm
        hidden_states = move_to_cpu(hidden_states, self.host)
        hidden_states = self.group_norm(hidden_states)
        hidden_states = move_to_device(hidden_states, self.device)

        hidden_states = ttl.tensor.reshape(hidden_states, 1, batch, channel, height * width)
        hidden_states = ttl.tensor.transpose_hc(hidden_states)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        scale = 1 / math.sqrt(self.channels / self.num_heads)

        query_proj = self.reshape_heads_to_batch_dim(query_proj)
        key_proj = self.reshape_heads_to_batch_dim(key_proj)
        value_proj = self.reshape_heads_to_batch_dim(value_proj)

        attention_scores = self._attention_score(query_proj, key_proj, value_proj, scale)


        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)

        hidden_states = ttl.tensor.transpose(hidden_states)
        hidden_states = ttl.tensor.reshape(hidden_states, batch, channel, height, width)

        hidden_states = ttl.tensor.add(hidden_states, residual)

        recip = torch.full(1/self.rescale_output_factor, residual.shape())
        recip = move_to_device(recip, self.device)

        return ttl.tensor.mul(hidden_states, recip)


# in_channels :  512
# temb channels:  None
# eps:  1e-06
# resnet groups 32
# dropout 0.0
# time_embedding_norm default
# output scale factor:  1
# pre norm True
# attn_num_head_channels,  None

def run_attention_inference(device):
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)

    vae = pipe.vae
    vae.eval()
    state_dict = vae.state_dict()
    vae_encoder = pipe.vae.encoder
    attention = vae_encoder.mid_block.attentions[0]


    in_channels = 512
    eps = 1e-06
    resnet_groups = 32

    input_shape  = [1, 512, 64, 64]
    input = torch.randn(input_shape)


    torch_out = attention(input)
    print("pytorch is done, moving on to device")
    tt_input = ttl.tensor.Tensor(tilize_to_list(input), input_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    tt_resnet = TtAttentionBlock(channels=in_channels, num_head_channels=None, norm_num_groups=resnet_groups, eps=eps,  state_dict=state_dict, device=device, host=host,)

    tt_out = tt_resnet(tt_input).to(host).data()

    tt_untilized_output = untilize(torch.Tensor(tt_out).reshape(torch_out.shape))
    print_diff_argmax(tt_untilized_output, torch_out)
    assert np.allclose(torch_out.detach().numpy(), tt_untilized_output.numpy(), 1e-5, 0.17)



if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_attention_inference(device)
    ttl.device.CloseDevice(device)
