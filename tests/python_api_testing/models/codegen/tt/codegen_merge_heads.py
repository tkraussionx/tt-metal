import torch
from torch.nn import functional as F

import tt_lib
from tt_lib.fallback_ops import fallback_ops

import math

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)


def pt_merge_heads(tensor, num_attention_heads, attn_head_size):
    """
    Merges attn_head_size dim and num_attn_heads dim into n_ctx
    """
    if len(tensor.shape) == 5:
        tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
    elif len(tensor.shape) == 4:
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
    else:
        raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
    new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
    print(new_shape)
    return tensor.view(new_shape)

def tt_merge_heads(tensor, num_attention_heads, attn_head_size):
    tt_permuted = tt_lib.tensor.permute(tensor, 0, 2, 1, 3)

    new_shape = torch.Size(tt_permuted.shape()[:-2]) + (num_attention_heads * attn_head_size,)
    print(new_shape)
    result = fallback_ops.reshape(tt_permuted, 1, new_shape[0], new_shape[1], new_shape[2])

    return result
