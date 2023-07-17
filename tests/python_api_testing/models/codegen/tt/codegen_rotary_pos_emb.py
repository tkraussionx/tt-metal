import torch
from torch.nn import functional as F

import tt_lib
from tt_lib.fallback_ops import fallback_ops
import python_api_testing.models.codegen.tt.codegen_rotate_every_two as codegen_rotate_every_two


import math

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

# Copied from transformers.models.gptj.modeling_gptj.apply_rotary_pos_emb
def pt_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = (duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :] for t in sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

def tt_rotary_pos_emb(x, device, sincos, offset=0):
    sin, cos = (duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :] for t in sincos)

    tt_cos = fallback_ops.full(x.shape(), cos)
    res1 = tt_lib.tensor.mul(x, tt_cos)

    res2 = codegen_rotate_every_two(x)
    tt_sin = fallback_ops.full(x.shape(), cos)

    res2 = tt_lib.tensor.mul(res2, tt_sin)

    output = tt_lib.tensor.add(res1, res2)

    return output
