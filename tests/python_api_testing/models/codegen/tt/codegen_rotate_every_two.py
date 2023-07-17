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

# Copied from transformers.models.gptj.modeling_gptj.rotate_every_two
def pt_rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    print(x1)
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    res = x.flatten(-2)
    print(res.shape)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def tt_rotate_every_two(x):

    x_shape = x.shape()
    slice_list_1 = [slice(None), slice(None), slice(None), slice(0, x_shape[3],2)]
    x1 = fallback_ops.tensor_slice(x, slice_list_1)
    #x1.print()
    pt_x1=tt2torch_tensor(x1)
    print(pt_x1)
    slice_list_2 = [slice(None), slice(None), slice(None), slice(1, x_shape[3],2)]
    x2 = fallback_ops.tensor_slice(x, slice_list_2)

    tt_const = fallback_ops.full(x2.shape(), -1.0)

    x2 = tt_lib.tensor.mul(x2, tt_const)

    x = fallback_ops.concat([x2, x1], dim=-1)

    x_shape = x.shape()

    #last_shape = x.shape()[-1]*x.shape()[-2]*x.shape()[-3]

    #x = tt_lib.fallback_ops.reshape(x, x.shape()[0], 1, 1, last_shape)

    x = fallback_ops.reshape(x, 1, 1, x_shape[2], x_shape[3])
    print(x.shape())
    return x
