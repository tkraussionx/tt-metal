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


def pt_split_heads(x, n_head, dim_head, mp_num):
    reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
    print('target')
    print(reshaped.shape)
    reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
    return reshaped


def tt_split_heads(x, n_head, dim_head, mp_num):

    x_shape = torch.Size(x.shape())
    new_shape_1 = x_shape[1:-1] + (n_head // mp_num, dim_head)
    print('nes1')
    print(new_shape_1)
    reshaped_1 = fallback_ops.reshape(x, new_shape_1[0], new_shape_1[1], new_shape_1[2], new_shape_1[3])
    reshaped_1_shape = torch.Size(reshaped_1.shape())
    new_shape_2 = x_shape[:-2] + (-1,) + reshaped_1_shape[-1:]
    reshaped_2 = fallback_ops.reshape(reshaped_1, new_shape_2[0], new_shape_2[1], new_shape_2[2], new_shape_2[3])
    return reshaped_2
