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
    reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
    return reshaped


def tt_split_heads(x, n_head, dim_head, mp_num):
    new_shape_1 = x.shape[:-1] + (n_head // mp_num, dim_head)
    reshaped_1 = fallback_ops.reshape(x, new_shape_1[0], new_shape_1[1], new_shape_1[2])
    new_shape_2 = x.shape[:-2] + (-1,) + reshaped_1.shape[-1:]
    reshaped_2 = fallback_ops.reshape(new_shape_2)
    return reshaped_2
