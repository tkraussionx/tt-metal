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


# Copied from transformers.models.gptj.modeling_gptj.duplicate_interleave
def pt_duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    print('PT')
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    print(m.shape)
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    print(m.shape)
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    print(m.shape)
    return m


def tt_duplicate_interleave(m):

    m_shape = m.shape()
    dim0 = m_shape[0]
    dim1 = m_shape[1]
    dim2 = m_shape[2]
    dim3 = m_shape[3]
    print('TT')
    m = fallback_ops.reshape(m, 1, 1, dim2*dim3, 1)
    print(m.shape())
    m = fallback_ops.repeat(m, torch.Size([1, 1, 1, 2]))
    print(m.shape())


    m_shape = m.shape()

    last_dim = m_shape[3]
    dim0 = m_shape[2]
    print(last_dim)
    print(dim0)

    m = fallback_ops.reshape(m, 1, 1, 1, last_dim*dim0)
    print(m.shape())

    return m
