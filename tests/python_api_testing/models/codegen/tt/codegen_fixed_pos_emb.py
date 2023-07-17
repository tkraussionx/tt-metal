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


# Copied from transformers.models.gptj.modeling_gptj.fixed_pos_embedding
def pt_fixed_pos_emb(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))

    pt_aranged = torch.arange(seq_len, dtype=torch.float)
    print(pt_aranged.shape)
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq).to(x.device).float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def tt_fixed_pos_emb(x, device, seq_dim=1, seq_len=None):

    x_shape = x.shape()
    dim = x_shape[-1]
    if seq_len is None:
        seq_len = x_shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))

    tt_aranged = tt_lib.tensor.arange(0, seq_len, 1)
    pt_aranged = tt2torch_tensor(tt_aranged)

    pt_aranged = pt_aranged.squeeze(1)
    pt_aranged = pt_aranged.squeeze(1)
    pt_aranged = pt_aranged.squeeze(1)

    sinusoid_inp = torch.einsum("i , j -> i j", pt_aranged, inv_freq)

    tt_sinusoid_inp = torch2tt_tensor(sinusoid_inp, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

    tt_sin = tt_lib.tensor.sin(tt_sinusoid_inp)
    tt_cos = tt_lib.tensor.cos(tt_sinusoid_inp)

    return tt_sin, tt_cos
