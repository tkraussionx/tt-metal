# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Tuple
import tt_lib
from models.utility_functions import tt2torch_tensor, torch2tt_tensor


def tt_all_reduce(tensors, output_mem_config=None):
    """
    reduction on a list of tensors
    """
    if len(tensors) == 1:
        return tensors[0]
    base_tensor = tensors[0]
    for tensor in tensors[1:]:
        # base_tensor = tt_lib.tensor.add(base_tensor, tensor, output_mem_config=output_mem_config)  Cbinding doesnt support this optional argument passed in as None
        if output_mem_config is not None:
            base_tensor = tt_lib.tensor.add(base_tensor, tensor, output_mem_config)
        else:
            base_tensor = tt_lib.tensor.add(base_tensor, tensor)
    dev = base_tensor.device()
    # Emulate replication on all chips
    res_pt = tt2torch_tensor(base_tensor)
    res = [torch2tt_tensor(res_pt.clone(), dev) for _ in range(len(tensors))]
    return res


def generate_cos_sin_cache(
    tt_devices,
    head_dim,
    base_url,
    max_position_embeddings=2048,
    base=10000,
    model_config=None,
):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(
        max_position_embeddings,
        device=inv_freq.device,
        dtype=inv_freq.dtype,
    )
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)

    tt_cos_cached_host = torch2tt_tensor(
        emb.cos()[None, None, :, :],
        None,
        tt_memory_config=model_config["COS_CACHED_WEIGHTS_MEMCFG"],
        tt_dtype=model_config["COS_CACHED_WEIGHTS_DTYPE"],
    )
    tt_cos_cached = [
        tt_cos_cached_host.to(tt_device, model_config["COS_CACHED_WEIGHTS_MEMCFG"]) for tt_device in tt_devices
    ]
    tt_sin_cached_host = torch2tt_tensor(
        emb.sin()[None, None, :, :],
        None,
        tt_memory_config=model_config["SIN_CACHED_WEIGHTS_MEMCFG"],
        tt_dtype=model_config["SIN_CACHED_WEIGHTS_DTYPE"],
    )
    tt_sin_cached = [
        tt_sin_cached_host.to(tt_device, model_config["SIN_CACHED_WEIGHTS_MEMCFG"]) for tt_device in tt_devices
    ]

    return tt_cos_cached, tt_sin_cached


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def freqs_to_rotation_matrix(cos_freqs, sin_freqs):
    """
    Transform cos/sin frequencies to a rotation matrix.
    """
    emb_size, emb_dim = cos_freqs.shape
    dhead = emb_dim * 2
    rot_emb_matrix = torch.zeros(emb_size, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    rot_emb_matrix = rot_emb_matrix.transpose(-1, -2)  # Necessary for correct rotation when applied as (x @ R)
    return rot_emb_matrix


def gather_rotary_emb(rot_emb_matrix, position_ids):
    """
    Gather the rotary embeddings for a given position_ids
    """
    batch_size, seqlen = position_ids.shape
    emb_size, _, dhead = rot_emb_matrix.shape
    position_ids = position_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, dhead, dhead)
    rot_emb = rot_emb_matrix.gather(0, position_ids).view(seqlen, batch_size, dhead, dhead)
    return rot_emb
