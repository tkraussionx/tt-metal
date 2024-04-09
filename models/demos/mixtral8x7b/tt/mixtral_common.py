# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def precompute_freqs(dim: int, end: int, theta: float = 1000000.0):
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
    return rot_emb_matrix, emb_size


def get_rotation_mat(dhead, end):
    cos, sin = precompute_freqs(dhead, end)
    rot_mat, emb_size = freqs_to_rotation_matrix(cos, sin)
    rot_mat = [rot_mat[i, :, :] for i in range(emb_size)]
    return rot_mat


def prepare_inputs_ttnn(x_bsh, hidden_size, head_dim, max_seq_len, devices):
    """
    Prepare inputs for decode mode. Assume that current token is at
    start_pos, and KV cache has valid data up to start_pos.
    x: (batch, seq, hidden_dim)
    start_pos: int

    B: batch (32)
    S: sequence len (1)
    H: dim (4096)
    """
    if x_bsh is None:  # First token
        rot_mat = get_rotation_mat(dhead=head_dim, end=max_seq_len * 2)
        rot_mats = []
        for device in devices:
            rot_mats.append(
                [
                    ttnn.from_torch(rot_mat_i, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                    for rot_mat_i in rot_mat
                ]
            )
        return rot_mats
    else:
        assert x_bsh.size(2) == hidden_size
        assert len(x_bsh.size()) == 3

        batch = x_bsh.size(0)
        seq_len = x_bsh.size(1)
        assert seq_len == 1, "Only supporting decode mode"

        rot_mat = get_rotation_mat(dhead=head_dim, end=max_seq_len * 2)

        x_1SBH = x_bsh.view(1, seq_len, batch, hidden_size)

        xs_1SBH, rot_mats = [], []
        for device in devices:
            xs_1SBH.append(ttnn.from_torch(x_1SBH, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
            rot_mats.append(
                [
                    ttnn.from_torch(rot_mat_i, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                    for rot_mat_i in rot_mat
                ]
            )
        return (
            xs_1SBH,
            rot_mats,
        )


# Sample logits from a distribution
def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs.squeeze(), top_p)
    else:
        next_token = torch.argmax(logits, dim=-1)

    return next_token


# Helper function to recreate Mixtral state dictionary. Reads the consolidated weights provided in HuggingFace, separates the 8 experts and saves the updated dict into a new single file.
def create_state_dict(model_args):
    state_dict = {}
    for i in range(1 + (model_args.n_layers - 1) // 4):
        state_dict_i = torch.load(model_args.consolidated_weights_path(i), map_location="cpu")
        state_dict.update(state_dict_i)

    partial_state_dict = {
        k: v
        for k, v in state_dict.items()
        if (
            any([f"layers.{i}." in k for i in range(model_args.n_layers)])
            or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
        )
    }

    base_address = "feed_forward."
    for l in range(model_args.n_layers):
        pre = f"layers.{l}."
        partial_state_dict[pre + base_address + "gate.weight"] = partial_state_dict[
            pre + "block_sparse_moe.gate.weight"
        ]
        del partial_state_dict[pre + "block_sparse_moe.gate.weight"]

        w1 = partial_state_dict[pre + "block_sparse_moe.w1"].contiguous().clone()
        w2 = partial_state_dict[pre + "block_sparse_moe.w2"].contiguous().clone()
        w3 = partial_state_dict[pre + "block_sparse_moe.w3"].contiguous().clone()
        ffn_dim = 14336
        for i in range(8):
            partial_state_dict[pre + base_address + f"experts.{i}.w1.weight"] = (
                w1[ffn_dim * i : ffn_dim * (i + 1), :].contiguous().clone()
            )
            partial_state_dict[pre + base_address + f"experts.{i}.w2.weight"] = (
                w2[ffn_dim * i : ffn_dim * (i + 1), :].T.clone().contiguous()
            )
            partial_state_dict[pre + base_address + f"experts.{i}.w3.weight"] = (
                w3[ffn_dim * i : ffn_dim * (i + 1), :].contiguous().clone()
            )
        partial_state_dict.pop(pre + "block_sparse_moe.w1")
        partial_state_dict.pop(pre + "block_sparse_moe.w2")
        partial_state_dict.pop(pre + "block_sparse_moe.w3")
    torch.save(partial_state_dict, "partial_state_dict.pt")
