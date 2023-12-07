# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.utility_functions import tt2torch_tensor
import torch
import tt_lib
from tt_lib import fallback_ops
from transformers import GPT2LMHeadModel
from tt_lib.utils import pad_weight
from models.experimental.nanogpt.tt.nanogpt_model import TtGPT
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm


def cache_weights_in_weka(device, dtype, reset_seeds):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    state_dict = model_hf.state_dict()
    weights_dtype = dtype

    # initial weights are stored in "models/experimental/nanogpt/weights/" and moved to weka path
    file_name = "models/experimental/nanogpt/weights/"
    for key, value in state_dict.items():
        if key.startswith("transformer.wte.") or key.startswith("transformer.wpe."):
            torch.save(value, file_name + str(key) + ".pt")
            continue
        elif len(value.shape) == 0:
            continue
        if len(value.shape) == 1:
            value = value.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif len(value.shape) == 3:
            value = value.unsqueeze(0)
        elif len(value.shape) == 2:
            value = value.unsqueeze(0).unsqueeze(0)
        if value.shape[-2] % 32 == 0 and value.shape[-1] % 32 == 0:
            value = tt_lib.tensor.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                weights_dtype,
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(tt_lib.tensor.Layout.TILE)
        else:
            value = pad_weight(value)
            value = tt_lib.tensor.Tensor(
                value.reshape(-1).tolist(),
                value.shape,
                weights_dtype,
                tt_lib.tensor.Layout.ROW_MAJOR,
            ).to(tt_lib.tensor.Layout.TILE)
        tt_lib.tensor.dump_tensor(file_name + str(key) + str(weights_dtype) + ".bin", value)


def generate(
    idx: torch.Tensor,
    tt_model: TtGPT,
    config=None,
    tokenizer=None,
    max_new_tokens: int = 20,
    device=None,
    temperature: int = 1.0,
    top_k=None,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size :]
        tt_logits = tt_model.forward(idx_cond)

        logits_shapes = tt_logits.shape()

        slice_list = [
            slice(None),
            slice(None),
            slice(logits_shapes[2] - 1, logits_shapes[2]),
            slice(None),
        ]
        tt_logits = fallback_ops.tensor_slice(tt_logits, slice_list)

        tt_temperature = tt_lib.tensor.full(tt_logits.shape(), temperature)

        tt_temperature = tt_lib.tensor.recip(tt_temperature)
        tt_logits = tt_lib.tensor.mul(tt_logits, tt_temperature)

        logits = tt_to_torch_tensor(tt_logits)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        tt_logits = torch_to_tt_tensor_rm(logits, device, put_on_device=False)
        tt_probs = fallback_ops.softmax(tt_logits, dim=-1)
        probs = tt_to_torch_tensor(tt_probs)
        probs = probs.squeeze(0)
        probs = probs.squeeze(1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

        res = []
        for i, x in enumerate(idx):
            res.append(tokenizer.decode(x))
    return res
