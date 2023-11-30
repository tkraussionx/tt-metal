# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.utility_functions import tt2torch_tensor
import torch
import tt_lib
from transformers import GPT2LMHeadModel
from tt_lib.utils import pad_weight


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
        while len(value.shape) < 4:
            value = value.unsqueeze(0)
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
