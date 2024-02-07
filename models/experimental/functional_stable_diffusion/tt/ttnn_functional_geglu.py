# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn.functional as F
from models.experimental.functional_stable_diffusion.configuration_file import PYTORCH_FALLBACK_OPS


def geglu(config, hidden_states, parameters, device):
    if PYTORCH_FALLBACK_OPS["bmm"]:
        value = ttnn.to_torch(hidden_states)
        weight = ttnn.to_torch(parameters.proj.weight)
        output = torch.matmul(value, weight)
        output = ttnn.from_torch(output, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        output = ttnn.matmul(hidden_states, parameters.proj.weight)

    if PYTORCH_FALLBACK_OPS["add"]:
        value = ttnn.to_torch(output)
        bias = ttnn.to_torch(parameters.proj.bias)
        output = torch.add(value, bias)
        output = ttnn.from_torch(output, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        output = ttnn.add(output, parameters.proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    if PYTORCH_FALLBACK_OPS["split"]:
        value = ttnn.to_torch(output)
        hidden_states, gate = torch.split(value, split_size_or_sections=output.shape[-1] // 2, dim=-1)
        hidden_states = ttnn.from_torch(hidden_states, device=device, layout=ttnn.TILE_LAYOUT)
        gate = ttnn.from_torch(gate, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        hidden_states, gate = ttnn.split(output, split_size=output.shape[-1] // 2, dim=-1)
    del output

    if PYTORCH_FALLBACK_OPS["gelu"]:
        value = ttnn.to_torch(gate)
        act = F.gelu(value)
        act = ttnn.from_torch(act, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        act = ttnn.gelu(gate, memory_config=ttnn.L1_MEMORY_CONFIG)
    del gate

    if PYTORCH_FALLBACK_OPS["mul"]:
        value = ttnn.to_torch(hidden_states)
        activation = ttnn.to_torch(act)
        op = torch.mul(value, activation)
        return ttnn.from_torch(op, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        return ttnn.mul(hidden_states, act)
