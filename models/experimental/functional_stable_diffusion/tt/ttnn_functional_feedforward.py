# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_geglu import geglu
from models.experimental.functional_stable_diffusion.configuration_file import PYTORCH_FALLBACK_OPS


def feedforward(config, hidden_states, parameters, device):
    act = geglu(config, hidden_states, parameters.net[0], device)
    if PYTORCH_FALLBACK_OPS["bmm"]:
        value = ttnn.to_torch(act)
        weight = ttnn.to_torch(parameters.net[2].weight)
        output = torch.matmul(value, weight)
        output = ttnn.from_torch(output, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        output = act @ parameters.net[2].weight
    if PYTORCH_FALLBACK_OPS["add"]:
        value = ttnn.to_torch(output)
        bias = ttnn.to_torch(parameters.net[2].bias)
        output = torch.add(value, bias)
        output = ttnn.from_torch(output, device=device, layout=ttnn.TILE_LAYOUT)
    else:
        output = ttnn.add(output, parameters.net[2].bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    return output
