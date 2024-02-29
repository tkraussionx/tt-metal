# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import torch.nn as nn
import ttnn


class TtRMSNorm(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        model_config,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        self.state_dict = state_dict

        torch_weight = self.state_dict["weight"].unsqueeze(0).expand(32, -1)
        cache_name = Path(model_config["DEFAULT_WEIGHT_PATH"]) / (base_address + ".weight")

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)
        return x
