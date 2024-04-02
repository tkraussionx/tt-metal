# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.demos.falcon7b.tt.model_utils import get_weights_cached


class TtFalconMLP(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"
        dense_h_to_4h_str = f"{layer_name}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h_str = f"{layer_name}.mlp.dense_4h_to_h.weight"

        self.dense_h_to_4h_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            dense_h_to_4h_str,
            weight_config_str="DENSE_H_TO_4H_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[dense_h_to_4h_str], -2, -1) if state_dict else None),
        )
        self.dense_4h_to_h_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            dense_4h_to_h_str,
            weight_config_str="DENSE_4H_TO_H_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[dense_4h_to_h_str], -2, -1) if state_dict else None),
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = []
        self.h4hin = tt2torch_tensor(x[0])
        # self.h4hweight = tt2torch_tensor(self.dense_h_to_4h_weights[0])
        self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"] = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
        )

        self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"] = tt_lib.tensor.DataType.BFLOAT8_B
        for device_id in range(len(x)):
            hidden_states.append(
                tt_lib.tensor.falcon_dense_h_to_4h_matmul(
                    x[device_id],
                    self.dense_h_to_4h_weights[device_id],
                    fused_activation=[tt_lib.tensor.FusibleActivation.GELU, True],
                    output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                )
            )
            # x[device_id].deallocate()
        self.h4hout = tt2torch_tensor(hidden_states[0])

        self.h4hin2 = tt2torch_tensor(x[0])
        # self.h4hweight2 = tt2torch_tensor(self.dense_h_to_4h_weights[0])
        assert torch.allclose(self.h4hin, self.h4hin2)
        # assert torch.allclose(self.h4hweight, self.h4hweight2)

        for device_id in range(len(x)):
            hidden_states[device_id] = tt_lib.tensor.falcon_dense_4h_to_h_matmul(
                hidden_states[device_id],
                self.dense_4h_to_h_weights[device_id],
                output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                packer_l1_acc=True,
            )

        # return TT Tensor
        return hidden_states
