# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib

from typing import List
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class TtFalconMLP:
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

        num_devices = len(devices)
        self.dense_h_to_4h_weights = []
        self.dense_4h_to_h_weights = []
        for i in range(num_devices):
            dense_h_to_4h_path = (
                tt_cache_path
                / f"{dense_h_to_4h_str}_{i}_{num_devices}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            if (dense_h_to_4h_path).exists():
                self.dense_h_to_4h_weights.append(
                    tt_lib.tensor.load_tensor(str(dense_h_to_4h_path)).to(
                        devices[i], self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"]
                    )
                )
            else:
                dense_h_to_4h_weights_host = torch2tt_tensor(
                    torch.transpose(
                        torch.chunk(self.state_dict[dense_h_to_4h_str], num_devices)[i],
                        -2,
                        -1,
                    ),
                    None,
                    tt_memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
                )
                self.dense_h_to_4h_weights.append(
                    dense_h_to_4h_weights_host.to(devices[i], self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"])
                )
                tt_lib.tensor.dump_tensor(
                    str(dense_h_to_4h_path),
                    dense_h_to_4h_weights_host,
                )
            dense_4h_to_h_path = (
                tt_cache_path
                / f"{dense_4h_to_h_str}_{i}_{num_devices}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
            )
            if (dense_4h_to_h_path).exists():
                self.dense_4h_to_h_weights.append(
                    tt_lib.tensor.load_tensor(str(dense_4h_to_h_path)).to(
                        devices[i], self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"]
                    )
                )
            else:
                dense_4h_to_h_weights_host = torch2tt_tensor(
                    torch.transpose(
                        torch.chunk(self.state_dict[dense_4h_to_h_str], num_devices)[i],
                        -2,
                        -1,
                    ),
                    None,
                    tt_memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
                )
                self.dense_4h_to_h_weights.append(
                    dense_4h_to_h_weights_host.to(devices[i], self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"])
                )
                tt_lib.tensor.dump_tensor(
                    str(dense_4h_to_h_path),
                    dense_4h_to_h_weights_host,
                )

    def __call__(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        # print("x")
        # x_print = x[0].cpu()
        # print(tt2torch_tensor(x_print))
        # print("weights")
        # weights_print = self.dense_h_to_4h_weights[0].cpu()
        # print(tt2torch_tensor(weights_print))

        hidden_states = []
        num_shards = len(x)

        assert num_shards == len(self.devices)
        # check if all devices are the same
        same_device = all(self.devices[i] == self.devices[i + 1] for i in range(num_shards - 1))

        for i in range(num_shards):
            # FUNCTIONAL: comment in
            # x[i] = tt_lib.tensor.sharded_to_interleaved( # FUNCTIONAL
            #     x[i],
            #     output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            # )
            hidden_states.append(
                tt_lib.operations.primary.matmul_1d(
                    x[i],
                    self.dense_h_to_4h_weights[i],
                    program_config=self.model_config["DENSE_H_TO_4H_MM_PROGCFG"],
                    output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                    # output_mem_config=self.model_config["DEFAULT_MEMCFG"],  # FUNCTIONAL
                    output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                )
            )
            x[i].deallocate(True)

        # FUNCTIONAL: comment out
        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.tensor.sharded_to_interleaved(
                hidden_states[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )

        if same_device:
            concat_hidden_states = tt_lib.tensor.concat(hidden_states, 3)
            for i in range(num_shards):
                hidden_states[i].deallocate(True)

            # FUNCTIONAL: comment out
            concat_hidden_states = tt_lib.tensor.interleaved_to_sharded(
                concat_hidden_states, sharded_mem_config=self.model_config["MLP_ALL_GATHER_OUTPUT_MEMCFG"]
            )

            mlp_output = []
            for i in range(num_shards):
                mlp_output.append(
                    tt_lib.operations.primary.matmul_1d(
                        concat_hidden_states,
                        self.dense_4h_to_h_weights[i],
                        program_config=self.model_config["DENSE_4H_TO_H_MM_PROGCFG"],
                        output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                        # output_mem_config=self.model_config["DEFAULT_MEMCFG"],  # FUNCTIONAL
                        output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                    )
                )

            concat_hidden_states.deallocate(True)
        else:
            hidden_states = tt_lib.tensor.all_gather(
                hidden_states,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            )
            for i in range(len(hidden_states)):
                hidden_states[i] = tt_lib.tensor.interleaved_to_sharded(
                    hidden_states[i], sharded_mem_config=self.model_config["MLP_ALL_GATHER_OUTPUT_MEMCFG"]
                )

            mlp_output = []
            for i in range(num_shards):
                mlp_output.append(
                    tt_lib.operations.primary.matmul_1d(
                        hidden_states[i],
                        self.dense_4h_to_h_weights[i],
                        program_config=self.model_config["DENSE_4H_TO_H_MM_PROGCFG"],
                        output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                        output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                    )
                )
                hidden_states.deallocate(True)

        # return TT Tensor
        return mlp_output
