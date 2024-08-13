# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import tt_lib
from models.experimental.functional_fadnetpp.tt.common import Conv


class TtResBlock:
    def __init__(self, parameters, n_in, n_out, model, stride=1) -> None:
        self.conv1 = Conv(
            [
                parameters["resblock_1_conv1"]["ttnn_module_args"]["batch_size"],
                parameters["resblock_1_conv1"]["ttnn_module_args"]["input_height"],
                parameters["resblock_1_conv1"]["ttnn_module_args"]["input_width"],
                parameters["resblock_1_conv1"]["ttnn_module_args"]["in_channels"],
            ],
            (
                parameters["resblock_1_conv1"]["ttnn_module_args"]["out_channels"],
                parameters["resblock_1_conv1"]["ttnn_module_args"]["in_channels"],
                parameters["resblock_1_conv1"]["ttnn_module_args"]["kernel_size"][0],
                parameters["resblock_1_conv1"]["ttnn_module_args"]["kernel_size"][1],
            ),
            model.resblock_1_conv1.weight,
            model.resblock_1_bn1.bias,
            bn_weights=model.resblock_1_bn1.weight,
            bn_running_var=model.resblock_1_bn1.running_var,
            bn_running_mean=model.resblock_1_bn1.running_mean,
            height_sharding=True,
            mesh_mapper=None,
        )
        self.conv2 = Conv(
            [
                parameters["resblock_2_conv2"]["ttnn_module_args"]["batch_size"],
                parameters["resblock_2_conv2"]["ttnn_module_args"]["input_height"],
                parameters["resblock_2_conv2"]["ttnn_module_args"]["input_width"],
                parameters["resblock_2_conv2"]["ttnn_module_args"]["in_channels"],
            ],
            (
                parameters["resblock_2_conv2"]["ttnn_module_args"]["out_channels"],
                parameters["resblock_2_conv2"]["ttnn_module_args"]["in_channels"],
                parameters["resblock_2_conv2"]["ttnn_module_args"]["kernel_size"][0],
                parameters["resblock_2_conv2"]["ttnn_module_args"]["kernel_size"][1],
            ),
            model.resblock_2_conv2.weight,
            model.resblock_2_bn2.bias,
            bn_weights=model.resblock_2_bn2.weight,
            bn_running_var=model.resblock_2_bn2.running_var,
            bn_running_mean=model.resblock_2_bn2.running_mean,
            height_sharding=True,
            mesh_mapper=None,
        )
        self.sc = False
        if stride != 1 or n_out != n_in:
            self.sc = True
            # self.shortcut = parameters["resblock_sc_conv"]
            self.shortcut = Conv(
                [
                    parameters["resblock_sc_conv"]["ttnn_module_args"]["batch_size"],
                    parameters["resblock_sc_conv"]["ttnn_module_args"]["input_height"],
                    parameters["resblock_sc_conv"]["ttnn_module_args"]["input_width"],
                    parameters["resblock_sc_conv"]["ttnn_module_args"]["in_channels"],
                ],
                (
                    parameters["resblock_sc_conv"]["ttnn_module_args"]["out_channels"],
                    parameters["resblock_sc_conv"]["ttnn_module_args"]["in_channels"],
                    parameters["resblock_sc_conv"]["ttnn_module_args"]["kernel_size"][0],
                    parameters["resblock_sc_conv"]["ttnn_module_args"]["kernel_size"][1],
                ),
                model.shortcut_c.weight,
                model.shortcut_b.bias,
                bn_weights=model.shortcut_b.weight,
                bn_running_var=model.shortcut_b.running_var,
                bn_running_mean=model.shortcut_b.running_mean,
                height_sharding=True,
                mesh_mapper=None,
            )

    def __call__(self, device, input_tensor):
        if self.sc:
            input_tensor = input_tensor.to(device)
            # input_tensor = tt_lib.tensor.interleaved_to_sharded(
            #     input_tensor, self.shortcut.conv.input_sharded_memory_config
            # )
            residual = input_tensor
            residual = self.shortcut(device, input_tensor)
        else:
            input_tensor = input_tensor.to(device, self.conv1.conv.input_sharded_memory_config)
            residual = input_tensor
        output_tensor_h = input_tensor
        output_tensor_1 = self.conv1(output_tensor_h)
        output_tensor_1 = ttnn.relu(output_tensor_1)

        if output_tensor_1.shape[3] > 256:
            memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED,
                tt_lib.tensor.BufferType.L1,
            )
            output_tensor_1 = tt_lib.tensor.sharded_to_interleaved(output_tensor_1, memory_config)
            residual = tt_lib.tensor.sharded_to_interleaved(residual, memory_config)
            output_tensor_1 = tt_lib.tensor.interleaved_to_sharded(
                output_tensor_1, self.conv2.conv.input_sharded_memory_config
            )
            residual = tt_lib.tensor.interleaved_to_sharded(residual, self.conv2.conv.input_sharded_memory_config)

        else:
            output_tensor_1 = output_tensor_1.to(device, self.conv2.conv.input_sharded_memory_config)

        output_tensor_h = self.conv2(output_tensor_1)

        output_tensor_h += residual
        output_tensor_h = ttnn.relu(output_tensor_h)
        return ttnn.from_device(output_tensor_h)
