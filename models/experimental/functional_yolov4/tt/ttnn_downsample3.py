# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.experimental.functional_yolov4.tt.ttnn_resblock import TtResBlock


import ttnn
import tt_lib


class TtDownSample3:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.res = TtResBlock(parameters.res, 8, True)
        self.c4 = parameters.c4
        self.c5 = parameters.c5

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        output_tensor_c1 = self.c1(input_tensor)
        output_tensor_c1 = ttnn.to_torch(output_tensor_c1)
        output_tensor_c1 = ttnn.from_torch(
            output_tensor_c1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        output_tensor_c1 = ttnn.mish(output_tensor_c1)
        output_tensor_c1 = tt_lib.tensor.interleaved_to_sharded(
            output_tensor_c1, self.c3.conv.input_sharded_memory_config
        )

        output_tensor_c2 = self.c2(output_tensor_c1)
        output_tensor_c2 = ttnn.to_torch(output_tensor_c2)
        output_tensor_c2 = ttnn.from_torch(
            output_tensor_c2, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        output_tensor_c2 = ttnn.mish(output_tensor_c2)
        # output_tensor_c2 = tt_lib.tensor.interleaved_to_sharded(output_tensor_c2, self.c4.conv.input_sharded_memory_config)

        output_tensor = self.c3(output_tensor_c1)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c4.conv.input_sharded_memory_config)

        output_tensor = self.res(device, output_tensor)
        output_tensor = output_tensor.to(device, self.c4.conv.input_sharded_memory_config)

        output_tensor = self.c4(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([output_tensor, output_tensor_c2], dim=3)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c5.conv.input_sharded_memory_config)

        output_tensor = self.c5(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c5.conv.input_sharded_memory_config)
        return ttnn.from_device(output_tensor)
