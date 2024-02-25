# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.helper_funcs import Linear
from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.tt.mamba_one_step_ssm import TtMambaSSM


class TtMambaBlock(torch.nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        device: tt_lib.device,
        state_dict,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args

        in_proj_weight_name = "mixer.in_proj.weight"
        self.ssm_in_proj_weights = torch2tt_tensor(
            self.state_dict[in_proj_weight_name][: self.args.d_inner, :],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        self.ssm_in_proj = Linear(self.args.d_model, self.args.d_inner, self.ssm_in_proj_weights, bias=None)

        self.mlp_proj_weights = torch2tt_tensor(
            self.state_dict[in_proj_weight_name][self.args.d_inner :, :],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        self.mlp_proj = Linear(self.args.d_model, self.args.d_inner, self.mlp_proj_weights, bias=None)

        out_proj_weight_name = "mixer.out_proj.weight"
        self.out_proj_weights = torch2tt_tensor(
            self.state_dict[out_proj_weight_name],
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )
        self.out_proj = Linear(self.args.d_inner, self.args.d_model, self.out_proj_weights, bias=None)

        conv1d_weight_name = "mixer.conv1d.weight"
        self.conv1d_weights = []
        # conv1d_weights (2E, 1, 4)->(2E, 1)->(2E, 1, 1)->(1, 1, 2E)
        for i in range(4):
            self.conv1d_weights.append(
                torch2tt_tensor(
                    self.state_dict[conv1d_weight_name][:, :, i]
                    .unsqueeze(-1)
                    .permute(1, 2, 0)
                    .repeat(self.args.batch_size, 1, 1)
                    .view(self.args.batch_size, 1, 1, self.args.d_inner),
                    self.device,
                    tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
                    tt_memory_config=tt_lib.tensor.MemoryConfig(
                        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
                    ),
                    tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
                )
            )

        conv1d_bias_name = "mixer.conv1d.bias"
        self.conv1d_bias = torch2tt_tensor(
            self.state_dict[conv1d_bias_name]
            .repeat(self.args.batch_size, 1)
            .view(self.args.batch_size, 1, 1, self.args.d_inner),
            self.device,
            tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
            tt_memory_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
            tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
        )

        self.conv_states = []
        for i in range(4):
            self.conv_states.append(
                torch2tt_tensor(
                    torch.zeros(self.args.batch_size, 1, 1, self.args.d_inner),
                    self.device,
                    tt_layout=tt_lib.tensor.Layout.ROW_MAJOR,
                    tt_memory_config=tt_lib.tensor.MemoryConfig(
                        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
                    ),
                    tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
                )
            )
        self.tt_ssm = TtMambaSSM(self.args, self.device, self.state_dict)

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        res = self.mlp_proj(x)
        x = self.ssm_in_proj(x)
        # left shift conv states
        for i in range(3):
            self.conv_states[i] = self.conv_states[i + 1]
        self.conv_states[3] = x

        x = tt_lib.tensor.mul(self.conv_states[0], self.conv1d_weights[0])
        for i in range(1, 4):
            x = tt_lib.tensor.add(x, tt_lib.tensor.mul(self.conv_states[i], self.conv1d_weights[i]))
        x = tt_lib.tensor.add(x, self.conv1d_bias)
        torch_data = tt2torch_tensor(x)
        torch.save(torch_data, "before_silu.pt")
        x = tt_lib.tensor.silu(x)
        torch_data = tt2torch_tensor(x)
        torch.save(torch_data, "after_silu.pt")
        x = self.tt_ssm(x)
        res = tt_lib.tensor.silu(res)
        x = tt_lib.tensor.mul(x, res)
        res.deallocate()
        x = self.out_proj(x)
        return x
