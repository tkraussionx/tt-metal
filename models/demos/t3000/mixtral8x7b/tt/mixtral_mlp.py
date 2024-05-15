# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh


class TtMixtralMLP(torch.nn.Module):
    def __init__(self, device_mesh, state_dict, args, layer_num, dtypes):
        super().__init__()

        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.dtypes = dtypes
        self.model_args = args
        self.model_config = args.get_model_config()

        base_name = lambda expert_num: f"layers.{layer_num}.feed_forward.experts.{expert_num}"
        torch_weight = lambda name: torch.concat(
            [
                self.state_dict[f"{base_name(expert_num)}.{name}.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0)
                for expert_num in range(8)
            ],
            dim=0,
        )
        cache_name = lambda name: args.weight_cache_path(dtypes[name]) / (
            f"layers.{layer_num}.feed_forward_multidevice_unsqueezed.experts.{name}"
        )
        as_tensor = lambda name: ttnn.as_tensor(
            torch_weight(name),
            dtype=dtypes[name],
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=self.model_config["MLP_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("w1")
        self.w1 = ttnn.to_device(self.w1, device_mesh)
        self.w2 = as_tensor("w2")
        self.w2 = ttnn.to_device(self.w2, device_mesh)
        self.w3 = as_tensor("w3")
        self.w3 = ttnn.to_device(self.w3, device_mesh)

        x_shape = ttnn.Shape([1, 1, args.max_batch_size, args.dim])
        h_shape = ttnn.Shape([1, 1, args.max_batch_size, args.hidden_dim])
        # self.w1_program_config = ttnn.experimental.operations.primary.create_matmul_1d_systolic_array_program_config(
        #     input_shape_a=x_shape,
        #     input_shape_b=self.w1.shape,
        #     core_grid=self.model_args.max_grid_size,
        #     fused_activation="silu",
        #     fp32_dst=self.model_args.get_compute_kernel_config().fp32_dest_acc_en,
        # )
        # self.w1_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=(8, 1),
        #     fused_activation="silu",
        # )
        # self.w2_program_config = ttnn.operations.matmul.create_matmul_1d_systolic_array_program_config(
        #     input_shape_a=h_shape,
        #     input_shape_b=self.w2.shape,
        #     core_grid=self.model_args.max_grid_size,
        #     fp32_dst=self.model_args.get_compute_kernel_config().fp32_dest_acc_en,
        # )
        # self.w3_program_config = ttnn.operations.matmul.create_matmul_1d_systolic_array_program_config(
        #     input_shape_a=x_shape,
        #     input_shape_b=self.w3.shape,
        #     core_grid=self.model_args.max_grid_size,
        #     fp32_dst=self.model_args.get_compute_kernel_config().fp32_dest_acc_en,
        # )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        w1_out = ttnn.linear(
            x,
            self.w1,
            core_grid=self.model_args.max_grid_size,
            use_1d_systolic_array=True,
            # program_config=self.w1_program_config,
            activation="silu",
            # compute_with_storage_grid_size=(7, 6),
            output_mem_config=self.model_config["FF1_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
        )

        w3_out = ttnn.matmul(  # experimental.operations.primary.matmul_1d(
            x,
            self.w3,
            core_grid=self.model_args.max_grid_size,
            use_1d_systolic_array=True,
            # program_config=self.w3_program_config,
            output_mem_config=self.model_config["FF3_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
        )
        w2_in = ttnn.mul(w1_out, w3_out)
        w2_out = ttnn.matmul(  # .experimental.operations.primary.matmul_1d(
            w2_in,
            self.w2,
            core_grid=self.model_args.max_grid_size,
            use_1d_systolic_array=True,
            # program_config=self.w2_program_config,
            output_mem_config=self.model_config["FF2_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_args.get_compute_kernel_config(),
            output_dtype=ttnn.bfloat16,
        )

        return w2_out
