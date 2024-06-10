# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import LightweightModule


class TtMixtralMLP(LightweightModule):
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
        if args.dummy_weights:
            cache_name = lambda _: None
        else:
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
        self.w2 = as_tensor("w2")
        self.w3 = as_tensor("w3")

        self.prefill_mlp_config = self.model_config["PREFILL_MLP_COMPUTE_CONFIG"]

    def forward(self, x: ttnn.Tensor, mode="prefill") -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        if mode == "prefill":
            x = ttnn.reshape(x, (1, 4, 2048, 4096))
            # from models.demos.t3000.mixtral8x7b.tt.create_program_config import create_matmul_program_config
            # pc = create_matmul_program_config(input_tensor_a=x, input_tensor_b=self.w1, core_grid=ttnn.CoreGrid(y=8, x=8), activation="silu", use_1d_systolic_array=False, compute_kernel_config=self.prefill_mlp_config)
            # print(pc)
            pc = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=16,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=56,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=ttnn.experimental.tensor.FusibleActivation.SILU,
                fuse_batch=False,
            )
            w1_out = ttnn.linear(
                x,
                self.w1,
                compute_kernel_config=self.prefill_mlp_config,
                # core_grid=ttnn.CoreGrid(y=8, x=8),
                dtype=ttnn.bfloat8_b,
                # activation="silu",
                program_config=pc,
            )

            print("w1_out shape: ", w1_out.shape)

            pc = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=16,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=56,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )

            w3_out = ttnn.linear(
                x,
                self.w3,
                compute_kernel_config=self.prefill_mlp_config,
                # core_grid=ttnn.CoreGrid(y=8, x=8),
                program_config=pc,
                dtype=ttnn.bfloat8_b,
            )

            print("w3_out shape: ", w3_out.shape)
            w2_in = ttnn.experimental.tensor.mul(w1_out, w3_out)

            pc = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,  # how much inner dim you take each time
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=16,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=16,  # N / TILE_WIDTH / Grid_Size
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            )
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=self.prefill_mlp_config,
                # core_grid=ttnn.CoreGrid(y=8, x=8),
                program_config=pc,
                dtype=ttnn.bfloat16,
            )
            w2_out = ttnn.reshape(w2_out, (1, 1, 8192, 4096))
            return w2_out

        else:
            w1_out = ttnn.experimental.operations.primary.matmul_1d(
                x,
                self.w1,
                program_config=self.model_config["FF1_OUTPUT_PROGCFG"],  # SILu activation fused in the op
                output_mem_config=self.model_config["FF1_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_args.get_compute_kernel_config(),
                output_dtype=ttnn.bfloat8_b,
            )
            w3_out = ttnn.experimental.operations.primary.matmul_1d(
                x,
                self.w3,
                program_config=self.model_config["FF3_OUTPUT_PROGCFG"],
                output_mem_config=self.model_config["FF3_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_args.get_compute_kernel_config(),
                output_dtype=ttnn.bfloat8_b,
            )
            w2_in = ttnn.experimental.tensor.mul(w1_out, w3_out)

            w2_out = ttnn.experimental.operations.primary.matmul_1d(
                w2_in,
                self.w2,
                program_config=self.model_config["FF2_OUTPUT_PROGCFG"],
                output_mem_config=self.model_config["FF2_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_args.get_compute_kernel_config(),
                output_dtype=ttnn.bfloat8_b,
            )

            return w2_out
