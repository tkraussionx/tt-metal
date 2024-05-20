# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh


class TtMoeLayer(torch.nn.Module):
    def __init__(self, device_mesh, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        self.device_mesh = device_mesh
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()

        gate_name = f"layers.{layer_num}.feed_forward.gate.weight"
        self.gates_H8 = ttnn.as_tensor(
            torch.nn.functional.pad(state_dict[gate_name].permute(1, 0), (0, 56), "constant", 0)
            .unsqueeze(0)
            .unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=self.model_config["GATE_W_LAYOUT_TILE"],
            memory_config=self.model_config["GATE_WEIGHTS_MEMCFG"],
            cache_file_name=args.weight_cache_path(dtype) / (gate_name + "_multidevice_padded"),
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        self.num_devices = 8
        self.compute_kernel = args.get_compute_kernel_attn_config()

        reduce_mask_torch = torch.zeros(1, 1, self.args.max_batch_size, self.args.max_batch_size * 8)
        for i in range(self.args.max_batch_size):
            reduce_mask_torch[:, :, i, range(i, self.args.max_batch_size * 8, self.args.max_batch_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        self.reduce_mask = ttnn.to_device(self.reduce_mask, device_mesh)
        self.expert_mask_11BB = ttnn.from_torch(
            torch.cat([torch.full((1, 1, 32, 32), fill_value=i) for i in range(8)], dim=3),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
        )
        top8_mask = torch.full((1, 1, 32, 64), fill_value=torch.finfo(torch.float).min)
        top8_mask[:, :, :, :8] = 0.0
        self.top8_mask_11B_64 = ttnn.from_torch(
            top8_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        top2_mask = torch.full((1, 1, 32, 32), fill_value=torch.finfo(torch.float).min)
        top2_mask[:, :, :, :2] = 0.0
        self.top2_mask_11BB = ttnn.from_torch(
            top2_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        self.x_mem = ttnn.create_sharded_memory_config(
            shape=(32, int(4096 / 32)),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.gates_prg_cfg = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=1,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def forward(self, inputs):
        """
        inputs: (seq_len, 1, batch, hidden_dim)

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        S : seq len (1)
        """
        input_i_1SBH = inputs
        expert_i_HH = self.experts
        # get logits for the experts
        from models.demos.t3000.mixtral8x7b.tt.create_prg_config import create_matmul_program_config

        # prg_cfg = create_matmul_program_config(
        #     input_tensor_a=input_i_1SBH,
        #     input_tensor_b=self.gates_H8,
        #     core_grid=ttnn.CoreGrid(y=4, x=8),
        #     use_1d_systolic_array=True,
        #     activation = None,
        #     compute_kernel_config=self.compute_kernel,
        # )

        # print("prg_cfg", prg_cfg)
        print("started moe layer")
        gate_logits_1SB8 = ttnn.experimental.operations.primary.matmul(
            input_i_1SBH,
            self.gates_H8,
            # memory_config=self.model_config["GATE_MM_OUTPUT_MEMCFG"],
            # compute_kernel_config=self.compute_kernel,
            # use_1d_systolic_array=True,
            # core_grid=ttnn.CoreGrid(y=4, x=8),
            # program_config = self.gates_prg_cfg,
            # dtype=ttnn.bfloat16,
        )
        # print("gates done", gate_logits_1SB8)
        # get weights for top-2 experts
        gate_logits_1SB8 = ttnn.add(gate_logits_1SB8, self.top8_mask_11B_64)
        ttl_topk_values, ttl_topk_indices = ttnn.experimental.operations.primary.topk(gate_logits_1SB8, 32)
        ttl_topk_values = ttnn.add(ttl_topk_values, self.top2_mask_11BB)
        mask_B2 = ttnn.eq(self.expert_mask_11BB, ttl_topk_indices)
        weights_1SB1 = ttnn.sum(ttnn.softmax(ttl_topk_values, dim=-1) * mask_B2, dim=3)
        # print("mask done", mask_B2)
        # MLP and masking
        weights = expert_i_HH(input_i_1SBH)

        results_11BH = ttnn.mul(weights, weights_1SB1)
        # print("MLP done", results_11BH)
        # # convert to bf8 for all-gather perf
        # results_11BH = ttnn.clone(results_11BH, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        results_11BH = ttnn.to_memory_config(results_11BH, self.x_mem)
        # all gather
        output_11BH_gathered = ttnn.all_gather(results_11BH, dim=2, num_links=1)
        # sum on each device
        # print("all gather done", output_11BH_gathered)
        output_11BH_gathered = ttnn.experimental.operations.primary.matmul(self.reduce_mask, output_11BH_gathered)
        return output_11BH_gathered
