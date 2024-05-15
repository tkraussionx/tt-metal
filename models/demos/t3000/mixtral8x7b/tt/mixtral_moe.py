# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh


def top_2_tt(gate_logits_1SB8, expert_mask_11B2, top2_mask, device_mesh):
    ttl_topk_values, ttl_topk_indices = ttnn.experimental.operations.primary.topk(gate_logits_1SB8, 32)
    ttl_topk_values = ttl_topk_values + top2_mask

    # print(ttnn.to_torch(ttl_topk_indices, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))[0])
    # print(ttnn.to_torch(expert_mask_11B2, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))[0])

    mask_B2 = ttnn.eq(expert_mask_11B2, ttl_topk_indices)
    # print(ttnn.to_torch(mask_B2, mesh_composer=ConcatMeshToTensor(device_mesh, dim=0))[0])

    weights = ttnn.sum(ttnn.softmax(ttl_topk_values, dim=-1) * mask_B2, dim=-1)
    return weights


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

        # self.gates_program_config = ttnn.operations.matmul.create_matmul_1d_systolic_array_program_config(
        #     input_shape_a=ttnn.Shape([1, 1, args.max_batch_size, args.dim]),
        #     input_shape_b=ttnn.Shape([1, 1, args.dim, 64]),
        #     core_grid=ttnn.CoreGrid(y=1, x=8),
        #     fp32_dst=self.compute_kernel.fp32_dest_acc_en,
        # )
        self.expert_mask_11B2 = ttnn.from_torch(
            torch.cat([torch.full((1, 1, 32, 64), fill_value=i) for i in range(8)], dim=3),
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

        top2_mask = torch.full((1, 1, 32, 64), fill_value=torch.finfo(torch.float).min)
        top2_mask[:, :, :, :2] = 0.0
        self.top2_mask_11B_64 = ttnn.from_torch(
            top2_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

    def forward(self, input_i_1SBH):
        """
        inputs: (seq_len, 1, batch, hidden_dim)

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        S : seq len (1)
        """
        expert_i_HH = self.experts
        # get logits for the experts
        # gate_logits_1SB8 = ttnn.experimental.operations.primary.matmul_1d(
        #     input_i_1SBH,
        #     self.gates_H8,
        #     output_mem_config=self.model_config["GATE_MM_OUTPUT_MEMCFG"],
        #     compute_kernel_config=self.compute_kernel,
        #     #use_1d_systolic_array=True,
        #     #core_grid=ttnn.CoreGrid(y=1, x=8),
        #     program_config = self.gates_program_config,
        #     output_dtype=ttnn.bfloat16,
        # )
        # #gate_logits_1SB8 = ttnn.to_device(gate_logits_1SB8, self.device_mesh)

        # gate_logits_1SB8=ttnn.add(gate_logits_1SB8,self.top8_mask_11B_64)
        # weights_1SB1 = top_2_tt(gate_logits_1SB8, self.expert_mask_11B2, self.top2_mask_11B_64, self.device_mesh)
        # #MLP and masking
        weights_1SBH = input_i_1SBH  # expert_i_HH(input_i_1SBH)

        results_11BH = weights_1SBH  # ttnn.mul(weights_1SBH, weights_1SB1)

        # convert to bf8 for all-gather perf
        results_11BH = ttnn.clone(results_11BH, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        results_11BH = ttnn.experimental.tensor.interleaved_to_sharded(
            results_11BH, sharded_mem_config=self.model_config["SHARDED_NORM_OUTPUT_MEMCFG"]
        )

        # all gather
        output_11BH_gathered = ttnn.all_gather(results_11BH, dim=2, num_links=1)
        # sum on each device
        # output_11BH_gathered = ttnn.experimental.operations.primary.matmul(self.reduce_mask, output_11BH_gathered) #, core_grid=ttnn.CoreGrid(y=1, x=8), use_1d_systolic_array=True)
        return output_11BH_gathered
