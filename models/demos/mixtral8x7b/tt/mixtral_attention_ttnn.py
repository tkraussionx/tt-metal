# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import List
from models.utility_functions import nearest_32


class TtMixtralAttention(torch.nn.Module):
    def __init__(self, devices, state_dict, args, layer_num, dtype):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.model_args = args

        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.sliding_window = args.sliding_window

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        layer_name = f"layers.{layer_num}.attention"
        cache_name = lambda name: self.model_args.weight_cache_path(dtype) / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.wqkv_list = []
        self.wo_list = []
        self.layer_past_list = []

        for i in range(self.num_devices):
            wqkv = ttnn.as_tensor(
                torch.concat(
                    [
                        torch.transpose(
                            torch.chunk(self.state_dict[wq_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                        torch.transpose(
                            torch.chunk(self.state_dict[wk_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                        torch.transpose(
                            torch.chunk(self.state_dict[wv_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                    ],
                    dim=-1,
                ),
                device=self.devices[i],
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name(f"wqkv_{i}_"),
            )
            wo = ttnn.as_tensor(
                torch.transpose(
                    torch.chunk(self.state_dict[wo_str], self.num_devices, dim=-1)[i],
                    -2,
                    -1,
                ),
                device=self.devices[i],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name(f"wo_{i}_"),
            )

            cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    # self.sliding_window,
                    32,  # TODO Update the initial cache size when scaling up
                    self.head_dim,
                )
            )  # torch.finfo(torch.float).min
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    # self.sliding_window,
                    32,  # TODO Update the initial cache size when scaling up
                    self.head_dim,
                )
            )  # torch.finfo(torch.float).min
            layer_past = [cache_k, cache_v]
            layer_past = [
                ttnn.from_torch(lp, device=self.devices[i], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                for lp in layer_past
            ]

            # add to the list
            self.wqkv_list.append(wqkv)
            self.wo_list.append(wo)
            self.layer_past_list.append(layer_past)

        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.compute_kernel_attn = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.core_grid = ttnn.CoreGrid(y=7, x=8)  # self.devices[0].compute_with_storage_grid_size()
        self.batched_attn = True
        self.model_config = args.model_config

    def forward(
        self,
        xs: ttnn.Tensor,
        start_pos: int,
        current_pos: int,
        rot_mats: List[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        """
        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)
        dense_outputs = []
        for i in range(self.num_devices):
            print(f"started device {i}")
            x_11BH = xs[i]
            device = self.devices[i]
            wqkv = self.wqkv_list[i]
            wo = self.wo_list[i]
            layer_past = self.layer_past_list[i]
            rot_mat = rot_mats[i][start_pos]
            ###
            # QKV matmuls
            ###
            xqkv_fused = ttnn.linear(
                x_11BH,
                wqkv,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=self.core_grid,
                compute_kernel_config=self.compute_kernel,
            )

            (
                q_heads_14BD,
                k_heads_11BD,
                v_heads_11BD,
            ) = ttnn.experimental.tensor.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                output_mem_config=ttnn.L1_MEMORY_CONFIG,
            )

            # "1D mcast for in0 or in1 is not implemented yet." - tt::tt_metal::matmul_multi_core_reuse_mcast_2d_optimized
            ###
            # Rotary embeddings
            ###
            q_heads_14BD = ttnn.linear(
                q_heads_14BD,
                rot_mat,
                core_grid=self.core_grid,
                use_1d_systolic_array=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            k_heads_11BD = ttnn.linear(
                k_heads_11BD,
                rot_mat,
                core_grid=self.core_grid,
                use_1d_systolic_array=True,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            ###
            # KV update
            ###
            layer_slice = min((start_pos + 1), self.sliding_window)
            keys_B1PD = layer_past[0]
            values_B1PD = layer_past[1]
            ttnn.kv_cache.update_cache_for_token_(keys_B1PD, k_heads_11BD, current_pos)
            ttnn.kv_cache.update_cache_for_token_(values_B1PD, v_heads_11BD, current_pos)
            self.layer_past_list[i] = [keys_B1PD, values_B1PD]
            keys_B1PD = keys_B1PD[:, :, :padded_layer_past_len, :]
            values_B1PD = values_B1PD[:, :, :layer_slice, :]

            ###
            # Attention
            ###

            keys_B1DP = ttnn.permute(keys_B1PD, (0, 1, 3, 2))

            if self.batched_attn:
                # transpose and shard q head
                q_heads_B14D = ttnn.permute(q_heads_14BD, (2, 0, 1, 3))
                q_heads_B14D = ttnn.to_memory_config(q_heads_B14D, self.model_config["Q_TRANSPOSE_MEMCFG"])

                # shard keys
                k_cache_memcfg = self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"](padded_layer_past_len)
                keys_B1DP = ttnn.to_memory_config(keys_B1DP, k_cache_memcfg)

                # create out cfg
                attn_output_memcfg = self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"](padded_layer_past_len)

                attn_B14P = ttnn.matmul(
                    q_heads_B14D,
                    keys_B1DP,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG,  # attn_output_memcfg
                    core_grid=self.core_grid,
                    compute_kernel_config=self.compute_kernel_attn,
                )
                attn_B14P = attn_B14P[:, :, :, :layer_slice]
                attn_B14P = ttnn.softmax(attn_B14P * (self.head_dim**-0.5), dim=-1, memory_config=attn_output_memcfg)
            else:
                attn_14BP = ttnn.experimental.operations.primary.transformers.attn_matmul(
                    q_heads_14BD,
                    keys_B1DP,
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                    # fp32_dest_acc_en=True,
                    # packer_l1_acc=True,
                    # output_mem_config=ttnn.L1_MEMORY_CONFIG,  # ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1), #self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    output_dtype=ttnn.bfloat16,  # Must be BFLOAT16
                )
                attn_14BP = attn_14BP[:, :, :, :layer_slice]
                attn_14BP = ttnn.softmax(attn_14BP * (self.head_dim**-0.5), dim=-1)

            if self.batched_attn:
                # shard values
                v_cache_memcfg = self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"](padded_layer_past_len)
                values_B1PD = ttnn.to_memory_config(values_B1PD, v_cache_memcfg)

                attn_output_B14D = ttnn.matmul(
                    attn_B14P,
                    values_B1PD,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    core_grid=self.core_grid,
                    compute_kernel_config=self.compute_kernel_attn,
                )
                attn_output_14BD = ttnn.permute(attn_output_B14D, (1, 2, 0, 3))
            else:
                attn_output_14BD = ttnn.experimental.operations.primary.transformers.attn_matmul(
                    attn_14BP,
                    values_B1PD,
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                    output_dtype=ttnn.bfloat16,
                )  # seqlen, n_heads, batch, dhead

            attn_output_11BH = ttnn.experimental.tensor.nlp_concat_heads(
                attn_output_14BD, output_mem_config=ttnn.L1_MEMORY_CONFIG
            )

            ###
            # Output matmul
            ###
            dense_out_11BH = ttnn.linear(
                attn_output_11BH,
                wo,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                core_grid=self.core_grid,
                compute_kernel_config=self.compute_kernel,
            )

            dense_outputs.append(dense_out_11BH)
            print(f"finished device {i}")

        # return the sum of the outputs
        if len(dense_outputs) > 1:
            dense_outputs = ttnn.experimental.tensor.all_gather(dense_outputs, dim=1, num_links=1)
            for i in range(len(dense_outputs)):
                dense_outputs[i] = ttnn.experimental.tensor.sum(dense_outputs[i], dim=1)
            print("done reduce")
            return dense_outputs
        else:
            return dense_outputs
