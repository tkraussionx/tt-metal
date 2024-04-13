# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch
from torch import nn

import ttnn
from models.utility_functions import (
    nearest_32,
)


class TtMistralAttention(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        configuration,
        rot_mat,
        start_pos,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.start_pos = start_pos

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        self.kv_seq_len = configuration.kv_seq_len
        self.sliding_window = (
            configuration.kv_seq_len
        )  # TODO Change sliding window back to configuration.sliding_window
        self.grid_size = configuration.max_grid_size

        self.model_config = configuration.get_model_config()
        self.compute_kernel_config = configuration.get_compute_kernel_config()

        self.rot_mat = rot_mat  # Rotational matrix in the form of a list of 8K tensors [1,1,head_dim,head_dim] for positional embedding on device

        layer_name = f"layers.{layer_num}.attention"
        cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

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
                memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                cache_file_name=cache_name("wqkv"),
            )

            # That's interleaving the Q, K and V weights according to their groups; we can use this in combination with the sharded nlp_create_qkv_heads (attention: needs as many num cores as we have groups!)
            #
            # for group_i in range(self.n_kv_heads):
            #     ### Fused QKV Weights
            #     # Chunk weights
            #     wq_chunks = torch.chunk(self.state_dict[wq_str], self.n_heads, dim=0)
            #     wk_chunks = torch.chunk(self.state_dict[wk_str], self.n_kv_heads, dim=0)
            #     wv_chunks = torch.chunk(self.state_dict[wv_str], self.n_kv_heads, dim=0)

            #     # Select chunks for the current device
            #     wq_selected = torch.cat(wq_chunks[group_i * self.n_local_heads : (group_i + 1) * self.n_local_heads], dim=0)
            #     wk_selected = torch.cat(wk_chunks[group_i * self.n_local_kv_heads : (group_i + 1) * self.n_local_kv_heads], dim=0)
            #     wv_selected = torch.cat(wv_chunks[group_i * self.n_local_kv_heads : (group_i + 1) * self.n_local_kv_heads], dim=0)

            #     # Transpose the selected chunks
            #     wq = torch.transpose(wq_selected, -2, -1)
            #     wk = torch.transpose(wk_selected, -2, -1)
            #     wv = torch.transpose(wv_selected, -2, -1)

            #     # Create interleaved qkv list
            #     n_repeat = self.n_heads // self.n_kv_heads
            #     qkv_interleaved = [
            #         [
            #             wq[..., j * n_repeat * self.head_dim : (j + 1) * n_repeat * self.head_dim],
            #             wk[..., j * self.head_dim : (j + 1) * self.head_dim],
            #             wv[..., j * self.head_dim : (j + 1) * self.head_dim],
            #         ]
            #         for j in range(self.n_local_kv_heads)
            #     ]
            #     qkv_interleaved = [item for sublist in qkv_interleaved for item in sublist]

            #     # Concatenate Q, K, V for the current group
            #     qkv = torch.cat(qkv_interleaved, dim=-1)

            #     wqkv = ttnn.as_tensor(
            #         qkv,
            #         device=self.devices[i],
            #         memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            #         dtype=self.dtype,
            #         layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            #         cache_file_name=cache_name("wqkv_interleaved"),
            #     )
            #     self.wqkv_list.append(wqkv)

            wo = ttnn.as_tensor(
                torch.transpose(
                    torch.chunk(self.state_dict[wo_str], self.num_devices, dim=-1)[i],
                    -2,
                    -1,
                ),
                device=self.devices[i],
                memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
                dtype=self.dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                cache_file_name=cache_name("wo"),
            )

            cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    self.kv_seq_len,  # self.sliding_window,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    self.kv_seq_len,  # self.sliding_window,
                    self.head_dim,
                )
            )
            layer_past = [cache_k, cache_v]
            layer_past = [
                ttnn.from_torch(
                    lp, device=self.devices[i], layout=self.model_config["ATTN_W_LAYOUT_TILE"], dtype=self.dtype
                )
                for lp in layer_past
            ]

            # add to the list
            self.wqkv_list.append(wqkv)
            self.wo_list.append(wo)
            self.layer_past_list.append(layer_past)

        # Pre-scaled head dimension (for softmax) to avoid fallbacking to host
        self.softmax_scale_factors = [
            ttnn.from_torch(
                torch.ones(1, self.n_heads, self.max_batch_size, self.head_dim)
                * (self.head_dim**-0.5),  # [seqlen, n_heads, bsz, head_dim] [1,32,32,128]
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            for i in range(self.num_devices)
        ]

        self.wqkv_program_config = ttnn.operations.matmul.create_matmul_1d_systolic_array_program_config(
            input_shape_a=ttnn.Shape([1, 1, self.max_batch_size, self.hidden_size]),
            input_shape_b=self.wqkv_list[0].shape,
            core_grid=self.grid_size,
            fp32_dst=self.compute_kernel_config.fp32_dest_acc_en,
        )
        self.dense_program_config = ttnn.operations.matmul.create_matmul_1d_systolic_array_program_config(
            input_shape_a=ttnn.Shape([1, 1, self.max_batch_size, self.hidden_size]),
            input_shape_b=self.wo_list[0].shape,
            core_grid=self.grid_size,
            fp32_dst=self.compute_kernel_config.fp32_dest_acc_en,
        )
        self.q_heads_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.experimental.tensor.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )
        self.k_heads_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.experimental.tensor.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )

    def forward(
        self,
        xs: List[ttnn.Tensor],
        current_pos: int,
        attn_masks: Optional[List[ttnn.Tensor]] = None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        attn_mask: (seq_len, n_heads, batch, cache_len + seqlen
        """
        self.start_pos += 1
        padded_layer_past_len = min(nearest_32(self.start_pos), self.sliding_window)
        layer_slice = min(self.start_pos, self.sliding_window)

        dense_outputs = []
        for i in range(self.num_devices):
            x = xs[i]
            if attn_masks is not None:
                attn_mask = attn_masks[i]
            else:
                attn_mask = None
            device = self.devices[i]
            wqkv = self.wqkv_list[i]
            wo = self.wo_list[i]
            layer_past = self.layer_past_list[i]
            softmax_scale_factor = self.softmax_scale_factors[i]

            ###
            # QKV matmuls
            ###
            xqkv_fused = ttnn.linear(
                x,
                wqkv,
                program_config=self.wqkv_program_config,
                memory_config=self.model_config["XQKV_MM_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                dtype=self.dtype,
            )

            ###
            # Reshape and rotary embeddings
            ###
            (
                q_heads,  # [seqlen, n_heads, bsz, head_dim]
                k_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
                v_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
            ) = ttnn.experimental.tensor.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                output_mem_config=self.model_config["QKV_HEADS_OUTPUT_MEMCFG"],
            )

            # Update rotary matrix on device
            rotary_mat = self.rot_mat[current_pos]

            queries = ttnn.linear(
                q_heads,
                rotary_mat,
                program_config=self.q_heads_program_config,
                memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
            )

            k_heads = ttnn.linear(
                k_heads,
                rotary_mat,
                program_config=self.k_heads_program_config,
                memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
            )

            ###
            # KV update
            ###
            keys = layer_past[0]
            values = layer_past[1]

            # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
            # v_heads [seqlen, n_kv_heads, bsz, head_dim]
            # keys, [max_batch_size, n_kv_heads // self.num_devices, sliding_window, head_dim]
            ttnn.experimental.tensor.update_cache(keys, k_heads, current_pos)  # self.current)
            ttnn.experimental.tensor.update_cache(values, v_heads, current_pos)  # self.current)
            self.layer_past_list[i] = [keys, values]

            keys = keys[:, :, :padded_layer_past_len, :]  # [batch, num_kv_heads, cache_len + seqlen, dhead]
            values = values[:, :, :layer_slice, :]  # [batch, num_kv_heads, cache_len + seqlen, dhead]

            self.scaled_dot_product_attention(
                queries, keys, values, softmax_scale_factor, padded_layer_past_len, layer_slice
            )

            attn_output = ttnn.transformer.concatenate_heads(
                attn_output, memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"]
            )
            # seqlen, 1, batch, hidden_size

            dense_out = ttnn.linear(
                attn_output,
                wo,
                program_config=self.dense_program_config,
                memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
            )  # seqlen, 1, batch, hidden_size

            dense_outputs.append(dense_out)

        # return the sum of the outputs
        if len(dense_outputs) > 1:
            return None  # tt_all_reduce(dense_outputs)
        else:
            return dense_outputs

    def scaled_dot_product_attention(
        self, queries, keys, values, softmax_scale_factor, padded_layer_past_len, layer_slice
    ):
        queries = queries * softmax_scale_factor  # Scale queries instead of QK before softmax

        print(f"queries.shape: {queries.shape}, values.shape: {values.shape}")

        # To use a normal matmul we need to separate both the users (batch dim) and the groups (kv_heads dim)
        # We're going to use height sharding in the batch rows to place each user on one core
        # And then we're going to iterate through the 8 groups of kv_heads
        # If we do this we can use a normal matmul for the Q*K^T instead of using group_attn_matmul which can't
        # handle sequences longer than ~384 tokens and is suuuuper slow
        # Users (batch): 32
        # Groups (kv_heads): 8
        # Q heads: 32
        # head dim: 128
        # hidden dim: 4096

        # Batched attention with slicing
        # Q: [seqlen, n_heads, batch, head_dim]
        # K: [seqlen, n_kv_heads, batch, head_dim]
        # V: [seqlen, n_kv_heads, batch, head_dim]
        queries_to_slice_in_num_heads = ttnn.permute(
            queries, (1, 2, 0, 3)
        )  # [1, n_heads, batch, head_dim] -> [n_heads, batch, 1, head_dim]

        keys_transposed_to_slice_in_num_heads = ttnn.permute(
            keys, (1, 2, 3, 0)
        )  #  [seqlen, n_kv_heads, batch, head_dim] -> [num_kv_heads, batch, dhead, seqlen]

        values_to_slice_in_num_heads = ttnn.permute(
            keys, (1, 2, 0, 3)
        )  #  [seqlen, n_kv_heads, batch, head_dim] -> [num_kv_heads, batch, seqlen, dhead]

        for slice_i in range(self.n_local_kv_heads):
            # each slice is [1, BS, head_dim, padded_layer_past_len]
            # then we shard it on 32 cores along the users, each shard is [1, 1, head_dim, padded_layer_past_len]
            key_transposed_per_group = ttnn.experimental.tensor.interleaved_to_sharded_partial(
                keys_transposed_to_slice_in_num_heads,
                (8, 4),
                [
                    self.head_dim,
                    padded_layer_past_len,
                ],  # shard size per core
                self.n_local_kv_heads,  # num_slices
                slice_i,  # slice_index
                ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            )

            value_per_group = ttnn.experimental.tensor.interleaved_to_sharded_partial(
                values_to_slice_in_num_heads,
                (8, 4),
                [
                    padded_layer_past_len,
                    self.head_dim,
                ],  # shard size per core
                self.n_local_kv_heads,  # num_slices
                slice_i,  # slice_index
                ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            )

            query_per_group = ttnn.experimental.tensor.interleaved_to_sharded_partial(
                queries_to_slice_in_num_heads,
                (8, 4),
                [
                    1,
                    self.head_dim,
                ],  # shard size per core
                self.n_local_kv_heads,  # num_slices
                slice_i,  # slice_index
                ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
            )

            # Q*KˆT
            shard_spec_32_cores_grid = ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(
                        ttnn.experimental.tensor.CoreCoord(0, 0),
                        ttnn.experimental.tensor.CoreCoord(7, 3),
                    ),
                }
            )
            qkt_out_memcfg = ttnn.experimental.tensor.MemoryConfig(
                ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.experimental.tensor.BufferType.L1,
                ttnn.experimental.tensor.ShardSpec(
                    shard_spec_32_cores_grid,  # Sharded on batch dim
                    [
                        self.n_local_heads,  # Each core has all the heads
                        padded_layer_past_len,  # Dynamic (padded seqlen)
                    ],
                    ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
            qkt_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=[8, 4],
                in0_block_w=self.head_dim // 32,  # HEAD_DIM // TILE_SIZE
                out_subblock_h=1,  # (TODO: Maximize)
                out_subblock_w=1,  # (TODO: Maximize)
                per_core_M=self.n_local_heads // 32,  # N_HEADS_PADDED // TILE_SIZE,
                per_core_N=padded_layer_past_len // 32,  # kv cache length // TILE_SIZE
            )
            attn = ttnn.experimental.operations.primary.matmul(
                query_per_group,
                key_transposed_per_group,
                program_config=qkt_program_config,
                output_mem_config=qkt_out_memcfg,
                compute_kernel_config=self.compute_kernel_config,
                output_dtype=ttnn.bfloat16,  # Force bfloat16 for higher accuracy
            )  # batch, n_heads, batch, cache_len + seqlen

            # Softmax
            attn = attn[:, :, :, :layer_slice]
            # TODO: do we need to do something for softmax to work with sharded inputs? what about tt_lib.operations.primary.transformers.scale_mask_softmax_in_place - is that the better choice?
            attn = ttnn.softmax(
                attn, dim=-1
            )  # No need to scale since we've done this above already (attn * (self.head_dim**-0.5))

            # Scores * V MM
            scores_v_memcfg = ttnn.experimental.tensor.MemoryConfig(
                ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.experimental.tensor.BufferType.L1,
                ttnn.experimental.tensor.ShardSpec(
                    shard_spec_32_cores_grid,
                    [
                        self.n_local_heads,
                        self.head_dim,  # head dim
                    ],
                    ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
            scores_v_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=[8, 4],
                in0_block_w=padded_layer_past_len // 32,  # SEQ_LEN // TILE_SIZE (dynamic)
                out_subblock_h=1,  # (TODO: Maximize)
                out_subblock_w=1,  # (TODO: Maximize)
                per_core_M=4
                // 32,  # N_HEADS_PADDED // TILE_SIZE, # TODO <-------- we can't do 4 qeuery heads, would need to pad this!! That wouldn't make any sense.
                per_core_N=self.head_dim // 32,  # HEAD_DIM // TILE_SIZE
            )
            attn_output = ttnn.experimental.operations.primary.matmul(
                attn,
                value_per_group,
                program_config=scores_v_program_config,
                output_mem_config=scores_v_memcfg,
                compute_kernel_config=self.compute_kernel_config,
                output_dtype=ttnn.bfloat16,  # Force bfloat16 for higher accuracy
            )  # batch, seqlen (=1), n_heads, dhead

            # Convert attn output back to interleaved
            attn_output_per_group = ttnn.experimental.tensor.sharded_to_interleaved(
                attn_output,
                output_mem_config=self.model_config["L1_MEMCFG"],
            )
            ttnn.experimental.tensor.sharded_to_interleaved_partial(
                attn_output_per_group,
                attn_output,  # TODO: pre-allocate this in dram!
                self.n_local_kv_heads,  # num_slices
                slice_i,  # slice_index
                self.model_config["DRAM_MEMCFG"],
            )

        return attn_output
