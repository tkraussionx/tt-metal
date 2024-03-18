# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
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
        tt_cos_cached,
        tt_sin_cached,
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
        self.sliding_window = configuration.sliding_window

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

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

        self.kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

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
                cache_file_name=cache_name("wqkv"),
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
                cache_file_name=cache_name("wo"),
            )

            cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    # self.sliding_window,
                    512,  # TODO Update the initial cache size when scaling up (Target = window size == 4096)
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    # self.sliding_window,
                    512,  # TODO Update the initial cache size when scaling up (Target = window size == 4096)
                    self.head_dim,
                )
            )
            layer_past = [cache_k, cache_v]
            layer_past = [
                ttnn.from_torch(lp, device=self.devices[i], layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
                for lp in layer_past
            ]

            # add to the list
            self.wqkv_list.append(wqkv)
            self.wo_list.append(wo)
            self.layer_past_list.append(layer_past)
        self.tt_sin_cached = tt_sin_cached
        self.tt_cos_cached = tt_cos_cached
        self.grid = ttnn.CoreGrid(x=8, y=8)

    def forward(
        self,
        xs: List[ttnn.Tensor],
        start_pos: int,
        current_pos: int,
        attn_masks: List[ttnn.Tensor],
        rot_mats: List[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        attn_mask: (seq_len, n_heads, batch, cache_len + seqlen
        """
        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)
        dense_outputs = []
        for i in range(self.num_devices):
            x = xs[i]
            attn_mask = attn_masks[i]
            device = self.devices[i]
            wqkv = self.wqkv_list[i]
            wo = self.wo_list[i]
            layer_past = self.layer_past_list[i]
            rot_mat = rot_mats[i]
            ###
            # QKV matmuls
            ###
            xqkv_fused = ttnn.linear(
                x,
                wqkv,
                core_grid=self.grid,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.kernel_config,
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
                output_mem_config=ttnn.DRAM_MEMORY_CONFIG,  # ttnn.L1_MEMORY_CONFIG,
            )

            ttnn.deallocate(xqkv_fused)

            # q_heads = ttnn.experimental.tensor.rotary_embedding(
            # q_heads = ttnn.transformer.rotary_embedding(
            #     q_heads, self.tt_cos_cached[i], self.tt_sin_cached[i], start_pos, memory_config=ttnn.DRAM_MEMORY_CONFIG
            # )
            q_heads = ttnn.linear(
                q_heads,
                rot_mat,
                core_grid=self.grid,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.kernel_config,
            )

            # k_heads = ttnn.experimental.tensor.rotary_embedding(
            # k_heads = ttnn.transformer.rotary_embedding(
            #     k_heads, self.tt_cos_cached[i], self.tt_sin_cached[i], start_pos, memory_config=ttnn.DRAM_MEMORY_CONFIG
            # )
            k_heads = ttnn.linear(
                k_heads,
                rot_mat,
                core_grid=self.grid,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.kernel_config,
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

            ttnn.deallocate(k_heads)
            ttnn.deallocate(v_heads)

            keys = ttnn.experimental.tensor.unpad(
                layer_past[0],
                [0, 0, 0, 0],
                [
                    self.max_batch_size - 1,
                    self.n_local_kv_heads - 1,
                    padded_layer_past_len - 1,
                    self.head_dim - 1,
                ],
                output_mem_config=ttnn.L1_MEMORY_CONFIG,
            )
            values = ttnn.experimental.tensor.unpad(
                layer_past[1],
                [0, 0, 0, 0],
                [
                    self.max_batch_size - 1,
                    self.n_local_kv_heads - 1,
                    padded_layer_past_len - 1,
                    self.head_dim - 1,
                ],
                output_mem_config=ttnn.L1_MEMORY_CONFIG,
            )

            ###
            # Attention
            ###

            keys = ttnn.permute(keys, (0, 1, 3, 2))  #  [batch, num_kv_heads, dhead, cache_len + seqlen]

            """
            q_heads = ttnn.to_memory_config(
                q_heads,
                memory_config=ttnn.create_sharded_memory_config(
                    (32, 128),
                    ttnn.CoreGrid(8, 4),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
            )  # [seqlen, n_heads, bsz, head_dim]

            # dynamic sharding
            keys = ttnn.to_memory_config(
                keys,
                memory_config=ttnn.create_sharded_memory_config(
                    (8 * 1 * 128, padded_layer_past_len),
                    ttnn.CoreGrid(8, 4),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                    use_height_and_width_as_shard_shape=True,
                ),
            )
            """

            attn = ttnn.experimental.operations.primary.transformers.group_attn_matmul(
                q_heads,
                keys,
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                output_mem_config=ttnn.L1_MEMORY_CONFIG,
                output_dtype=ttnn.bfloat16,  # Must be BFLOAT16
            )  # seqlen, n_heads, batch, cache_len + seqlen

            attn = ttnn.transformer.attention_softmax_(attn, head_size=self.head_dim, attention_mask=attn_mask)

            """
            attn = ttnn.to_memory_config(attn, memory_config=ttnn.create_sharded_memory_config(
                (32, padded_layer_past_len),
                ttnn.CoreGrid(8, 4),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ))

            values = ttnn.to_memory_config(values, memory_config=ttnn.create_sharded_memory_config(
                (1 * 8 * padded_layer_past_len, 128),
                ttnn.CoreGrid(8, 4),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
                use_height_and_width_as_shard_shape=True,
            ))
            """

            attn_output = ttnn.experimental.operations.primary.transformers.group_attn_matmul(
                attn,
                values,
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                # output_mem_config=ttnn.L1_MEMORY_CONFIG,
                output_dtype=ttnn.bfloat16,
            )  # seqlen, n_heads, batch, dhead

            ttnn.deallocate(attn)
            ttnn.deallocate(q_heads)

            attn_output = ttnn.transformer.concatenate_heads(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
            # seqlen, 1, batch, hidden_size

            dense_out = ttnn.linear(
                attn_output,
                wo,
                core_grid=self.grid,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.kernel_config,
            )  # seqlen, 1, batch, hidden_size

            dense_outputs.append(dense_out)

        # return the sum of the outputs
        if len(dense_outputs) > 1:
            return None  # tt_all_reduce(dense_outputs)
        else:
            return dense_outputs
