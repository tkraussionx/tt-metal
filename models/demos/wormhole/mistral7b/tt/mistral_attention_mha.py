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
        self.sliding_window = configuration.kv_seq_len
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
                    1,
                    self.kv_seq_len,  # self.sliding_window,
                    self.head_dim * self.n_kv_heads // self.num_devices,
                )
            )
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    1,
                    self.kv_seq_len,  # self.sliding_window,
                    self.head_dim * self.n_kv_heads // self.num_devices,
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
        self.head_dims = [
            ttnn.from_torch(
                torch.ones(self.n_heads, 1, self.max_batch_size, self.head_dim)
                * (self.head_dim**-0.5),  # [seqlen, n_heads, bsz, head_dim] [1,32,32,128]
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(self.num_devices)
        ]

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
        expand_D_8D_torch = torch.eye(128, 128).repeat(1, 1, 1, 8)
        self.expand_D_8D = [
            ttnn.from_torch(
                expand_D_8D_torch,
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(len(devices))
        ]

        reduce_8D_D_torch = torch.eye(128, 128).repeat(1, 1, 8, 1)
        self.reduce_8D_D = [
            ttnn.from_torch(
                reduce_8D_D_torch,
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(len(devices))
        ]

        mask_Q_8D_torch = torch.zeros(32, 1, 32, 8 * 128)
        for j in range(8):
            mask_Q_8D_torch[:, :, j * 4 : (j + 1) * 4, j * 128 : (j + 1) * 128] = 1
        self.mask_Q_8D = [
            ttnn.from_torch(
                mask_Q_8D_torch,
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(len(devices))
        ]
        self.expand_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.experimental.tensor.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=4,
            per_core_N=4,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.reduce_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.experimental.tensor.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.attn_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.experimental.tensor.CoreCoord(8, 4),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=32,
        )
        self.compute_kernel_config_attn = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.attention_grid = ttnn.experimental.tensor.CoreCoord(8, 4)

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
        layer_slice = min((self.start_pos), self.sliding_window)

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
            head_dim = self.head_dims[i]
            expand_D_8D = self.expand_D_8D[i]
            reduce_8D_D = self.reduce_8D_D[i]
            mask_Q_8D = self.mask_Q_8D[i]

            ###
            # QKV matmuls
            ###
            xqkv_fused = ttnn.linear(
                x,
                wqkv,
                memory_config=self.model_config["XQKV_MM_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                dtype=self.dtype,
                core_grid=self.grid_size,
            )

            ###
            # Reshape and rotary embeddings
            ###
            (
                q_heads_Q1BD,  # q_heads_pre_rot_Q1BD,
                k_heads_11BC,  # k_heads_pre_rot_11BC,
                v_heads_11BC,
            ) = ttnn.experimental.tensor.nlp_create_qkv_heads_mistral(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            ttnn.deallocate(xqkv_fused)

            # Update rotary matrix on device
            # rotary_mat = self.rot_mat[current_pos]

            # q_heads_Q1BD = ttnn.linear(
            #     q_heads_pre_rot_Q1BD,
            #     rotary_mat,
            #     program_config=self.q_heads_program_config,
            #     memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
            #     compute_kernel_config=self.compute_kernel_config,
            #     dtype=self.dtype,
            # )

            # k_heads_11BC = ttnn.linear(
            #     k_heads_pre_rot_11BC,
            #     rotary_mat,
            #     program_config=self.k_heads_program_config,
            #     memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
            #     compute_kernel_config=self.compute_kernel_config,
            #     dtype=self.dtype,
            # )

            # ttnn.deallocate(q_heads_pre_rot_Q1BD)
            # ttnn.deallocate(k_heads_pre_rot_11BC)

            ###
            # KV update
            ###
            keys_B1WC = layer_past[0]
            values_B1WC = layer_past[1]
            # ttnn.experimental.tensor.update_cache(keys_B1WC , k_heads_11BC, current_pos)  # self.current)
            ttnn.deallocate(k_heads_11BC)
            # ttnn.experimental.tensor.update_cache(values_B1WC , v_heads_11BC, current_pos)  # self.current)
            self.layer_past_list[i] = [keys_B1WC, values_B1WC]

            ttnn.deallocate(v_heads_11BC)

            ###
            # Attention
            ###
            keys_sliced_B1PC = keys_B1WC[:, :, :padded_layer_past_len, :]
            keys_sliced_B1CP = ttnn.experimental.tensor.transpose(keys_sliced_B1PC, 2, 3)
            ttnn.deallocate(keys_sliced_B1PC)

            q_heads_Q1BD = q_heads_Q1BD * head_dim
            # q_heads_Q1BD = ttnn.clone(
            #         q_heads_Q1BD, dtype=ttnn.bfloat16, memory_config=self.model_config["KV_UNPAD_OUTPUT_MEMCFG"]
            #     )
            # q_heads_B1QD = ttnn.experimental.tensor.transpose(q_heads_Q1BD, 0, 2)
            q_heads_B1QD = q_heads_Q1BD
            q_heads_B1QC = (
                ttnn.matmul(
                    q_heads_B1QD,
                    expand_D_8D,
                    program_config=self.expand_program_config,
                    compute_kernel_config=self.compute_kernel_config,
                    memory_config=self.model_config["KV_UNPAD_OUTPUT_MEMCFG"],
                )
                * mask_Q_8D
            )

            # scores matmul
            attn_B1QP = ttnn.matmul(
                q_heads_B1QC,
                keys_sliced_B1CP,
                core_grid=ttnn.CoreGrid(y=4, x=8),
                compute_kernel_config=self.compute_kernel_config_attn,
                dtype=ttnn.bfloat16,
                memory_config=self.model_config["KV_UNPAD_OUTPUT_MEMCFG"],
            )
            ttnn.deallocate(keys_sliced_B1CP)
            ttnn.deallocate(keys_sliced_B1CP)

            attn_sliced_B1QP = attn_B1QP[:, :, :, :layer_slice]
            attn_sliced_B1QP = ttnn.softmax(attn_sliced_B1QP, dim=-1)

            values_sliced_B1PC = values_B1WC[:, :, :layer_slice, :]

            attn_output_B1QC = ttnn.matmul(
                attn_sliced_B1QP,
                values_sliced_B1PC,
                program_config=self.attn_program_config,
                memory_config=self.model_config["QKV_MM_OUTPUT_MEMCFG"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config_attn,
            )

            ttnn.deallocate(attn_sliced_B1QP)
            ttnn.deallocate(values_sliced_B1PC)

            # reduce and reshape
            attn_output_B1QD = ttnn.matmul(
                attn_output_B1QC * mask_Q_8D,
                reduce_8D_D,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self.reduce_program_config,
                memory_config=self.model_config["QKV_MM_OUTPUT_MEMCFG"],
            )

            # concat heads
            attn_output_11BH = ttnn.experimental.tensor.reshape(attn_output_B1QD, 1, 1, 32, 4096)

            ttnn.deallocate(attn_output_B1QD)

            dense_out = ttnn.linear(
                attn_output_11BH,
                wo,
                memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=self.grid_size,
            )

            ttnn.deallocate(attn_output_11BH)
            dense_outputs.append(dense_out)

        # return the sum of the outputs
        if len(dense_outputs) > 1:
            return None  # tt_all_reduce(dense_outputs)
        else:
            return dense_outputs
