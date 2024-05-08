# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from time import time
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
        self.args = configuration
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
        self.head_dims = [
            ttnn.from_torch(
                torch.ones(1, self.n_heads, self.max_batch_size, self.head_dim)
                * (self.head_dim**-0.5),  # [seqlen, n_heads, bsz, head_dim] [1,32,32,128]
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(self.num_devices)
        ]
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

        mask_Q_8D_torch = torch.zeros(1, 32, 32, 8 * 128)
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

        self.compute_kernel_config_attn = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
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

        self.scores_program_config = lambda p: ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.experimental.tensor.CoreCoord(8, 4),
            in0_block_w=32,
            out_subblock_h=1,
            out_subblock_w=p,
            per_core_M=1,
            per_core_N=p,
        )
        self.attn_program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.experimental.tensor.CoreCoord(8, 4),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=32,
        )

    def forward(
        self,
        attn_norms: List[ttnn.Tensor],
        current_pos: int,
        attn_masks: Optional[List[ttnn.Tensor]] = None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        attn_mask: (seq_len, n_heads, batch, cache_len + seqlen
        """
        self.start_pos += 1
        padded_layer_past_len = min(nearest_32(current_pos + 1), self.sliding_window)
        layer_slice = min((current_pos + 1), self.sliding_window)

        dense_outputs = []
        for i in range(self.num_devices):
            attn_norm = attn_norms[i]
            device = self.devices[i]
            wqkv = self.wqkv_list[i]
            wo = self.wo_list[i]
            layer_past = self.layer_past_list[i]
            head_dim_1QBD = self.head_dims[i]
            expand_D_8D = self.expand_D_8D[i]
            reduce_8D_D = self.reduce_8D_D[i]
            mask_Q_8D = self.mask_Q_8D[i]

            ###
            # QKV matmuls
            ###
            xqkv_fused = ttnn.linear(
                attn_norm,
                wqkv,
                program_config=self.wqkv_program_config,
                memory_config=self.model_config["XQKV_MM_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                dtype=self.dtype,
            )

            # Comment out to work around a deterministic hang when from_torch is called on the model's output
            # ttnn.deallocate(attn_norm, force=False)

            ###
            # Reshape and rotary embeddings
            ###
            (
                q_heads_1QBD,  # [seqlen, n_heads, bsz, head_dim]
                k_heads_1KBD,  # [seqlen, n_kv_heads, bsz, head_dim]
                v_heads_1KBD,  # [seqlen, n_kv_heads, bsz, head_dim]
            ) = ttnn.experimental.tensor.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                output_mem_config=self.model_config["QKV_HEADS_OUTPUT_MEMCFG"],
            )
            xqkv_fused.deallocate()

            # Update rotary matrix on device
            rotary_mat = self.rot_mat[current_pos]

            q_heads_1QBD = ttnn.linear(
                q_heads_1QBD,
                rotary_mat,
                program_config=self.q_heads_program_config,
                memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                dtype=ttnn.bfloat16,
            )
            k_heads_1KBD_rot = ttnn.linear(
                k_heads_1KBD,
                rotary_mat,
                program_config=self.k_heads_program_config,
                memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
            )
            k_heads_1KBD.deallocate()

            ###
            # KV update
            ###
            keys_BKCD = layer_past[0]
            values_BKCD = layer_past[1]

            ttnn.experimental.tensor.update_cache(keys_BKCD, k_heads_1KBD_rot, current_pos)
            ttnn.experimental.tensor.update_cache(values_BKCD, v_heads_1KBD, current_pos)
            self.layer_past_list[i] = [keys_BKCD, values_BKCD]

            k_heads_1KBD_rot.deallocate()
            v_heads_1KBD.deallocate()

            ###
            # Attention
            ###
            # reshape and shard keys
            keys_BKPD = keys_BKCD[:, :, :padded_layer_past_len, :]
            keys_1B_P_8D = ttnn.unsqueeze_to_4D(ttnn.transformer.concatenate_heads(keys_BKPD))
            keys_1B_8D_P_preshard = ttnn.permute(keys_1B_P_8D, (0, 1, 3, 2))
            keys_1B_8D_P = ttnn.to_memory_config(
                keys_1B_8D_P_preshard, self.model_config["KEYS_BATCHED_CONFIG"](padded_layer_past_len)
            )
            keys_BKPD.deallocate()
            keys_1B_P_8D.deallocate()
            keys_1B_8D_P_preshard.deallocate()

            # reshape values
            values_BKPD = values_BKCD[:, :, :padded_layer_past_len, :]
            values_B1_P_8D = ttnn.transformer.concatenate_heads(values_BKPD)
            values_1B_P_8D_preshard = ttnn.unsqueeze_to_4D(values_B1_P_8D)  # [:, :, :layer_slice, :]
            values_BKPD.deallocate()

            # reshape and shard queries
            q_heads_1QBD = q_heads_1QBD * head_dim_1QBD  # Scale q_heads instead of QK before softmax
            q_heads_1BQD = ttnn.permute(q_heads_1QBD, (0, 2, 1, 3))
            q_heads_1QBD.deallocate()
            q_heads_1B_Q_8D_preshard = (
                ttnn.matmul(
                    q_heads_1BQD,
                    expand_D_8D,
                    program_config=self.expand_program_config,
                    compute_kernel_config=self.compute_kernel_config,
                )
                * mask_Q_8D
            )
            q_heads_1BQD.deallocate()
            q_heads_1B_Q_8D = ttnn.to_memory_config(
                q_heads_1B_Q_8D_preshard, self.model_config["QUERIES_BATCHED_CONFIG"]
            )
            q_heads_1B_Q_8D_preshard.deallocate()

            # scores matmul
            attn_1BQP = ttnn.matmul(
                q_heads_1B_Q_8D,
                keys_1B_8D_P,
                # program_config=self.scores_program_config(padded_layer_past_len // 32),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                compute_kernel_config=self.compute_kernel_config_attn,
                dtype=ttnn.bfloat16,
            )
            keys_1B_8D_P.deallocate()
            q_heads_1B_Q_8D.deallocate()

            # scores softmax
            attn_1BQP_presoftmax = attn_1BQP[:, :, :, :layer_slice]
            attn_1BQP = ttnn.softmax(attn_1BQP_presoftmax, dim=-1)
            attn_1BQP = ttnn.pad(attn_1BQP, ((0, 0), (0, 0), (0, 0), (0, 0)), value=0.0)

            # shard values
            values_1B_P_8D = ttnn.to_memory_config(
                values_1B_P_8D_preshard, self.model_config["VALUES_BATCHED_CONFIG"](padded_layer_past_len)
            )
            values_1B_P_8D_preshard.deallocate()

            # attention matmul
            attn_output_1B_Q_8D = ttnn.matmul(
                attn_1BQP,
                values_1B_P_8D,
                program_config=self.attn_program_config,
                memory_config=self.model_config["QKV_MM_OUTPUT_MEMCFG"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config_attn,
            )

            attn_1BQP.deallocate()
            values_1B_P_8D.deallocate()

            # reduce and reshape
            attn_output_1BQD = ttnn.matmul(
                attn_output_1B_Q_8D * mask_Q_8D,
                reduce_8D_D,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self.reduce_program_config,
            )
            attn_output_1QBD = ttnn.permute(attn_output_1BQD, (0, 2, 1, 3))

            attn_output_1BQD.deallocate()
            attn_output_1B_Q_8D.deallocate()

            attn_output_11BH = ttnn.transformer.concatenate_heads(
                attn_output_1QBD, memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"]
            )
            attn_output_1QBD.deallocate()

            # dense output matmul
            dense_out = ttnn.linear(
                attn_output_11BH,
                wo,
                program_config=self.dense_program_config,
                memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
            )
            attn_output_11BH.deallocate()

            dense_outputs.append(dense_out)

            # start = time()
            # xqkv_fused.deallocate()
            # k_heads_1KBD.deallocate()
            # k_heads_1KBD_rot.deallocate()
            # v_heads_1KBD.deallocate()
            # keys_BKPD.deallocate()
            # keys_1B_P_8D.deallocate()
            # keys_1B_8D_P_preshard.deallocate()
            # values_BKPD.deallocate()
            # q_heads_1QBD.deallocate()
            # q_heads_1BQD.deallocate()
            # q_heads_1B_Q_8D_preshard.deallocate()
            # keys_1B_8D_P.deallocate()
            # q_heads_1B_Q_8D.deallocate()
            # attn_1BQP.deallocate()
            # attn_1BQP_presoftmax.deallocate()
            # values_1B_P_8D.deallocate()
            # attn_output_1BQD.deallocate()
            # attn_output_1QBD.deallocate()
            # attn_output_1B_Q_8D.deallocate()
            # attn_output_1BQD.deallocate()
            # attn_output_1QBD.deallocate()
            # attn_output_11BH.deallocate()
            # duration = time() - start
            # print(f'22 deallocate calls took: {duration * 1e3:.3f} ms')

        return dense_outputs
