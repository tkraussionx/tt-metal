# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import nearest_32
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor


class TtMixtralAttention(torch.nn.Module):
    def __init__(self, device_mesh, state_dict, args, layer_num, dtype):
        super().__init__()
        self.num_devices = 8
        self.state_dict = state_dict
        self.device_mesh = device_mesh
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

        self.model_config = self.model_args.get_model_config()

        layer_name = f"layers.{layer_num}.attention"

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: self.model_args.weight_cache_path(dtype) / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        self.wqkv = ttnn.as_tensor(
            torch.concat(
                [
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
                    )
                    for i in range(self.num_devices)
                ],
                dim=-1,
            ),
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=1),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wqkv_multidevice"),
        )
        self.wqkv = ttnn.to_device(self.wqkv, self.device_mesh)
        self.wo = ttnn.as_tensor(
            torch.transpose(
                self.state_dict[wo_str],
                -2,
                -1,
            ),
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wo_multidevice"),
        )
        self.wo = ttnn.to_device(self.wo, self.device_mesh)
        self.wo_prefill = ttnn.as_tensor(
            torch.transpose(
                self.state_dict[wo_str],
                -2,
                -1,
            ),
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wo_prefill"),
        )
        self.wo_prefill = ttnn.to_device(self.wo_prefill, self.device_mesh)
        cache_k = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.sliding_window,
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.sliding_window,
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.as_tensor(
                lp,
                device=self.device_mesh,
                mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
                dtype=ttnn.bfloat8_b,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                memory_config=self.model_config["ATTN_CACHE_WEIGHTS_MEMCFG"],
                cache_file_name=cache_name(f"empty_attn_cache_{cache_k.shape}"),
            )
            for lp in layer_past
        ]

        self.layer_past = [ttnn.to_device(lp, self.device_mesh) for lp in self.layer_past]

        self.scale = self.head_dim**-0.5

        reduce_mask_torch = torch.zeros(1, 1, self.max_batch_size, self.max_batch_size * 8)
        for i in range(self.max_batch_size):
            reduce_mask_torch[:, :, i, range(i, self.max_batch_size * 8, self.max_batch_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )

        self.reduce_mask = ttnn.to_device(self.reduce_mask, self.device_mesh)
        self.compute_kernel = self.model_args.get_compute_kernel_config()
        self.compute_kernel_attn = self.model_args.get_compute_kernel_attn_config()

        self.core_grid = self.model_args.max_grid_size
        self.core_grid_attention = self.model_args.core_grid_attention

    def forward(
        self,
        xs,
        start_pos,
        current_pos,
        attn_masks,
        rot_mats,
    ):
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        current_pos: start_pos % self.sliding_window
        attn_masks: (seq_len, batch, n_heads, cache_len+seq_len)
        rot_mats: list of rotation matrices for each device

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        D : head_dim (128)
        P : padded_layer_past_len
        """
        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)

        x_11BH = xs
        wo = self.wo
        layer_past = self.layer_past
        rot_mat = rot_mats[start_pos]
        attn_mask_1B4P = attn_masks

        ###
        # QKV matmuls
        ###
        xqkv_fused = ttnn.linear(
            x_11BH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            core_grid=self.core_grid_attention,
            compute_kernel_config=self.compute_kernel,
        )

        # split qkv into heads
        (
            q_heads_1B4D,
            k_heads_1B1D,
            v_heads_1B1D,
        ) = ttnn.experimental.tensor.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###
        q_mem_config = q_heads_1B4D.memory_config()
        k_mem_config = k_heads_1B1D.memory_config()
        q_heads_1B4D = ttnn.experimental.operations.primary.matmul(
            q_heads_1B4D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            output_mem_config=q_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"]
            # [seqlen, bsz, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, bsz, padd_heads, head_dim]
        )
        k_heads_1B1D = ttnn.experimental.operations.primary.matmul(
            k_heads_1B1D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            output_mem_config=k_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )

        ###
        # KV update
        ###
        keys_1BPD = layer_past[0]
        values_1BPD = layer_past[1]
        ttnn.kv_cache.update_cache_for_token_(keys_1BPD, k_heads_1B1D, current_pos)
        ttnn.kv_cache.update_cache_for_token_(values_1BPD, v_heads_1B1D, current_pos)
        self.layer_past = [keys_1BPD, values_1BPD]
        k_heads_1B1D.deallocate(True)
        v_heads_1B1D.deallocate(True)

        keys_1BPD = ttnn.experimental.tensor.nlp_kv_cache_load_slice(
            keys_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )

        ###
        # Attention
        ###
        # transpose keys
        keys_1BDP = ttnn.experimental.tensor.transpose(
            keys_1BPD,
            -2,
            -1,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        keys_1BPD.deallocate(True)

        # scores matmul
        attn_1B4P = ttnn.matmul(
            q_heads_1B4D,
            keys_1BDP,
            dtype=ttnn.bfloat16,
            core_grid=self.core_grid_attention,
            memory_config=self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"](padded_layer_past_len),
            compute_kernel_config=self.compute_kernel_attn,
        )
        q_heads_1B4D.deallocate(True)
        keys_1BDP.deallocate(True)

        # Softmax and scaling
        attn_1B4P = ttnn.experimental.operations.primary.transformers.scale_mask_softmax_in_place(
            attn_1B4P,
            self.scale,
            attn_mask_1B4P,
            program_config=self.model_config["ATTN_BATCHED_SOFTMAX_PROGCFG"](padded_layer_past_len),
            is_causal_mask=True,
        )

        # values matmul
        values_1BPD = ttnn.experimental.tensor.nlp_kv_cache_load_slice(
            values_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )
        attn_output_1B4D = ttnn.matmul(
            attn_1B4P,
            values_1BPD,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
            core_grid=self.core_grid_attention,
            compute_kernel_config=self.compute_kernel_attn,
        )
        attn_1B4P.deallocate(True)
        values_1BPD.deallocate(True)

        attn_output_11BH = ttnn.experimental.tensor.nlp_concat_heads_decode(
            attn_output_1B4D,
            num_heads=4,
        )
        attn_output_1B4D.deallocate(True)

        attn_output_11BH = ttnn.experimental.tensor.sharded_to_interleaved(
            attn_output_11BH, output_mem_config=ttnn.L1_MEMORY_CONFIG
        )

        ###
        # Output matmul
        ###
        dense_out_11BH = ttnn.linear(
            attn_output_11BH,
            wo,
            memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
            core_grid=self.core_grid,
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat8_b,
        )
        attn_output_11BH.deallocate(True)
        # All gather
        dense_outputs_11BH = ttnn.all_gather(dense_out_11BH, dim=2, num_links=1)

        # return the sum of the outputs
        dense_outputs_11BH = ttnn.matmul(self.reduce_mask, dense_outputs_11BH)
        return dense_outputs_11BH

    def forward_prefill(self, xs_11SH, start_pos, current_pos, attn_masks, rot_mats, user_id: int = 0):
        assert xs_11SH.shape[2] % 128 == 0 and xs_11SH.shape[2] > 0, "Seqlen must be divisible by 128"
        padded_layer_len = nearest_32(start_pos + 1)
        ###
        # QKV matmuls
        ###
        xqkv_fused = ttnn.linear(
            xs_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            core_grid=self.core_grid_attention,
            compute_kernel_config=self.compute_kernel,
        )

        print("xqkv_fused", xqkv_fused.shape)

        # split qkv into heads
        (
            q_heads_14SD,
            k_heads_11SD,
            v_heads_11SD,
        ) = ttnn.experimental.tensor.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            output_mem_config=ttnn.L1_MEMORY_CONFIG,  # self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        print("q_heads_14SD", q_heads_14SD.shape, k_heads_11SD.shape, v_heads_11SD.shape)

        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###

        # TODO: Implement the rotation matrix
        seq_len = q_heads_14SD.shape[2]
        slice_size = 256 if seq_len == 2048 else 128
        cores_y = 4 if slice_size == 128 else 8
        num_slices = seq_len // slice_size  # we do q_lens of 128 per iteration (slice), then we concat the result.

        # FILL K CACHE
        keys = self.layer_past[0]
        keys_11SD = ttnn.reshape(keys, [1, self.n_local_kv_heads, -1, self.head_dim])
        ttnn.experimental.tensor.fill_cache(
            keys_11SD, ttnn.experimental.tensor.typecast(k_heads_11SD, ttnn.bfloat8_b), user_id
        )

        print("keys_11SD", keys_11SD.shape)

        # FILL V CACHE
        values = self.layer_past[0]
        values_11SD = ttnn.reshape(values, [1, self.n_local_kv_heads, -1, self.head_dim])
        ttnn.experimental.tensor.fill_cache(
            values_11SD, ttnn.experimental.tensor.typecast(v_heads_11SD, ttnn.bfloat8_b), user_id
        )

        print("values_11SD", values_11SD.shape)
        # SDPA
        program_config = ttnn.experimental.operations.primary.transformers.SDPAMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 8],
            q_chunk_size=128,
            k_chunk_size=128,
        )
        attn_output_14SD = ttnn.experimental.operations.primary.transformers.scaled_dot_product_attention(
            q_heads_14SD,
            k_heads_11SD,
            v_heads_11SD,
            attn_masks,
            is_causal=True,
            scale=self.scale,
            program_config=program_config,
        )

        print("attn_output_14SD", attn_output_14SD.shape)

        # deallocate keys and values
        q_heads_14SD.deallocate(True)
        k_heads_11SD.deallocate(True)
        v_heads_11SD.deallocate(True)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.tensor.nlp_concat_heads(
            attn_output_14SD,
            output_mem_config=ttnn.L1_MEMORY_CONFIG,
        )

        print("attn_output_11SH", attn_output_11SH.shape)
        attn_output_11SH = ttnn.all_gather(
            attn_output_11SH,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print("attn_output_11SH", attn_output_11SH.shape)

        # seq_tiles = attn_output_11SH.shape[2] // 32
        # cores_y = 8 if seq_tiles % 8 == 0 else 4
        # seq_len_tiles = seq_len // 32
        # cores_y = 4  # 8 if seq_len_tiles % 8 == 0 else 4
        # self.model_config["SELFOUT_MM_PROGCFG_LAMBDA"] = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        #     compute_with_storage_grid_size=(8, cores_y),
        #     in0_block_w=8,  # how much inner dim you take each time
        #     out_subblock_h=1,  # Must be divisible by per_core_M
        #     out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        #     per_core_M=seq_len_tiles // cores_y,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        #     per_core_N=4,  # N / TILE_WIDTH / Grid_Size
        #     transpose_mcast=False,
        #     fused_activation=None,
        # )
        # dense_out_prog_cfg = self.model_config["SELFOUT_MM_PROGCFG_LAMBDA"]

        # attn_output_11SH = ttnn.experimental.operations.primary.matmul(
        #     attn_output_11SH,
        #     self.wo,
        #     #program_config=dense_out_prog_cfg,
        #     output_mem_config=ttnn.L1_MEMORY_CONFIG,
        #     output_dtype=ttnn.bfloat8_b,
        #     compute_kernel_config=self.compute_kernel,
        # )

        attn_output_11SH = ttnn.linear(attn_output_11SH, self.wo_prefill, core_grid=ttnn.CoreGrid(y=8, x=8))

        print("attn_output_11SH", attn_output_11SH.shape)

        return attn_output_11SH
