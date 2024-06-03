# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
import torch
from torch import nn
import ttnn.experimental as tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.experimental.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    get_weight_cache_path,
    get_weight_cache_path_ttnn,
)
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


class TtLlamaMLP_optimized:
    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        emulated=False,
        cache_path=None,
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        self.model_config = model_config
        self.emulated = emulated
        self.read_cache = read_cache

        self.hidden_size = hidden_size

        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path

        self.load_weights()

    def set_model_config(self, model_config):
        self.model_config = model_config

    def load_weights(self):
        assert not hasattr(self, "w1_list"), "w1_list is already an attribute of this object"
        assert not hasattr(self, "w3_list"), "w3_list is already an attribute of this object"
        assert not hasattr(self, "w2_list"), "w2_list is already an attribute of this object"

        w1_str = f"{self.layer_name}.feed_forward.w1.weight"
        w2_str = f"{self.layer_name}.feed_forward.w2.weight"
        w3_str = f"{self.layer_name}.feed_forward.w3.weight"

        w1_dtype = ttnn.bfloat4_b
        w2_dtype = ttnn.bfloat8_b
        w3_dtype = ttnn.bfloat4_b

        padded_w1 = None
        padded_w2 = None
        padded_w3 = None
        if not self.read_cache:
            # Do padding
            H = 8 * 1024
            PADDED_H4 = 32 * 1024
            H4 = 28 * 1024
            padded_w1 = torch.zeros(1, 1, H, PADDED_H4)
            padded_w2 = torch.zeros(1, 1, PADDED_H4, H)
            padded_w3 = torch.zeros(1, 1, H, PADDED_H4)
            padded_w1[:, :, :, :H4] = self.state_dict[w1_str].transpose(-2, -1)
            padded_w2[:, :, :H4, :] = self.state_dict[w2_str].transpose(-2, -1)
            padded_w3[:, :, :, :H4] = self.state_dict[w3_str].transpose(-2, -1)

        # w1: 8k x 4k. width-sharded on 12 banks, 4224 over 12 banks.
        device = self.device_mesh.get_device(0)
        weight_grid = ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )
        w1_shard_shape = (8192, 4224 // 12)  # padded cols to divide by 12
        w1_shard_spec = ttnn.ShardSpec(weight_grid, w1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w1_shard_spec)
        self.w1 = ttnn.from_torch(
            padded_w1,
            dtype=w1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.model_config["DRAM_MEMCFG"],
            memory_config=w1_mem_config,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            # cache_file_name=self.cache_path / w1_str,
        )
        # breakpoint()
        # self.w1 = ttnn.to_device(w1_ttnn, self.device_mesh)

        w2_shard_shape = (32768, 96)  # Padded cols 1024/12
        w2_shard_spec = ttnn.ShardSpec(weight_grid, w2_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        w2_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)
        self.w2 = ttnn.as_tensor(
            padded_w2,
            dtype=w2_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.model_config["DRAM_MEMCFG"],
            memory_config=w2_memory_config,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            # cache_file_name=self.cache_path / w2_str,
        )
        # self.w2 = ttnn.to_device(w2_ttnn, self.device_mesh)
        self.w3 = ttnn.from_torch(
            padded_w3,
            dtype=w3_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            # memory_config=self.model_config["DRAM_MEMCFG"],
            memory_config=w1_mem_config,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            # cache_file_name=self.cache_path / w3_str,
        )
        # self.w3 = ttnn.to_device(w3_ttnn, self.device_mesh)

    def prepare_inputs(self, x):
        x_multichip = ttnn.from_torch(
            x,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.model_config["LN_MLP_OUTPUT_DTYPE"],
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
        )

        if self.model_config["LLM_MODE"] == "decode":
            x_multichip = ttnn.to_memory_config(
                x_multichip,
                ttnn.create_sharded_memory_config(
                    shape=(32, 8192 // 32),
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                ),
            )
        elif self.model_config["LLM_MODE"] == "prefill":
            x_multichip = ttnn.to_memory_config(x_multichip, self.model_config["DRAM_MEMCFG"])

        return x_multichip

    def __call__(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(x)
        # Prefill should have input tensor of shape (1, batch, seqlen, hidden_size)
        elif self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(x)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def prefill_forward(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        # TODO: Use FP32 accumulate after the issue with primary.matmul with FP32 accumulate is fixed
        w1_out = tt_lib.operations.primary.matmul(
            x,
            self.w1,
            program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
            output_mem_config=self.model_config["MLP_BLOCK_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG_LOFI"],
            # output_dtype=self.model_config["BFP8_DTYPE"],
            output_dtype=self.model_config["BFLOAT16_DTYPE"],
        )

        w3_out = tt_lib.operations.primary.matmul(
            x,
            self.w3,
            program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
            output_mem_config=self.model_config["MLP_BLOCK_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG_LOFI"],
            # output_dtype=self.model_config["BFP8_DTYPE"],
            output_dtype=self.model_config["BFLOAT16_DTYPE"],
        )
        x.deallocate(True)

        hidden_states = ttnn.mul(w1_out, w3_out, dtype=ttnn.bfloat8_b, memory_config=self.model_config["DRAM_MEMCFG"])
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
        )

        hidden_states = tt_lib.operations.primary.matmul(
            hidden_states,
            self.w2,
            program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
            output_dtype=self.model_config["BFLOAT16_DTYPE"],
        )

        return hidden_states

    def decode_forward(self, x: List[tt_lib.tensor.Tensor]) -> List[tt_lib.tensor.Tensor]:
        hidden_states = []
        w1_outs = []
        w3_outs = []

        # They only used 8 cores for input. Let's first see if 32 works.

        sharded_output_memcfg = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        w1_prog_cfg = (
            program_config
        ) = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=ttnn.experimental.tensor.FusibleActivation.SILU,
            # fused_activation=None,
        )

        w3_prog_cfg = (
            program_config
        ) = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            # fused_activation=ttnn.experimental.tensor.FusibleActivation.SILU,
            fused_activation=None,
        )

        # x_reshard = ttnn.experimental.tensor.reshard(x, ttnn.MemoryConfig(
        #     memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #     buffer_type=ttnn.BufferType.L1,
        #     shard_spec=ttnn.ShardSpec(
        #         ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        #             (32, 1024),
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         )
        #     )
        # )

        # breakpoint()
        # TODO: Fix output sharding
        w1_out = ttnn.matmul(
            x,
            self.w1,
            # program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
            program_config=w1_prog_cfg,
            memory_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )

        # # w1_out = ttnn.experimental.tensor.reshard(w1_out, self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"])
        # w1_out = ttnn.experimental.tensor.reshard(w1_out, ttnn.MemoryConfig(
        #     memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #     buffer_type=ttnn.BufferType.L1,
        #     shard_spec=ttnn.ShardSpec(
        #         ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        #             (32, 128),
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         )
        #     )
        # )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            # program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
            program_config=w3_prog_cfg,
            memory_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG_LOFI"],
        )
        x.deallocate(True)

        hidden_states = tt_lib.tensor.mul(
            w1_out,
            w3_out,
            output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            output_dtype=self.model_config["BFP8_DTYPE"],
            # fused_activations=[tt_lib.tensor.FusibleActivation.SILU,],
        )
        w1_out.deallocate(True)
        w3_out.deallocate(True)

        # hidden_states = tt_lib.tensor.sharded_to_interleaved(
        #     hidden_states, output_mem_config=self.model_config["L1_MEMCFG"]
        # )

        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            # memory_config=self.model_config["L1_MEMCFG"],
            memory_config=self.model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # hidden_states = tt_lib.tensor.interleaved_to_sharded(
        #     hidden_states, sharded_mem_config=self.model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"]
        # )

        w2_prog_cfg = (
            program_config
        ) = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
        )

        hidden_states = ttnn.matmul(
            hidden_states,
            self.w2,
            # program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
            program_config=w2_prog_cfg,
            memory_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            # output_dtype=self.model_config["BFP8_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        return hidden_states
