from tests.didt.test_didt_base import DidtTestBase
import ttnn


class Ff1Test(DidtTestBase):
    def __init__(
        self,
        num_devices,
        all_devices,
        seq_len,
        inner_dim,
        weights_n,
        per_core_M,
        per_core_N,
        in_block_w,
        out_subblock_h,
        out_subblock_w,
        loop_count,
        determinism_check_enabled,
        determinism_check_iterations,
    ):
        super().__init__(
            num_devices,
            all_devices,
            seq_len,
            inner_dim,
            weights_n,
            per_core_M,
            per_core_N,
            in_block_w,
            out_subblock_h,
            out_subblock_w,
            loop_count,
            determinism_check_enabled,
            determinism_check_iterations,
        )

    def set_in0_mem_config(self):
        in0_block_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        # Volume must match batch size
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    ),
                }
            ),
            [
                128,
                576,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )

        self.in0_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_block_shard_spec
        )

    def set_in1_mem_config(self):
        self.in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    def set_out_mem_config(self):
        self.out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    def set_data_formats(self):
        self.in0_dtype = ttnn.DataType.BFLOAT16
        self.in1_dtype = ttnn.DataType.BFLOAT8_B
        self.out_dtype = ttnn.DataType.BFLOAT16

    def set_program_config(self):
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=self.in_block_w,
            out_subblock_h=self.out_subblock_h,
            out_subblock_w=self.out_subblock_w,
            per_core_M=self.per_core_M,
            per_core_N=self.per_core_N,
            transpose_mcast=False,
            fused_activation=[ttnn.UnaryOpType.GELU, True],
        )

    def set_compute_config(self):
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
