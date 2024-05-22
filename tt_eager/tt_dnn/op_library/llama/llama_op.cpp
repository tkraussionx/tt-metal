#include <cmath>

#include "tt_eager/tt_dnn/op_library/llama/llama_op.hpp"
#include "tt_eager/tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_eager/tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_eager/tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_eager/tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_eager/tt_dnn/op_library/update_cache/update_cache_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_eager/tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_eager/tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_eager/tensor/types.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

namespace transformers {

Tensor llama_mlp_decode_forward(Tensor& input_tensor, const Tensor& w1, const Tensor& w2, const Tensor& w3) {

    auto w1_out = matmul_1d(
        input_tensor,
        w1,
        std::nullopt, // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 4),
            8,
            1,
            4,
            1,
            4,
            true,
            UnaryWithParam(UnaryOpType::SILU),
            true
        },
        MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1}, // ouptut_memconfig
        std::nullopt, // output_dtype
        WormholeComputeKernelConfig{
            MathFidelity::LoFi,
            true,
            true,
            true
        }
    );


    auto w3_out = matmul_1d(
        input_tensor,
        w3,
        std::nullopt, // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 4),
            8,
            1,
            4,
            1,
            4,
            true,
            std::nullopt, // fused activation
            true
        },
        MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1}, // ouptut_memconfig
        std::nullopt, // output_dtype
        WormholeComputeKernelConfig{
            MathFidelity::LoFi,
            true,
            true,
            true
        }
    );
    input_tensor.deallocate(true);

    auto hidden_states = mul(
        w1_out,
        w3_out,
        std::nullopt, // fused activation
        MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1}, // ouptut_memconfig
        DataType::BFLOAT8_B // output_dtype
    );
    w1_out.deallocate(true);
    w3_out.deallocate(true);

    auto hidden_states_gathered = tt::operations::ccl::all_gather(
        hidden_states,
        3, // dim
        1, // num_links
        MemoryConfig{
            TensorMemoryLayout::WIDTH_SHARDED,
            BufferType::L1,
            ShardSpec{
                CoreRangeSet({CoreCoord{0, 0}, CoreCoord{7, 3}}),
                {32, 1024},
                ShardOrientation::ROW_MAJOR,
                false
            }
        }
    );
    hidden_states.deallocate(true);

    auto output = matmul_1d(
        hidden_states_gathered,
        w2,
        std::nullopt, // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 4),
            32,
            1,
            1,
            1,
            1,
            true,
            std::nullopt, // fused activation
            true
        },
        MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1}, // ouptut_memconfig
        std::nullopt, // output_dtype
        WormholeComputeKernelConfig{
            MathFidelity::HiFi2,
            true,
            true,
            true
        }
    );
    hidden_states_gathered.deallocate(true);

    return output;
}

std::tuple<Tensor, Tensor, Tensor> llama_attn_qkv_decode_forward(const Tensor& input_tensor, const Tensor& rot_mat, const Tensor& wqkv, const MemoryConfig sharded_mem_config) {
    // ShardSpec cores_40_qkv_shard_spec = {
    //     CoreRangeSet({CoreCoord{0, 0}, CoreCoord{7, 4}}),
    //     {32, 256},
    //     ShardOrientation::ROW_MAJOR,
    //     false
    // };
    // MemoryConfig qkv_inp_memcfg = {
    //     TensorMemoryLayout::WIDTH_SHARDED,
    //     BufferType::L1,
    //     cores_40_qkv_shard_spec
    // };

    auto input_interleaved = sharded_to_interleaved(
        input_tensor,
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}
    );
    // input_tensor.deallocate(true);
    auto input_resharded = interleaved_to_sharded(
        input_interleaved,
        // qkv_inp_memcfg
        //tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::WIDTH_SHARDED,buffer_type=BufferType::L1,shard_spec=tt::tt_metal::ShardSpec(grid={[(x=0,y=0) - (x=7,y=4)]}, shape={32, 256}, orientation=ShardOrientation::ROW_MAJOR, halo=false))
        sharded_mem_config
        // std::nullopt // output dtype
    );
    // input_interleaved.deallocate(true);

    auto fused_query_key_value = matmul_1d(
        input_resharded,
        wqkv,
        std::nullopt, // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 5),
            8,
            1,
            1,
            1,
            1,
            true,
            std::nullopt, // fused activation
            true
        },
        MemoryConfig{
            TensorMemoryLayout::WIDTH_SHARDED,
            BufferType::L1,
            ShardSpec{
                CoreRangeSet({CoreCoord{0, 0}, CoreCoord{7, 0}}),
                {32, 160},
                ShardOrientation::ROW_MAJOR,
                false
            }
        }, // output_memconfig
        // DataType::BFLOAT16, // output_dtype
        std::nullopt,
        WormholeComputeKernelConfig{
            MathFidelity::HiFi2,
            true,
            true,
            true
        }
    );
    // return std::make_tuple(fused_query_key_value, fused_query_key_value, fused_query_key_value);

    // input_resharded.deallocate(true);

    auto heads = nlp_create_qkv_heads_decode(
        fused_query_key_value,
        8, // num Q heads
        1, // num KV heads
        MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1}
    );

    auto query_layer = heads.at(0);
    auto key_layer = heads.at(1);
    auto value_layer = heads.at(2);

    auto query_rotated = matmul(
        query_layer,
        rot_mat,
        std::nullopt, // bias
        MatmulMultiCoreReuseProgramConfig{
            CoreCoord(8, 4),
            4,
            1,
            4,
            1,
            4,
        },
        MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1}, // output_memconfig
        std::nullopt, // output_dtype
        WormholeComputeKernelConfig{
            MathFidelity::HiFi4,
            false,
            true,
            true
        }
    );

    auto key_rotated = matmul(
        key_layer,
        rot_mat,
        std::nullopt, // bias
        MatmulMultiCoreReuseProgramConfig{
            CoreCoord(8, 4),
            4,
            1,
            4,
            1,
            4,
        },
        MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1}, // output_memconfig
        std::nullopt, // output_dtype
        WormholeComputeKernelConfig{
            MathFidelity::HiFi4,
            false,
            true,
            true
        }
    );

    return std::make_tuple(query_rotated, key_rotated, value_layer);
}

Tensor llama_attn_mqa_decode_forward(Tensor& query_layer, Tensor& key_layer, Tensor& value_layer, const uint32_t start_pos, const Tensor& attn_masks, const uint32_t batch_offset, Tensor& K_cache, Tensor& V_cache, const float scale,
MemoryConfig kv_cache_mem_config, MemoryConfig dram_mem_config, MemoryConfig height_mem_config, MemoryConfig attn_output_memcfg, MemoryConfig scores_output_memcfg) {
/*

        padded_layer_past_len = nearest_32(start_pos + 1)

        # K Cache Update
        kv_cache_memcfg = self.model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"]
        if kv_cache_memcfg.is_sharded():
            kv_cache_shard_shape = kv_cache_memcfg.shard_spec.shape
            kv_cache_shard_shape[0] = self.layer_past[0].shape[0] * padded_layer_past_len
            kv_cache_memcfg.shard_spec.shape = kv_cache_shard_shape

        keys = self.layer_past[0]
        tt_lib.tensor.update_cache(keys, key_layer, start_pos, batch_offset=batch_offset)
        key_layer.deallocate(True)
        # key and value layers will have kv_seq_len padded to nearest 32

        keys = self.layer_past[0]
        key_layer = tt_lib.tensor.unpad(
            keys,
            [0, 0, 0, 0],
            [
                self.n_local_kv_heads - 1,
                self.max_batch_size - 1,
                padded_layer_past_len - 1,
                self.head_dim - 1,
            ],
            output_mem_config=self.model_config["DRAM_MEMCFG"],
        )

        key_layer = tt_lib.tensor.interleaved_to_sharded(key_layer, sharded_mem_config=kv_cache_memcfg)

        # PRE-SOFTMAX MM

        key_layer_transposed = tt_lib.tensor.transpose(
            key_layer,
            -2,
            -1,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )

        key_layer.deallocate(True)

        attn_prog_config = self.model_config["ATTN_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)
        attn_output_memcfg = self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"]
        attn_output_memcfg.shard_spec.shape[1] = padded_layer_past_len

        attn_weights = tt_lib.operations.primary.matmul(
            query_layer,
            key_layer_transposed,
            program_config=attn_prog_config,
            output_mem_config=attn_output_memcfg,
            output_dtype=self.model_config["ATTN_BATCHED_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        query_layer.deallocate(True)
        key_layer_transposed.deallocate(True)

        # SOFTMAX
        softmax_progcfg = self.model_config["BATCHED_SOFTMAX_PROGCFG"]
        softmax_progcfg.block_w = padded_layer_past_len // 32

        attn_weights = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
            attn_weights,
            self.scale,
            attn_masks,
            program_config=self.model_config["BATCHED_SOFTMAX_PROGCFG"],
            is_causal_mask=True,
        )

        # V CACHE UPDATE

        values = self.layer_past[1]
        tt_lib.tensor.update_cache(values, value_layer, start_pos, batch_offset=batch_offset)
        value_layer.deallocate(True)

        values = self.layer_past[1]
        value_layer = tt_lib.tensor.unpad(
            values,
            [0, 0, 0, 0],
            [
                self.n_local_kv_heads - 1,
                self.max_batch_size - 1,
                padded_layer_past_len - 1,
                self.head_dim - 1,
            ],
            output_mem_config=self.model_config["DRAM_MEMCFG"],
        )

        value_layer = tt_lib.tensor.interleaved_to_sharded(value_layer, sharded_mem_config=kv_cache_memcfg)

        # POST-SOFTMAX MM
        scores_prog_config = self.model_config["SCORES_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)

        attn_output = tt_lib.operations.primary.matmul(
            attn_weights,
            value_layer,
            program_config=scores_prog_config,
            output_mem_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["BFLOAT16_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        attn_weights.deallocate(True)
        value_layer.deallocate(True)

        return attn_output*/

        // math.ceil(x / 32) * 32
        uint32_t padded_layer_past_len = std::ceil((start_pos+1) / 32.0) * 32;

        // auto kv_cache_memcfg = MemoryConfig{
        //     TensorMemoryLayout::HEIGHT_SHARDED,
        //     BufferType::L1,
        //     ShardSpec{
        //         CoreRangeSet({CoreCoord{0, 0}, CoreCoord{7, 3}}),
        //         {padded_layer_past_len, 128},
        //         ShardOrientation::ROW_MAJOR,
        //         false
        //     }
        // };

        update_cache(K_cache, key_layer, start_pos, batch_offset);
        key_layer.deallocate(true);

        // unpad
        auto key_layer_unpadded = unpad(
            K_cache,
            {0, 0, 0, 0},
            {0, 31, padded_layer_past_len - 1, 127}
        );

        auto key_layer_sharded = interleaved_to_sharded(
            key_layer_unpadded,
            kv_cache_mem_config
        );
        key_layer_unpadded.deallocate(true);

        auto key_layer_transposed = transpose(
            key_layer_sharded,
            -2,
            -1,
            // MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1}
            height_mem_config
        );

        key_layer_sharded.deallocate(true);

        auto attn_prog_config = MatmulMultiCoreReuseProgramConfig{
            CoreCoord(8, 4),
            128 / 32, // head_dim tiles
            1,
            1,
            1,
            padded_layer_past_len / 32
        };

        // auto attn_output_memcfg = MemoryConfig{
        //     TensorMemoryLayout::HEIGHT_SHARDED,
        //     BufferType::L1,
        //     ShardSpec{
        //         CoreRangeSet({CoreCoord{0, 0}, CoreCoord{7, 3}}),
        //         {32, padded_layer_past_len},
        //         ShardOrientation::ROW_MAJOR,
        //         false
        //     }
        // };

        auto attn_weights = matmul(
            query_layer,
            key_layer_transposed,
            std::nullopt, // bias
            attn_prog_config,
            // attn_output_memcfg,
            attn_output_memcfg,
            std::nullopt, // output_dtype
            WormholeComputeKernelConfig{
                MathFidelity::HiFi2,
                true,
                true,
                true
            }
        );
        query_layer.deallocate(true);
        key_layer_transposed.deallocate(true);

        auto softmax_progcfg = SoftmaxShardedMultiCoreProgramConfig{
            CoreCoord(8, 4),
            1,
            1,
            padded_layer_past_len / 32
        };
        auto attn_weights_softmax = scale_mask_softmax_in_place(
            attn_weights,
            scale,
            attn_masks,
            softmax_progcfg,
            true
        );

        // V cache update
        update_cache(V_cache, value_layer, start_pos, batch_offset);
        value_layer.deallocate(true);
        // unpad V
        auto value_layer_unpadded = unpad(
            V_cache,
            {0, 0, 0, 0},
            {0, 31, padded_layer_past_len - 1, 127}
        );
        auto value_layer_sharded = interleaved_to_sharded(
            value_layer_unpadded,
            kv_cache_mem_config
        );
        value_layer.deallocate(true);

        auto scores_prog_config = MatmulMultiCoreReuseProgramConfig{
            CoreCoord(8, 4),
            padded_layer_past_len / 32,
            1,
            1,
            1,
            128 / 32 // head_dim
        };

        auto attn_output = matmul(
            attn_weights_softmax,
            value_layer_sharded,
            std::nullopt, // bias
            scores_prog_config,
            // MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1},
            // DataType::BFLOAT16,
            // WormholeComputeKernelConfig{
            //     MathFidelity::HiFi2,
            //     true,
            //     true,
            //     true
            // }
            scores_output_memcfg
        );

        attn_weights_softmax.deallocate(true);
        value_layer_sharded.deallocate(true);

        return attn_output;
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
