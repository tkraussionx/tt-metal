#include "tt_eager/tt_dnn/op_library/llama/llama_op.hpp"
#include "tt_eager/tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_eager/tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_eager/tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_eager/tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
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

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
