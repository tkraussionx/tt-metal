#include "tt_eager/tt_dnn/op_library/llama/llama_op.hpp"
#include "tt_eager/tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_eager/tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_eager/tensor/types.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

namespace transformers {

    Tensor llama_mlp_decode_forward(Tensor& input_tensor, Tensor& w1, Tensor& w2, Tensor& w3) {

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

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
