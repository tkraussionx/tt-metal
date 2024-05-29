#include "tt_eager/tt_dnn/op_library/llama/llama_op.hpp"

#include <cmath>

#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_dnn/op_library/all_gather/all_gather_op.hpp"
#include "tt_eager/tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_eager/tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "tt_eager/tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_eager/tt_dnn/op_library/softmax/softmax_op.hpp"
#include "tt_eager/tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_eager/tt_dnn/op_library/update_cache/update_cache_op.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

namespace transformers {

Tensor llama_mlp_decode_forward(Tensor& input_tensor, const Tensor& w1, const Tensor& w2, const Tensor& w3) {
    auto w1_out = matmul_1d(
        input_tensor,
        w1,
        std::nullopt,  // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 4), 8, 1, 4, 1, 4, true, UnaryWithParam(UnaryOpType::SILU), true},
        MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1},  // ouptut_memconfig
        std::nullopt,                                                     // output_dtype
        WormholeComputeKernelConfig{MathFidelity::LoFi, true, true, true});

    auto w3_out = matmul_1d(
        input_tensor,
        w3,
        std::nullopt,  // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 4),
            8,
            1,
            4,
            1,
            4,
            true,
            std::nullopt,  // fused activation
            true},
        MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1},  // ouptut_memconfig
        std::nullopt,                                                     // output_dtype
        WormholeComputeKernelConfig{MathFidelity::LoFi, true, true, true});
    input_tensor.deallocate(true);

    auto hidden_states =
        mul(w1_out,
            w3_out,
            std::nullopt,                                                     // fused activation
            MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1},  // ouptut_memconfig
            DataType::BFLOAT8_B                                               // output_dtype
        );
    w1_out.deallocate(true);
    w3_out.deallocate(true);

    auto all_gather_memcfg = MemoryConfig{
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec{
            CoreRangeSet({CoreRange(CoreCoord({0, 0}), CoreCoord({7, 3}))}),
            {32, 1024},
            ShardOrientation::ROW_MAJOR,
            false}};

    auto hidden_states_gathered = tt::operations::ccl::all_gather(
        hidden_states,
        3,  // dim
        1,  // num_links
        all_gather_memcfg);
    hidden_states.deallocate(true);

    auto output = matmul_1d(
        hidden_states_gathered,
        w2,
        std::nullopt,  // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 4),
            32,
            1,
            1,
            1,
            1,
            true,
            std::nullopt,  // fused activation
            true},
        MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1},  // ouptut_memconfig
        std::nullopt,                                                     // output_dtype
        WormholeComputeKernelConfig{MathFidelity::HiFi2, true, true, true});
    hidden_states_gathered.deallocate(true);

    return output;
}

std::tuple<Tensor, Tensor, Tensor> llama_attn_qkv_decode_forward(
    const Tensor& input_tensor, const Tensor& rot_mat, const Tensor& wqkv, const MemoryConfig sharded_mem_config) {
    auto input_interleaved =
        sharded_to_interleaved(input_tensor, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
    // input_tensor.deallocate(true);
    auto input_resharded = interleaved_to_sharded(input_interleaved, sharded_mem_config);
    // input_interleaved.deallocate(true);

    auto fused_query_key_value = matmul_1d(
        input_resharded,
        wqkv,
        std::nullopt,  // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 5),
            8,
            1,
            1,
            1,
            1,
            true,
            std::nullopt,  // fused activation
            true},
        MemoryConfig{
            TensorMemoryLayout::WIDTH_SHARDED,
            BufferType::L1,
            ShardSpec{
                CoreRangeSet({CoreRange(CoreCoord({0, 0}), CoreCoord({7, 0}))}),
                {32, 160},
                ShardOrientation::ROW_MAJOR,
                false}},  // output_memconfig
        std::nullopt,
        WormholeComputeKernelConfig{MathFidelity::HiFi2, true, true, true});

    // input_resharded.deallocate(true);

    auto heads = nlp_create_qkv_heads_decode(
        fused_query_key_value,
        8,  // num Q heads
        1,  // num KV heads
        MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1});

    auto query_layer = heads.at(0);
    auto key_layer = heads.at(1);
    auto value_layer = heads.at(2);

    auto query_rotated = matmul(
        query_layer,
        rot_mat,
        std::nullopt,  // bias
        MatmulMultiCoreReuseProgramConfig{
            CoreCoord(8, 4),
            4,
            1,
            4,
            1,
            4,
        },
        MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1},  // output_memconfig
        std::nullopt,                                                      // output_dtype
        WormholeComputeKernelConfig{MathFidelity::HiFi4, false, true, true});

    auto key_rotated = matmul(
        key_layer,
        rot_mat,
        std::nullopt,  // bias
        MatmulMultiCoreReuseProgramConfig{
            CoreCoord(8, 4),
            4,
            1,
            4,
            1,
            4,
        },
        MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1},  // output_memconfig
        std::nullopt,                                                      // output_dtype
        WormholeComputeKernelConfig{MathFidelity::HiFi4, false, true, true});

    return std::make_tuple(query_rotated, key_rotated, value_layer);
}

Tensor llama_attn_mqa_decode_forward(
    Tensor& query_layer,
    Tensor& key_layer,
    Tensor& value_layer,
    const uint32_t start_pos,
    const Tensor& attn_masks,
    const uint32_t batch_offset,
    Tensor& K_cache,
    Tensor& V_cache,
    const float scale,
    MemoryConfig kv_cache_mem_config /* necessary */) {
    uint32_t padded_layer_past_len = std::ceil((start_pos + 1) / 32.0) * 32;

    update_cache(K_cache, key_layer, start_pos, batch_offset);
    key_layer.deallocate(true);

    // unpad
    auto key_layer_unpadded = unpad(K_cache, {0, 0, 0, 0}, {0, 31, padded_layer_past_len - 1, 127});

    auto key_layer_sharded = interleaved_to_sharded(key_layer_unpadded, kv_cache_mem_config);
    key_layer_unpadded.deallocate(true);

    auto key_layer_transposed = transpose(
        key_layer_sharded, -2, -1, MemoryConfig{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1}
        // height_mem_config
    );

    key_layer_sharded.deallocate(true);

    auto attn_prog_config = MatmulMultiCoreReuseProgramConfig{
        CoreCoord(8, 4),
        128 / 32,  // head_dim tiles
        1,
        1,
        1,
        padded_layer_past_len / 32};

    auto attn_output_memcfg_mine = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{
            CoreRangeSet({CoreRange(CoreCoord({0, 0}), CoreCoord({7, 3}))}),
            {32, padded_layer_past_len},
            ShardOrientation::ROW_MAJOR,
            false}};

    auto attn_weights = matmul(
        query_layer,
        key_layer_transposed,
        std::nullopt,  // bias
        attn_prog_config,
        attn_output_memcfg_mine,
        std::nullopt,  // output_dtype
        WormholeComputeKernelConfig{MathFidelity::HiFi2, true, true, true});
    query_layer.deallocate(true);
    key_layer_transposed.deallocate(true);

    auto softmax_progcfg = SoftmaxShardedMultiCoreProgramConfig{CoreCoord(8, 4), 1, 1, padded_layer_past_len / 32};
    auto attn_weights_softmax = scale_mask_softmax_in_place(attn_weights, scale, attn_masks, softmax_progcfg, true);

    // V cache update
    update_cache(V_cache, value_layer, start_pos, batch_offset);
    value_layer.deallocate(true);
    // unpad V
    auto value_layer_unpadded = unpad(V_cache, {0, 0, 0, 0}, {0, 31, padded_layer_past_len - 1, 127});
    auto value_layer_sharded = interleaved_to_sharded(value_layer_unpadded, kv_cache_mem_config);
    value_layer.deallocate(true);

    auto scores_prog_config = MatmulMultiCoreReuseProgramConfig{
        CoreCoord(8, 4),
        padded_layer_past_len / 32,
        1,
        1,
        1,
        128 / 32  // head_dim
    };

    auto scores_output_memcfg_mine = MemoryConfig{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{
            CoreRangeSet({CoreRange(CoreCoord{0, 0}, CoreCoord{7, 3})}),
            {32, 128},
            ShardOrientation::ROW_MAJOR,
            false}};

    auto attn_output = matmul(
        attn_weights_softmax,
        value_layer_sharded,
        std::nullopt,  // bias
        scores_prog_config,
        scores_output_memcfg_mine,
        std::nullopt,  // output_dtype
        WormholeComputeKernelConfig{MathFidelity::HiFi2, true, true, true});

    attn_weights_softmax.deallocate(true);
    value_layer_sharded.deallocate(true);

    return attn_output;
}

Tensor llama_attn_selfout_decode_forward(
    const Tensor& input_tensor, const Tensor& wo, MemoryConfig all_gather_memcfg, MemoryConfig mm_inp_memcfg) {
    auto attn_output = nlp_concat_heads_decode(input_tensor, 8);

    auto attn_output_gathered = tt::operations::ccl::all_gather(
        attn_output,
        3,  // dim
        1,  // num_links
        all_gather_memcfg);

    auto attn_output_reshard = reshard(attn_output_gathered, mm_inp_memcfg);

    auto attn_mm_output = matmul_1d(
        attn_output_reshard,
        wo,
        std::nullopt,  // bias
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            CoreCoord(8, 4),
            8,
            1,
            1,
            1,
            1,
            true,
            std::nullopt,  // fused activation
            true},
        MemoryConfig{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1},
        DataType::BFLOAT8_B,  // output_dtype
        WormholeComputeKernelConfig{MathFidelity::HiFi2, true, true, true});

    return attn_mm_output;
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
