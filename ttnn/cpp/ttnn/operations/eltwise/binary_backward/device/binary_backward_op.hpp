// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary_backward {

constexpr uint8_t DefaultQueueId = 0;
enum class BinaryBackwardOpType {
    ATAN2_BW,
    EMBEDDING_BW,
    ADDALPHA_BW,
    SUBALPHA_BW,
    SUB_BW,
    XLOGY_BW,
    HYPOT_BW,
    LDEXP_BW,
    LOGADDEXP_BW,
    LOGADDEXP2_BW,
    SQUARED_DIFFERENCE_BW,
    ADD_BW,
    EQ_BW,
    ASSIGN_BW,
    CONCAT_BW,
    BINARY_LE_BW,
    RSUB_BW,
    BIAS_GELU_BW,
    BINARY_GT_BW,
    BINARY_LT_BW,
    BINARY_NE_BW,
    BINARY_GE_BW,
    MIN_BW,
    MAX_BW,
    DIV_BW,
    LERP_BW,
    MUL_BW,
};

std::vector<ttnn::Tensor> _atan2_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<ttnn::Tensor> _embedding_bw(
    const Tensor& grad, const Tensor& input, const Tensor& weight, const MemoryConfig& output_mem_config);

std::vector<std::optional<ttnn::Tensor>> _addalpha_bw(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad);

std::vector<ttnn::Tensor> _addalpha_bw_inter(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config);

std::vector<std::optional<Tensor>> _addalpha_bw_overload(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    float alpha,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad);

std::vector<ttnn::Tensor> _subalpha_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config);

std::vector<ttnn::Tensor> _sub_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<std::optional<Tensor>> _add_bw(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad);

std::vector<ttnn::Tensor> _add_bw_inter(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<std::optional<Tensor>> _add_bw_overload(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad);

std::vector<ttnn::Tensor> _xlogy_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<ttnn::Tensor> _hypot_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<ttnn::Tensor> _ldexp_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<ttnn::Tensor> _logaddexp_bw(
    const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<ttnn::Tensor> _logaddexp2_bw(
    const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config);


std::vector<ttnn::Tensor> _squared_difference_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<std::optional<Tensor>> _eq_bw(
    uint8_t cq_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad);

std::vector<ttnn::Tensor> _eq_bw_inter(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<std::optional<Tensor>> _eq_bw_overload(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad);

std::vector<Tensor> _assign_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<Tensor> _concat_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, int dim, const MemoryConfig& output_mem_config);

std::vector<Tensor> _binary_comp_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<Tensor> _rsub_bw( const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) ;

std::vector<Tensor> _bias_gelu_bw(
    const Tensor& grad,
    const Tensor& input_a,
    const Tensor& input_b,
    string approximate,
    const MemoryConfig& output_mem_config);

std::vector<Tensor> _binary_gt_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) ;

std::vector<Tensor> _binary_ne_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) ;

std::vector<Tensor> _binary_ge_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) ;

std::vector<Tensor> _binary_le_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) ;

std::vector<Tensor> _binary_lt_bw(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<Tensor> _min_or_max_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<Tensor> _div_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    string round_mode,
    const MemoryConfig& output_mem_config);

std::vector<Tensor> _lerp_bw(
    const Tensor& grad, const Tensor& input, const Tensor& end, float weight, const MemoryConfig& output_mem_config);

std::vector<std::optional<Tensor>> _mul_bw(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad);

std::vector<ttnn::Tensor> _mul_bw_inter(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config);

std::vector<std::optional<Tensor>> _mul_bw_overload(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad);


// OpHandler struct template
template <BinaryBackwardOpType OpType>
struct OpHandler_type1;

template <BinaryBackwardOpType OpType>
struct OpHandler_type1_w_float;

template <BinaryBackwardOpType OpType>
struct OpHandler_type1_w_string;

template <BinaryBackwardOpType OpType>
struct OpHandler_type3;

template <BinaryBackwardOpType OpType>
struct OpHandler_type3_wo_qid;

template <BinaryBackwardOpType OpType>
struct OpHandler_type2;

template <BinaryBackwardOpType OpType>
struct OpHandler_type2_wo_qid;


template <>
struct OpHandler_type1<BinaryBackwardOpType::ATAN2_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _atan2_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::EMBEDDING_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& weight, const MemoryConfig& output_mem_config) {
        return _embedding_bw(grad, input, weight, output_mem_config);
    }
};

template <>
struct OpHandler_type2<BinaryBackwardOpType::ADDALPHA_BW> {
    static std::vector<std::optional<Tensor>> handle(
        uint8_t queue_id,
        const Tensor& grad,
        const Tensor& input,
        const Tensor& other,
        float alpha,
        const MemoryConfig& output_mem_config,
        const std::vector<bool>& are_required_outputs,
        std::optional<Tensor> input_grad,
        std::optional<Tensor> other_grad) {
        return _addalpha_bw(queue_id, grad, input, other, alpha, output_mem_config, are_required_outputs, input_grad, other_grad);
    }
};

template <>
struct OpHandler_type1_w_float<BinaryBackwardOpType::SUBALPHA_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, float alpha, const MemoryConfig& output_mem_config) {
        return _subalpha_bw(grad, input, other, alpha, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::SUB_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _sub_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::XLOGY_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _xlogy_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::HYPOT_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _hypot_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::LDEXP_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _ldexp_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::LOGADDEXP_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _logaddexp_bw(grad, input_a, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::LOGADDEXP2_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input_a, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _logaddexp2_bw(grad, input_a, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::SQUARED_DIFFERENCE_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _squared_difference_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type3<BinaryBackwardOpType::ADD_BW> {
    static std::vector<std::optional<Tensor>> handle(
        uint8_t queue_id,
        const Tensor& grad,
        const Tensor& input,
        const Tensor& other,
        const MemoryConfig& output_mem_config,
        const std::vector<bool>& are_required_outputs,
        std::optional<Tensor> input_grad,
        std::optional<Tensor> other_grad) {
        return _add_bw(queue_id, grad, input, other, output_mem_config, are_required_outputs, input_grad, other_grad);
    }
};

template <>
struct OpHandler_type3<BinaryBackwardOpType::EQ_BW> {
    static std::vector<std::optional<Tensor>> handle(
        uint8_t cq_id,
        const Tensor& grad,
        const Tensor& input,
        const Tensor& other,
        const MemoryConfig& output_mem_config,
        const std::vector<bool>& are_required_outputs,
        std::optional<Tensor> input_grad,
        std::optional<Tensor> other_grad) {
        return _eq_bw(cq_id, grad, input, other, output_mem_config, are_required_outputs, input_grad, other_grad);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::ASSIGN_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _assign_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1_w_float<BinaryBackwardOpType::CONCAT_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, int dim, const MemoryConfig& output_mem_config) {
        return _concat_bw(grad, input, other, dim, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::BINARY_LE_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _binary_le_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::RSUB_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _rsub_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1_w_string<BinaryBackwardOpType::BIAS_GELU_BW> {
    static std::vector<Tensor> handle(
        const Tensor& grad,
        const Tensor& input_a,
        const Tensor& input_b,
        std::string approximate,
        const MemoryConfig& output_mem_config) {
        return _bias_gelu_bw(grad, input_a, input_b, approximate, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::BINARY_GT_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _binary_gt_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::BINARY_LT_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _binary_lt_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::BINARY_NE_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _binary_ne_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::BINARY_GE_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _binary_ge_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::MIN_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _min_or_max_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1<BinaryBackwardOpType::MAX_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
        return _min_or_max_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler_type1_w_string<BinaryBackwardOpType::DIV_BW> {
    static std::vector<Tensor> handle(
        const Tensor& grad,
        const Tensor& input,
        const Tensor& other,
        std::string round_mode,
        const MemoryConfig& output_mem_config) {
        return _div_bw(grad, input, other, round_mode, output_mem_config);
    }
};

template <>
struct OpHandler_type1_w_float<BinaryBackwardOpType::LERP_BW> {
    static std::vector<Tensor> handle(const Tensor& grad, const Tensor& input, const Tensor& end, float weight, const MemoryConfig& output_mem_config) {
        return _lerp_bw(grad, input, end, weight, output_mem_config);
    }
};

template <>
struct OpHandler_type3<BinaryBackwardOpType::MUL_BW> {
    static std::vector<std::optional<Tensor>> handle(
        uint8_t queue_id,
        const Tensor& grad,
        const Tensor& input,
        const Tensor& other,
        const MemoryConfig& output_mem_config,
        const std::vector<bool>& are_required_outputs,
        std::optional<Tensor> input_grad,
        std::optional<Tensor> other_grad) {
        return _mul_bw(queue_id, grad, input, other, output_mem_config, are_required_outputs, input_grad, other_grad);
    }
};



// Template functions to get the function pointers
template <BinaryBackwardOpType OpType>
auto get_function_type1() {
    return &OpHandler_type1<OpType>::handle;
}
template <BinaryBackwardOpType OpType>
auto get_function_type1_w_float() {
    return &OpHandler_type1_w_float<OpType>::handle;
}
template <BinaryBackwardOpType OpType>
auto get_function_type1_w_string() {
    return &OpHandler_type1_w_string<OpType>::handle;
}
template <BinaryBackwardOpType OpType>
auto get_function_type3() {
    return &OpHandler_type3<OpType>::handle;
}
template <BinaryBackwardOpType OpType>
auto get_function_type3_wo_qid() {
    return &OpHandler_type3_wo_qid<OpType>::handle;
}
template <BinaryBackwardOpType OpType>
auto get_function_type2() {
    return &OpHandler_type2<OpType>::handle;
}
template <BinaryBackwardOpType OpType>
auto get_function_type2_wo_qid() {
    return &OpHandler_type2_wo_qid<OpType>::handle;
}


}  // namespace ttnn::operations::binary_backward
