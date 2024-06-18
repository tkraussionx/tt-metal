// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/downsample/downsample_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/validation.hpp"

namespace ttnn {

namespace operations {

namespace unary {

namespace detail {

inline const std::array<ttnn::TensorSchema, 1> input_tensor_schemas() {
    return {ttnn::TensorSchema{
        2,
        4,
        {ttnn::bfloat16, ttnn::bfloat8_b},
        {ttnn::TILE_LAYOUT, ttnn::ROW_MAJOR_LAYOUT},
        true,
        false,
        false,
        false}};
}

template <typename... Args>
inline auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
    return std::forward_as_tuple(input_tensor);
}

inline Tensor execute_on_worker_thread(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    DataType output_dtype = (op_chain[0].op_type == UnaryOpType::TYPECAST) ? static_cast<DataType>(op_chain[0].params[1]) : input_tensor.get_dtype();
    bool preserve_fp32_precision = (op_chain[0].op_type == UnaryOpType::TYPECAST) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision or
                            output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or
                            input_tensor.get_dtype() == DataType::UINT32 or
                            input_tensor.get_dtype() == DataType::INT32;  // MT: Currently only uint32/int32 is moved to
                                                                          // DST directly, fp32 is converted to fp16b
    return operation::run(
               EltwiseUnary{op_chain, memory_config.value_or(input_tensor.memory_config()), fp32_dest_acc_en, preserve_fp32_precision, output_dtype},
               {input_tensor}, {}, {optional_output_tensor}, queue_id).at(0);
}

}  // namespace detail

template <UnaryOpType... unary_op_types>
struct ExecuteUnary {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }
    static Tensor execute_on_worker_thread(
        uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(queue_id, input_tensor, {UnaryWithParam{unary_op_types}...}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(DefaultQueueId, input_tensor, {UnaryWithParam{unary_op_types}...}, memory_config, optional_output_tensor);
    }
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFastAndApproximateMode {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const bool parameter = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFloatParameter {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            queue_id, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}}, memory_config, optional_output_tensor);
    }
};

struct Softplus {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const float beta,
        const float threshold,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Softplus is not currently supported on Grayskull");
        return detail::execute_on_worker_thread(
            DefaultQueueId, input, {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}}, memory_config, optional_output_tensor);
    }
};

struct Sigmoid_accurate {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    static Tensor execute_on_worker_thread(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Softplus is not currently supported on Grayskull");
        return detail::execute_on_worker_thread(
            DefaultQueueId, input, {UnaryWithParam(UnaryOpType::NEG),
                                    UnaryWithParam(UnaryOpType::EXP, 1.0f),
                                    UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
                                    UnaryWithParam(UnaryOpType::RECIP)},
                                    memory_config,
                                    optional_output_tensor);
    }
};

struct Unary_chain {
    static const std::array<TensorSchema, 1> input_tensor_schemas() { return detail::input_tensor_schemas(); }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& input_tensor, Args&&... args) {
        return detail::input_tensors_to_validate(input_tensor, std::forward<Args>(args)...);
    }

    // static Tensor execute_on_worker_thread(
    //     uint8_t queue_id,
    //     const Tensor& input_tensor,
    //     const std::vector<UnaryWithParam>& ops_chain,
    //     const std::optional<MemoryConfig>& memory_config = std::nullopt,
    //     const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    //     return detail::execute_on_worker_thread(
    //         queue_id, input_tensor, ops_chain, memory_config, optional_output_tensor);
    // }

    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor,
        const std::vector<UnaryWithParam>& ops_chain,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return detail::execute_on_worker_thread(
            DefaultQueueId, input_tensor, ops_chain, memory_config, optional_output_tensor);
    }
};

// tteager composite unary op => ttnn composite unary op?
Tensor deg2rad(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    // return tt::tt_metal::deg2rad(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    return multiply(queue_id, input_tensor, (float)(M_PI / 180.0), std::nullopt, memory_config.value_or(input_tensor.memory_config()), optional_output_tensor);
}
Tensor rad2deg(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    // return tt::tt_metal::rad2deg(input_tensor, memory_config.value_or(input_tensor.memory_config()));
    return multiply(queue_id, input_tensor, (float)(180.0 / M_PI), std::nullopt, memory_config.value_or(input_tensor.memory_config()), optional_output_tensor);
}

// TODO: update these composite unary ops pending decision on TensorAsync implementation.

Tensor acosh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::acosh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}

Tensor asinh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::asinh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}

Tensor atanh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::atanh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}

Tensor cbrt(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::cbrt(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor cosh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::cosh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
// Tensor deg2rad(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
//     return tt::tt_metal::deg2rad(input_tensor, memory_config.value_or(input_tensor.memory_config()));
// }
Tensor digamma(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::digamma(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor hardswish(
    const Tensor& input_tensor,
    float scale,
    float shift,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    //return tt::tt_metal::hardswish(input_tensor, scale, shift, memory_config.value_or(input_tensor.memory_config()));
    return tt::tt_metal::hardswish(input_tensor, scale, shift, memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG));
}
Tensor hardsigmoid(
    const Tensor& input_tensor,
    float scale,
    float shift,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    //return tt::tt_metal::hardsigmoid(input_tensor, scale, shift, memory_config.value_or(input_tensor.memory_config()));
    return tt::tt_metal::hardsigmoid(input_tensor, scale, shift, memory_config.value_or(operation::DEFAULT_OUTPUT_MEMORY_CONFIG));
}
Tensor hardtanh(
    const Tensor& input_tensor,
    float low /* = -1.0f */,
    float high /* = +1.0f */,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return tt::tt_metal::hardtanh(input_tensor, low, high, memory_config.value_or(input_tensor.memory_config()));
}
Tensor lgamma(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::lgamma(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor log1p(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::log1p(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor mish(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::mish(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor multigammaln(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::multigammaln(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
// Tensor rad2deg(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
//     return tt::tt_metal::rad2deg(input_tensor, memory_config.value_or(input_tensor.memory_config()));
// }
// Tensor sigmoid_accurate(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
//     return tt::tt_metal::sigmoid_accurate(input_tensor, memory_config.value_or(input_tensor.memory_config()));
// }
Tensor sinh(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::sinh(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor softsign(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::softsign(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor swish(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::swish(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor tanhshrink(uint8_t queue_id, const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt, const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return tt::tt_metal::tanhshrink(input_tensor, memory_config.value_or(input_tensor.memory_config()));
}
Tensor tril(
    const Tensor& input_tensor,
    int32_t diag=0,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return tt::tt_metal::tril(input_tensor, diag, memory_config.value_or(input_tensor.memory_config()));
}
Tensor triu(
    const Tensor& input_tensor,
    int32_t diag=0,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return tt::tt_metal::triu(input_tensor, diag, memory_config.value_or(input_tensor.memory_config()));
}

}  // namespace unary
}  // namespace operations

#define REGISTER_UNARY_OPERATION(operation_name, operation_type)                                      \
    constexpr auto operation_name = ttnn::register_operation<                                         \
        ttnn::operations::unary::ExecuteUnary<ttnn::operations::unary::UnaryOpType::operation_type>>( \
        "ttnn::" #operation_name);

#define REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(operation_name, operation_type)   \
    constexpr auto operation_name =                                                               \
        ttnn::register_operation<ttnn::operations::unary::ExecuteUnaryWithFastAndApproximateMode< \
            ttnn::operations::unary::UnaryOpType::operation_type>>("ttnn::" #operation_name);

#define REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(operation_name, operation_type)                                 \
    constexpr auto operation_name = ttnn::register_operation<ttnn::operations::unary::ExecuteUnaryWithFloatParameter< \
        ttnn::operations::unary::UnaryOpType::operation_type>>("ttnn::" #operation_name);

REGISTER_UNARY_OPERATION(abs, ABS);
REGISTER_UNARY_OPERATION(acos, ACOS);
REGISTER_UNARY_OPERATION(asin, ASIN);
REGISTER_UNARY_OPERATION(atan, ATAN);
REGISTER_UNARY_OPERATION(cos, COS);
REGISTER_UNARY_OPERATION(erfinv, ERFINV);
REGISTER_UNARY_OPERATION(exp2, EXP2);
REGISTER_UNARY_OPERATION(expm1, EXPM1);
REGISTER_UNARY_OPERATION(eqz, EQZ);
REGISTER_UNARY_OPERATION(gez, GEZ);
REGISTER_UNARY_OPERATION(gtz, GTZ);
REGISTER_UNARY_OPERATION(i0, I0);
REGISTER_UNARY_OPERATION(isfinite, ISFINITE);
REGISTER_UNARY_OPERATION(isinf, ISINF);
REGISTER_UNARY_OPERATION(isnan, ISNAN);
REGISTER_UNARY_OPERATION(isneginf, ISNEGINF);
REGISTER_UNARY_OPERATION(isposinf, ISPOSINF);
REGISTER_UNARY_OPERATION(lez, LEZ);
REGISTER_UNARY_OPERATION(log, LOG);
REGISTER_UNARY_OPERATION(log10, LOG10);
REGISTER_UNARY_OPERATION(log2, LOG2);
REGISTER_UNARY_OPERATION(logical_not, LOGICAL_NOT_UNARY);
REGISTER_UNARY_OPERATION(ltz, LTZ);
REGISTER_UNARY_OPERATION(neg, NEG);
REGISTER_UNARY_OPERATION(nez, NEZ);
REGISTER_UNARY_OPERATION(reciprocal, RECIP);
REGISTER_UNARY_OPERATION(relu, RELU);
REGISTER_UNARY_OPERATION(relu6, RELU6);
REGISTER_UNARY_OPERATION(sigmoid, SIGMOID);
REGISTER_UNARY_OPERATION(sign, SIGN);
REGISTER_UNARY_OPERATION(signbit, SIGNBIT);
REGISTER_UNARY_OPERATION(silu, SILU);
REGISTER_UNARY_OPERATION(sin, SIN);
REGISTER_UNARY_OPERATION(sqrt, SQRT);
REGISTER_UNARY_OPERATION(square, SQUARE);
REGISTER_UNARY_OPERATION(tan, TAN);
REGISTER_UNARY_OPERATION(tanh, TANH);

constexpr auto log_sigmoid = ttnn::register_operation<ttnn::operations::unary::ExecuteUnary<
    ttnn::operations::unary::UnaryOpType::SIGMOID,
    ttnn::operations::unary::UnaryOpType::LOG>>("ttnn::log_sigmoid");

// Unaries with fast_and_approximate_mode
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(exp, EXP);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erf, ERF);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erfc, ERFC);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(gelu, GELU);
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt, RSQRT);

// Unaries with float parameter
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(elu, ELU);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(heaviside, HEAVISIDE);
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(leaky_relu, LEAKY_RELU);
auto prelu = leaky_relu;  // Alias for leaky_relu. TODO(#8544): implement PReLU properly

// Other unaries
constexpr auto softplus = ttnn::register_operation<ttnn::operations::unary::Softplus>("ttnn::softplus");
constexpr auto sigmoid_accurate = ttnn::register_operation<ttnn::operations::unary::Sigmoid_accurate>("ttnn::sigmoid_accurate");

constexpr auto unary_chain = ttnn::register_operation<ttnn::operations::unary::Unary_chain>("ttnn::unary_chain");

constexpr auto acosh = ttnn::register_operation("ttnn::acosh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::acosh));
constexpr auto asinh = ttnn::register_operation("ttnn::asinh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::asinh));
constexpr auto atanh = ttnn::register_operation("ttnn::atanh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::atanh));
constexpr auto cbrt = ttnn::register_operation("ttnn::cbrt", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::cbrt));
constexpr auto cosh = ttnn::register_operation("ttnn::cosh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::cosh));
constexpr auto deg2rad = ttnn::register_operation("ttnn::deg2rad", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::deg2rad));
constexpr auto digamma = ttnn::register_operation("ttnn::digamma", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::digamma));
constexpr auto hardswish = ttnn::register_operation("ttnn::hardswish", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::hardswish));
constexpr auto hardsigmoid =
    ttnn::register_operation("ttnn::hardsigmoid", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::hardsigmoid));
constexpr auto hardtanh = ttnn::register_operation("ttnn::hardtanh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::hardtanh));
constexpr auto lgamma = ttnn::register_operation("ttnn::lgamma", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::lgamma));
constexpr auto log1p = ttnn::register_operation("ttnn::log1p", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::log1p));
constexpr auto mish = ttnn::register_operation("ttnn::mish", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::mish));
constexpr auto multigammaln =
    ttnn::register_operation("ttnn::multigammaln", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::multigammaln));
constexpr auto rad2deg = ttnn::register_operation("ttnn::rad2deg", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::rad2deg));
// constexpr auto sigmoid_accurate =
//     ttnn::register_operation("ttnn::sigmoid_accurate", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::sigmoid_accurate));
constexpr auto sinh = ttnn::register_operation("ttnn::sinh", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::sinh));
constexpr auto softsign = ttnn::register_operation("ttnn::softsign", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::softsign));
constexpr auto swish = ttnn::register_operation("ttnn::swish", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::swish));
constexpr auto tanhshrink =
    ttnn::register_operation("ttnn::tanhshrink", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::tanhshrink));
constexpr auto tril = ttnn::register_operation("ttnn::tril", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::tril));
constexpr auto triu = ttnn::register_operation("ttnn::triu", TO_LAMBDA_WITH_RESHAPE(ttnn::operations::unary::triu));

}  // namespace ttnn
