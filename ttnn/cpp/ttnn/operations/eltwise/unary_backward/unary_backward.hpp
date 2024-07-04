
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary_backward/device/unary_backward_op.cpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::unary_backward {

template <UnaryBackwardOpType unary_backward_op_type>
struct ExecuteUnaryBackward {

    static inline const std::array<TensorSchema, 2> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false}};
    }

    static inline std::vector<ttnn::Tensor> create_async_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
        const auto& input_tensor = input_tensors.at(0);
        return {Tensor(operation::get_workers_for_op_output({input_tensor}))};
    }


        //binary overload
    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    }

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {

        auto op_type = utils::unary_get_function_overload_type1(unary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config);
        }

    // //binary overload
    // template <typename... Args>
    // static auto input_tensors_to_validate(uint8_t queue_id, const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, Args &&...args) {
    //     return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    // }

    // static std::vector<std::optional<ttnn::Tensor>> execute_on_worker_thread(
    //     uint8_t queue_id,
    //     const Tensor &grad_tensor_arg,
    //     const Tensor &input_tensor_a_arg,
    //     const Tensor &input_tensor_b_arg,
    //     const std::optional<MemoryConfig> &memory_config = std::nullopt,
    //     const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    //     std::optional<Tensor> input_a_grad = std::nullopt,
    //     std::optional<Tensor> input_b_grad = std::nullopt) {

    //     auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
    //     auto op_type = utils::unary_get_function_overload_type2(unary_backward_op_type);
    //     return op_type(queue_id, grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    // }

    // //binary overload
    // template <typename... Args>
    // static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor_a, const Tensor &input_tensor_b, std::vector<bool> are_required_outputs, Args &&...args) {
    //     return std::forward_as_tuple(grad_tensor, input_tensor_a, input_tensor_b);
    // }

    // static std::vector<std::optional<ttnn::Tensor>> execute_on_worker_thread(
    //     const Tensor &grad_tensor_arg,
    //     const Tensor &input_tensor_a_arg,
    //     const Tensor &input_tensor_b_arg,
    //     const std::optional<MemoryConfig> &memory_config = std::nullopt,
    //     const std::vector<bool>& are_required_outputs = std::vector<bool>{true, true},
    //     std::optional<Tensor> input_a_grad = std::nullopt,
    //     std::optional<Tensor> input_b_grad = std::nullopt) {

    //     auto output_memory_config = memory_config.value_or(input_tensor_a_arg.memory_config());
    //     auto op_type = utils::unary_get_function_overload_type2_wo_qid(unary_backward_op_type);
    //     return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, output_memory_config, are_required_outputs, input_a_grad, input_b_grad);
    // }

    //Type 1: 2 inputs, 1 grad tensor
    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &grad_tensor, const Tensor &input_tensor, Args &&...args) {
        return std::forward_as_tuple(grad_tensor, input_tensor);
    }

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {

        auto op_type = utils::unary_get_function_type1(unary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, output_memory_config);
        }

    //Type 1: Type 1 with 1 float
    template <typename... Args>

    static std::vector<ttnn::Tensor> execute_on_worker_thread(
        const Tensor &grad_tensor_arg,
        const Tensor &input_tensor_arg,
        float alpha,
        const std::optional<MemoryConfig> &memory_config = std::nullopt) {

        auto op_type = utils::unary_get_function_type1_w_float(unary_backward_op_type);
        auto output_memory_config = memory_config.value_or(input_tensor_arg.memory_config());
        return op_type(grad_tensor_arg, input_tensor_arg, alpha, output_memory_config);
        }

};

}  // operations::unary

//type 1
constexpr auto mul_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::MUL_BW>>("ttnn::mul_bw");
constexpr auto clamp_min_bw = ttnn::register_operation<operations::unary_backward::ExecuteUnaryBackward<operations::unary_backward::UnaryBackwardOpType::CLAMP_MIN_BW>>("ttnn::clamp_min_bw");

}  // namespace ttnn
