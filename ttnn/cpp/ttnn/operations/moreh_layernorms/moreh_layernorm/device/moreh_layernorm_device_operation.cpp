// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layernorm_device_operation.hpp"

namespace ttnn::operations::moreh_layernorms {

MorehLayernormDeviceOperation::program_factory_t MorehLayernormDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
    // if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
        return SingleCore{};
    // }
    // return MultiCore{};
}

void MorehLayernormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void MorehLayernormDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

MorehLayernormDeviceOperation::shape_return_value_t MorehLayernormDeviceOperation::compute_output_shapes(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {

    auto output_shape = tensor_args.input.tensor_attributes->shape;
    return {output_shape, output_shape, output_shape};
}

MorehLayernormDeviceOperation::tensor_return_value_t MorehLayernormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    const auto& input = tensor_args.input;

    auto output = input;
    return {output, output, output};
}

}  // namespace ttnn::operations::moreh_layernorms
