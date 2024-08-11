// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_relu_device_operation.hpp"
#include "tt_metal/common/assert.hpp"
#include <iostream>
namespace ttnn::operations::moreh_eltwise {

tt::stl::hash::hash_t MorehReluDeviceOperation::compute_program_hash(
    const operation_attributes_t &attributes,
    const tensor_args_t &tensor_args) {

  // reflect::for_each(
  //     [&attributes](auto I) {
  //       const auto &attribute_name = reflect::member_name<I>(attributes);
  //       const auto &attribute = reflect::get<I>(attributes);
  //       std::cout << "moreh_relu reflection" << std::endl;
  //       std::cout << attribute_name << " " << attribute << std::endl;
  //     },
  //     attributes);

  return tt::stl::hash::hash_objects_with_default_seed(
      tt::stl::hash::type_hash<MorehReluDeviceOperation>,
      attributes.which_relu, tensor_args.input_tensor,
      tensor_args.output_tensor);

  // return tt::stl::hash::hash_objects_with_default_seed(
  //     tt::stl::hash::type_hash<MorehReluDeviceOperation>, attributes,
  //     tensor_args);
}

void MorehReluDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t &attributes,
    const tensor_args_t &tensor_args) {

  TT_FATAL(0 <= attributes.which_relu && attributes.which_relu <= 3,
           "which_relu must be 0, 1 or 2");
  TT_FATAL(attributes.bound >= 0, "Upper bound must be not negative.");

  const auto &input_tensor = tensor_args.input_tensor;
  TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16,
           "input tensor data format must be bfloat16");
  TT_FATAL(input_tensor.get_layout() == Layout::TILE,
           "input tensor layout must be TILE");
  TT_FATAL(input_tensor.device_buffer()->buffer_type() == BufferType::DRAM);
  if (tensor_args.output_tensor) {
    const auto output_tensor = tensor_args.output_tensor.value();
    TT_FATAL(output_tensor.get_dtype() == DataType::BFLOAT16,
             "output tensor data format must be bfloat16");
    TT_FATAL(output_tensor.get_layout() == Layout::TILE,
             "output tensor layout must be TILE");
  }
}

void MorehReluDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t &attributes,
    const tensor_args_t &tensor_args) {
  TT_FATAL(attributes.bound >= 0, "Upper bound must be not negative.");
  // Cache hit means that the other values used in
  // create_program are same.
}

MorehReluDeviceOperation::shape_return_value_t
MorehReluDeviceOperation::compute_output_shapes(
    const operation_attributes_t &, const tensor_args_t &tensor_args) {
  return tensor_args.input_tensor.tensor_attributes->shape;
}

MorehReluDeviceOperation::tensor_return_value_t
MorehReluDeviceOperation::create_output_tensors(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args) {
  if (tensor_args.output_tensor) {
    return tensor_args.output_tensor.value();
  } else {
    auto output_shape =
        compute_output_shapes(operation_attributes, tensor_args);
    const auto &input_tensor = tensor_args.input_tensor;
    return create_device_tensor(
        output_shape, input_tensor.tensor_attributes->dtype,
        input_tensor.tensor_attributes->layout, input_tensor.device());
  }
}

MorehReluDeviceOperation::program_factory_t
MorehReluDeviceOperation::select_program_factory(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args) {
  return MultiCore{};
}

} // namespace ttnn::operations::moreh_eltwise
