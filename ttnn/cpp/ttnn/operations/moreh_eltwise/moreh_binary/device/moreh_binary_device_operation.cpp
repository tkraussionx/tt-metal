// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_binary_device_operation.hpp"
#include "tt_metal/common/assert.hpp"
#include <iostream>
namespace ttnn::operations::moreh_eltwise {

tt::stl::hash::hash_t MorehBinaryDeviceOperation::compute_program_hash(
    const operation_attributes_t &attributes,
    const tensor_args_t &tensor_args) {
  /* TODO */
  // TODO. Use all tensors and program_selector of scalars in hash
  // Do not use scalar0 and scalar1 in attributes
  return tt::stl::hash::hash_objects_with_default_seed(
      tt::stl::hash::type_hash<MorehBinaryDeviceOperation>);
}

void MorehBinaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t &attributes,
    const tensor_args_t &tensor_args) {

  TT_FATAL(attributes.program_selector == 0 or attributes.program_selector == 1,
           "0 is binary add. 1 is exercise for SFPU");

  auto check_tensor = [](auto t) {
    TT_FATAL(t.get_dtype() == DataType::BFLOAT16,
             "input tensor data format must be bfloat16");
    TT_FATAL(t.get_layout() == Layout::TILE,
             "input tensor layout must be TILE");
    TT_FATAL(t.device_buffer()->buffer_type() == BufferType::DRAM);
  };

  check_tensor(tensor_args.input_tensor0);
  check_tensor(tensor_args.input_tensor1);

  if (tensor_args.output_tensor) {
    check_tensor(tensor_args.output_tensor.value());
  }
}

void MorehBinaryDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t &attributes,
    const tensor_args_t &tensor_args) {}

MorehBinaryDeviceOperation::shape_return_value_t
MorehBinaryDeviceOperation::compute_output_shapes(
    const operation_attributes_t &, const tensor_args_t &tensor_args) {
  return tensor_args.input_tensor0.tensor_attributes->shape;
}

MorehBinaryDeviceOperation::tensor_return_value_t
MorehBinaryDeviceOperation::create_output_tensors(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args) {
  if (tensor_args.output_tensor) {
    return tensor_args.output_tensor.value();
  } else {
    auto output_shape =
        compute_output_shapes(operation_attributes, tensor_args);
    const auto &input_tensor0 = tensor_args.input_tensor0;
    return create_device_tensor(
        output_shape, input_tensor0.tensor_attributes->dtype,
        input_tensor0.tensor_attributes->layout, input_tensor0.device());
  }
}

MorehBinaryDeviceOperation::program_factory_t
MorehBinaryDeviceOperation::select_program_factory(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args) {

  // TODO. Return Add if program_selector is 0 . Return Funsion if
  // prograam_selector is 1.

  return Fusion{};
}

} // namespace ttnn::operations::moreh_eltwise
