// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_relu_device_operation.hpp"
#include "tt_metal/common/assert.hpp"
namespace ttnn::operations::moreh_eltwise {

MorehReluDeviceOperation::program_factory_t
MorehReluDeviceOperation::select_program_factory(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args) {
  return MultiCore{};
}

void MorehReluDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t &attributes,
    const tensor_args_t &tensor_args) {
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
    const tensor_args_t &tensor_args) {}

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

} // namespace ttnn::operations::moreh_eltwise
