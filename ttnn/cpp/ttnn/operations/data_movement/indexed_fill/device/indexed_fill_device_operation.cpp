// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexed_fill_device_operation.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace ttnn::operations::data_movement {
IndexedFillOperation::program_factory_t IndexedFillOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // For now we litteraly don't care and return a single factory. Whatever
    return MultiCore{};
}

void IndexedFillOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;
    const auto& batch_ids = tensor_args.batch_ids;
    auto input_a_shape = input_a.get_shape();
    auto input_b_shape = input_b.get_shape();
    TT_FATAL(input_a.get_layout() == Layout::ROW_MAJOR, "Currently only supporting row major layout");
    TT_FATAL(input_b.get_layout() == input_a.get_layout(), "Inputs must be same layout");
    TT_FATAL(input_a_shape[1] == input_b_shape[1] &&
            input_a_shape[2] == input_b_shape[2] &&
            input_a_shape[3] == input_b_shape[3]
            , "Dims except batch dim must be the same on inputs");
    TT_FATAL(input_b_shape[0] == batch_ids.get_legacy_shape()[-1], "Second input and batch ids must be same outer dim");
    TT_FATAL(batch_ids.get_layout() == Layout::ROW_MAJOR, "Batch IDs must be ROW MAJOR");
    TT_FATAL(operation_attributes.dim == 0, "Currently only supporting batch dimension");
    TT_FATAL(input_a.storage_type() == StorageType::DEVICE, "Operands to Index Fill need to be on device!");
    TT_FATAL(input_a.buffer() != nullptr , "Operands to Index Fill need to be allocated in buffers on device!");
    TT_FATAL(input_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(input_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Index Fill does not currently support sharding");

}

void IndexedFillOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

void IndexedFillOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

IndexedFillOperation::shape_return_value_t IndexedFillOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input_a.get_shape();
}


IndexedFillOperation::tensor_return_value_t IndexedFillOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    const auto& input = tensor_args.input_a;
    return create_device_tensor(
        output_shape,
        input.tensor_attributes->dtype,
        input.tensor_attributes->layout,
        input.device(),
        operation_attributes.memory_config);
}

std::tuple<IndexedFillOperation::operation_attributes_t, IndexedFillOperation::tensor_args_t>
IndexedFillOperation::invoke(
        const Tensor &batch_ids,
        const Tensor &input_a,
        const Tensor &input_b,
        const int64_t dim,
        const std::optional<MemoryConfig> &memory_config) {
    return {
        operation_attributes_t{dim, memory_config.value_or(input_a.memory_config())},
        tensor_args_t{batch_ids, input_a, input_b}
    };
}

}
