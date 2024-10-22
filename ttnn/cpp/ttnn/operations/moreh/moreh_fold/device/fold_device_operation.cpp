// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::moreh::moreh_fold {
void MorehFoldOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& input = tensor_args.input;
    auto input_shape = input.get_logical_shape();

    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Fold: only support input in ROW_MAJOR");
    TT_FATAL(operation_attributes.output_size.size() == 2, "Fold: output_size takes 2 elements");
    TT_FATAL(operation_attributes.kernel_size.size() == 2, "Fold: kernel_size takes 2 elements");
    TT_FATAL(operation_attributes.dilation.size() == 2, "Fold: dilation takes 2 elements");
    TT_FATAL(operation_attributes.padding.size() == 2, "Fold: padding takes 2 elements");
    TT_FATAL(operation_attributes.stride.size() == 2, "Fold: stride takes 2 elements");

    uint32_t kernel_size_product = 1;
    uint32_t l = 1;
    for (uint32_t i = 0; i < 2; ++i) {
        l *=
            (((operation_attributes.output_size[i] + 2 * operation_attributes.padding[i] -
               operation_attributes.dilation[i] * (operation_attributes.kernel_size[i] - 1) - 1) /
              operation_attributes.stride[i]) +
             1);
        kernel_size_product *= operation_attributes.kernel_size[i];
    }
    auto input_rank = input.get_logical_shape().rank();

    TT_FATAL((input_rank == 3) || (input_rank == 2), "Fold: Only support 3D or 2D input tensor");
    TT_FATAL(input_shape[input_rank - 1] == l, "Fold: Invalid input tensor size");
    TT_FATAL(input_shape[input_rank - 2] % kernel_size_product == 0, "Fold: Invalid input tensor size");

    if (tensor_args.output) {
        auto output_shape = tensor_args.output->get_logical_shape();
        auto output_rank = tensor_args.output->get_logical_shape().rank();
        TT_FATAL(
            output_shape[output_rank - 3] == input_shape[input_rank - 2] / kernel_size_product,
            "Fold: Invalid output tensor size");
        TT_FATAL(
            output_shape[output_rank - 2] == operation_attributes.output_size[0], "Fold: Invalid output shape size");
        TT_FATAL(
            output_shape[output_rank - 1] == operation_attributes.output_size[1], "Fold: Invalid output shape size");
        TT_FATAL(output_rank == 4 || output_rank == 3, "Fold: Only support 4D and 3D output tensor");
    }
}

MorehFoldOperation::program_factory_t MorehFoldOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
};

void MorehFoldOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehFoldOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehFoldOperation::shape_return_value_t MorehFoldOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor_shape = tensor_args.input.get_logical_shape();
    auto input_tensor_rank = tensor_args.input.get_logical_shape().rank();
    uint32_t kernel_size_product = operation_attributes.kernel_size[0] * operation_attributes.kernel_size[1];
    if (input_tensor_rank == 3) {
        uint32_t N = input_tensor_shape[0];
        uint32_t C = input_tensor_shape[1] / kernel_size_product;
        auto output_shape = ttnn::SimpleShape(
            std::vector<uint32_t>{N, C, operation_attributes.output_size[0], operation_attributes.output_size[1]});
        return output_shape;
    }
    // If input_tensor_rank == 2
    uint32_t C = input_tensor_shape[0] / kernel_size_product;
    auto output_shape = ttnn::SimpleShape(
        std::vector<uint32_t>{C, operation_attributes.output_size[0], operation_attributes.output_size[1]});
    return output_shape;
};

MorehFoldOperation::tensor_return_value_t MorehFoldOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output.value();
    }

    const auto& output_shape = compute_output_shapes(operation_attributes, tensor_args);

    return create_device_tensor(
        output_shape,
        tensor_args.input.get_dtype(),
        tensor_args.input.get_layout(),
        tensor_args.input.device(),
        operation_attributes.memory_config);
}

std::tuple<MorehFoldOperation::operation_attributes_t, MorehFoldOperation::tensor_args_t> MorehFoldOperation::invoke(
    const Tensor& input,
    const std::optional<Tensor>& output,
    const std::vector<uint32_t> output_size,
    const std::vector<uint32_t> kernel_size,
    const std::vector<uint32_t> dilation,
    const std::vector<uint32_t> padding,
    const std::vector<uint32_t> stride,
    const std::optional<MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{
            output_size, kernel_size, dilation, padding, stride, memory_config.value_or(input.memory_config())},
        tensor_args_t{input, output}};
}
}  // namespace ttnn::operations::moreh::moreh_fold
