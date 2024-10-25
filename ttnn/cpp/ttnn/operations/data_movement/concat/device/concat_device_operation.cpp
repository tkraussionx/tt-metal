// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/common/logger.hpp"

using namespace tt::constants;
namespace ttnn::operations::data_movement {


ConcatOpParallelizationStrategy ConcatDeviceOperation::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors[0].is_sharded()) {
        return ConcatOpParallelizationStrategy::SHARDED_MULTI_CORE;
    } else {
        return ConcatOpParallelizationStrategy::MULTI_CORE;
    }
}

void ConcatDeviceOperation::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &first_input = input_tensors[0];
    tt::tt_metal::LegacyShape shape_first = first_input.get_legacy_shape();
    TT_FATAL(this->dim < shape_first.rank(), "ConcatDeviceOperation dim specified is larger than input tensor rank.");
    shape_first[this->dim] = 0;
    bool shard_first = input_tensors[0].is_sharded();
    bool warn_about_alignment = false;

    for (int i = 0; i < input_tensors.size(); i++) {
        const Tensor &in_ref = input_tensors[i];
        TT_FATAL(in_ref.buffer(), "Operand to concat needs to be allocated in a buffer on device.");
        TT_FATAL(in_ref.device(), "Operand to concat needs to be on device.");
        TT_FATAL(in_ref.device() == first_input.device(), "Operands to concat need to be on the same device.");
        TT_FATAL(in_ref.get_layout() == first_input.get_layout(), "All Tensors should have same layouts.");
        TT_FATAL(in_ref.get_dtype() == first_input.get_dtype(), "All Tensors should have same dtypes.");
        tt::tt_metal::LegacyShape curr_shape = in_ref.get_legacy_shape();
        TT_FATAL(curr_shape.rank() == shape_first.rank(), "Input tensor ranks must be equal");
        curr_shape[this->dim] = 0;
            // last tensor can support without any kernel changes
        if(in_ref.get_layout() == Layout::TILE and !in_ref.get_shape().has_tile_padding(this->dim)) {
            warn_about_alignment = true;
        }
        TT_FATAL(curr_shape == shape_first, "concat tensors differ in shape across non-concat dimensions.");
        TT_FATAL(in_ref.is_sharded() == shard_first, "All tensors must be sharded or all must be interleaved");
        if (shard_first) {
            // TODO(jerrysky3): Remove this when we replace the two tensors concat kernel with the general one.
            TT_FATAL(
                input_tensors.size() > 2 || in_ref.get_layout() == Layout::ROW_MAJOR,
                "Only row major supported for sharded two tensors concat.");
            TT_FATAL(in_ref.shard_spec().has_value(), "Sharded tensors must have a shard spec.");
            TT_FATAL(
                in_ref.shard_spec().value().grid == first_input.shard_spec().value().grid,
                "Sharded tensors must have the same grid.");
            TT_FATAL(
                in_ref.memory_config().memory_layout == first_input.memory_config().memory_layout,
                "Sharded tensors must have the same memory layout.");
            // TODO(jerrysky3): Remove this when we replace the two tensors concat kernel with the general one.
            TT_FATAL(
                input_tensors.size() > 2 || in_ref.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED,
                "Width sharded inputs are not supported for two tensors concat yet");
            TT_FATAL(
                in_ref.memory_config().memory_layout != TensorMemoryLayout::BLOCK_SHARDED,
                "Block sharded inputs are not supported");
        }
    }
    if (warn_about_alignment) {
        tt::log_warning("ttnn.concat: Tile padding along concatenated dim ({}) is not "
                "directly supported. ttnn.concat will proceed by converting to "
                "row-major then retilizing. This may have adverse performance impacts.",
                this->dim);
    }
    if (shard_first) {
        const auto memory_layout = first_input.memory_config().memory_layout;
        TT_FATAL(
            this->output_mem_config.memory_layout == memory_layout,
            "Sharded output and inputs must have the same memory layout.");
        TT_FATAL(this->output_mem_config.is_sharded(), "Output must be sharded if input is sharded.");
        TT_FATAL(
            this->output_mem_config.shard_spec.value().grid == first_input.shard_spec().value().grid,
            "Sharded output and inputs must have the same grid.");
        if (this->dim == shape_first.rank() - 1) {
            TT_FATAL(
                memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
                "Only support width concat on height-sharded tensors.");
        } else if (this->dim == shape_first.rank() - 2) {
            TT_FATAL(
                memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
                "Only support height concat on width-sharded tensors.");
        } else {
            TT_FATAL(false, "Only width or height concat on sharded tensors");
        }
    }
}

std::vector<ttnn::SimpleShape> ConcatDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    ttnn::SimpleShape shape_out = input_tensors[0].get_logical_shape();
    shape_out[this->dim] = 0;
    for (const Tensor &in_ref : input_tensors) {
        ttnn::SimpleShape curr_shape = in_ref.get_logical_shape();
        shape_out[this->dim] += curr_shape[this->dim];
    }
    return {shape_out};
}

std::vector<Tensor> ConcatDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const Tensor &ref_in_tensor = input_tensors.at(0);

    if (this->output_mem_config.is_sharded()) {
        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            ref_in_tensor.get_dtype(),
            ref_in_tensor.get_layout(),
            ref_in_tensor.device(),
            this->output_mem_config)};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, ref_in_tensor.get_dtype(), ref_in_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks ConcatDeviceOperation::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    switch (this->get_parallelization_strategy(input_tensors)) {
        case ConcatOpParallelizationStrategy::SHARDED_MULTI_CORE:
            return detail::sharded_concat_multi_core(input_tensors, this->dim, output_tensors[0]);
        case ConcatOpParallelizationStrategy::MULTI_CORE:
        default:
            return detail::concat_multi_core(input_tensors, this->dim, output_tensors[0]);
    };
}

Tensor concat_impl(std::vector<Tensor> &input_tensors, const std::int64_t dim, const MemoryConfig &output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensors[0]}))};
    operation::launch_op(
        [dim, output_mem_config](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) -> std::vector<Tensor> {
            TT_FATAL(input_tensors.size() > 0, "need 1 or more tensors");
            if (input_tensors.size() == 1) {
                return {ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_mem_config(input_tensors[0], output_mem_config)};
            }
            uint32_t ref_rank = input_tensors[0].get_legacy_shape().rank();
            uint32_t normalized_dim = input_tensors[0].get_legacy_shape().get_normalized_index(dim);

            if (input_tensors[0].is_sharded()) {
                return operation::run(ConcatDeviceOperation{normalized_dim, output_mem_config}, {input_tensors});
            } else {
                if (input_tensors[0].get_layout() == Layout::ROW_MAJOR && normalized_dim == ref_rank - 1) {
                    for (const auto &input_tensor : input_tensors) {
                        TT_FATAL(
                            (input_tensor.get_padded_shape()[dim] * input_tensor.element_size()) % input_tensor.buffer()->alignment() ==
                                0,
                            "Current concat implementation requires aligned last dim when concatting on last dim");
                    }
                }
                // row major should default to row major and tilized to tilized implementations, but the below loop turned RM to tilized when possible
                Layout target_layout = Layout::TILE;
                // this should be dead code when instantiating layout to match the input
                for (const auto &input_tensor : input_tensors) {
                    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
                        const auto &input_shape = input_tensor.get_legacy_shape();
                        if (input_shape.rank() < 2 || input_shape[-2] % TILE_HEIGHT != 0 ||
                            input_shape[-1] % TILE_WIDTH != 0) {
                            target_layout = Layout::ROW_MAJOR;
                            break;
                        }
                    }
                }
                std::vector<ttnn::operations::experimental::auto_format::FormatParams> input_format_params;
                input_format_params.reserve(input_tensors.size());
                for (const auto &input_tensor : input_tensors) {
                    if (target_layout == Layout::ROW_MAJOR) {
                        input_format_params.push_back(ttnn::operations::experimental::auto_format::FormatParams{
                            .pad_shape = input_tensor.get_legacy_shape(),
                            .pad_value = 0.0,
                            .target_layout = target_layout});
                    } else {
                        tt::tt_metal::LegacyShape pad_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
                        input_format_params.push_back(
                            ttnn::operations::experimental::auto_format::FormatParams{.pad_shape = pad_shape, .pad_value = 0.0, .target_layout = target_layout});
                    }
                }

                return operation::run_with_autoformat(
                    ConcatDeviceOperation{normalized_dim, output_mem_config}, {input_tensors}, {input_format_params}, {target_layout});
            }
        },
        input_tensors,
        output_tensors);
    return output_tensors.at(0);
}

} // namespace ttnn::operations::data_movement
