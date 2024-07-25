// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "max_pool2d_device_operation.hpp"
#include "ttnn/operations/pool/maxpool/max_pool.hpp"

// #include <algorithm>
// #include <cmath>

// #include "detail/util.hpp"
// #include "tensor/host_buffer/functions.hpp"
// #include "tensor/tensor_utils.hpp"
// #include "ttnn/experimental/tt_dnn/op_library/reduce/reduce_op.hpp"  // for reduce_op_utils
// #include "ttnn/experimental/tt_dnn/op_library/work_split.hpp"
// #include "tt_metal/host_api.hpp"

/**
 * New maxpool2d implementation that uses the new sliding window infrastructure.
 */

namespace ttnn::operations::pool {

void MaxPoolNew::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.buffer() != nullptr , "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input.get_dtype() == DataType::BFLOAT16, "Only BFLOAT16 supported for now");
    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR supported for now");

    // NOTE: This is not a hard requirement. If need to support non-power-of-2, simply change the address generator in reader to generic one.
    uint32_t in_nbytes_c = (input.get_legacy_shape()[3]) * (input.get_dtype() == DataType::BFLOAT16 ? 2 : 1);
    bool is_pow2 = (in_nbytes_c & (in_nbytes_c - 1)) == 0;
    TT_FATAL(is_pow2, "Row size (nchannels * bytes = {}) should be power of 2 ({}).", in_nbytes_c, is_pow2);

    TT_FATAL(input.memory_config().is_sharded(), "Input needs to be sharded");
    TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");

    TT_FATAL(this->out_mem_config_.is_sharded(), "Output memory config needs to be sharded");
    TT_FATAL(this->out_mem_config_.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
}

std::vector<tt::tt_metal::Shape> MaxPoolNew::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    // NOTE: Only for RM
    // NOTE2: Assuming { N, 1, H * W, C }
    // NOTE3: Assuming output data type is same as input
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.get_legacy_shape();

    // confirm that the output size supplied to the function matches
    uint32_t out_h = sliding_window_config_.get_output_shape()[1];
    uint32_t out_w = sliding_window_config_.get_output_shape()[2];

    // need to pad the last dim to TILE_WIDTH
    uint32_t out_c = input_shape[3];
    uint32_t out_c_padded = ceil_multiple_of(out_c, (out_c <= 16) ? 16 : tt::constants::TILE_WIDTH);
    uint32_t out_pagesize = out_c_padded * datum_size(datatype_to_dataformat_converter(input.get_dtype()));
    uint32_t out_nhw = sliding_window_config_.batch_size_ * out_h * out_w;
    uint32_t out_nhw_padded =
        this->out_mem_config_.shard_spec->shape[0] * this->out_mem_config_.shard_spec->num_cores();

    // {1, 1, N * H * W, C}
    const auto out_dims = std::vector<uint32_t>({1, 1, out_nhw_padded, out_c_padded});
    const auto padding = Padding(
        {{0, 0}, {0, 0}, {0, out_nhw_padded - out_nhw}, {0, out_c_padded - out_c}},
        Padding::PadValue::NegativeInfinity);
    auto out_shape = tt::tt_metal::Shape{out_dims, padding};
    return {out_shape};
}

std::vector<Tensor> MaxPoolNew::create_output_tensors(const std::vector<Tensor> &inputs) const {
    const auto& input = inputs.at(0);
    tt::tt_metal::Shape output_shape = compute_output_shapes(inputs).at(0);
    auto mem_config = this->out_mem_config_;
    if (mem_config.shard_spec.has_value()) {
        mem_config.shard_spec->shape[1] = output_shape[3];
    } else {
        uint32_t ncores = input.shard_spec().value().num_cores();
        TT_FATAL(ncores == sliding_window_config_.num_cores_nhw_, "Number of cores should match");
        uint32_t nbatch = output_shape[0];
        uint32_t out_nhw = output_shape[0] * output_shape[1] * output_shape[2];
        uint32_t out_nhw_per_core = out_nhw / ncores;
        CoreRangeSet shard_grid = sliding_window_config_.core_range_set_;
        std::array<uint32_t, 2> shard_shape = {out_nhw_per_core, input.get_legacy_shape()[-1]};
        mem_config.shard_spec = ShardSpec{shard_grid, shard_shape, ShardOrientation::ROW_MAJOR, false};
    }

    return {create_device_tensor(output_shape, input.get_dtype(), input.get_layout(), input.device(), mem_config)};
}

operation::ProgramWithCallbacks MaxPoolNew::create_program(const std::vector<Tensor>& inputs, std::vector<Tensor> &outputs) const {
    const auto& input = inputs.at(0);
    auto& output = outputs.at(0);
        const auto& reader_indices = inputs.at(1);
        TT_FATAL(input.memory_config().is_sharded(), "Input needs to be sharded for UTWHv2");
        return {max_pool_2d_multi_core_sharded_with_halo_v2_new(
                    input,
                    output,
                    sliding_window_config_,
                    out_mem_config_)};
}

operation::OpPerformanceModel MaxPoolNew::create_op_performance_model(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors, const std::vector<Tensor> &output_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_shape();
    uint32_t batch_size = sliding_window_config_.batch_size_;
    uint32_t activation_h = sliding_window_config_.input_hw_.first;
    uint32_t activation_w = sliding_window_config_.input_hw_.second;
    uint32_t activation_c = input_shape[3];
    uint32_t output_channels = input_shape[3];

    uint32_t filter_h = sliding_window_config_.window_hw_.first;
    uint32_t filter_w = sliding_window_config_.window_hw_.second;
    uint32_t stride_h = sliding_window_config_.stride_hw_.first;
    uint32_t stride_w = sliding_window_config_.stride_hw_.second;
    uint32_t pad_h = sliding_window_config_.pad_hw_.first;
    uint32_t pad_w = sliding_window_config_.pad_hw_.second;

    // GS specific parameters
    int num_cores = 9 * 12;
    int tensix_mul_adds_per_cycle_lofi = 2048;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, maxpool, downsample)
    int output_height = std::floor((activation_h - filter_h + 2 * pad_h) / stride_h + 1);
    int output_width = std::floor((activation_w - filter_w + 2 * pad_w) / stride_w + 1);

    // Calculate number of mul/add / compare operations
    int64_t num_mul_adds_per_elem = activation_c * filter_h * filter_w; // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi));

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

// Tensor max_pool2d_internal(const Tensor &input,
//                            const SlidingWindowConfig& sliding_window_config,
//                            const MemoryConfig& out_mem_config) {
//     std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input}))};
//     operation::launch_op(
//         [sliding_window_config, out_mem_config]
//             (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
//                 return operation::run_without_autoformat(MaxPoolNew{sliding_window_config, out_mem_config},
//                                                          input_tensors);
//             }, {input}, output_tensors);
//     return output_tensors.at(0);
// }

} // namespace ttnn::operations::pool
