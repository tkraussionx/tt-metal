// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_pools.hpp"

#include "impl/buffers/buffer_constants.hpp"
#include "tt_metal/common/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn {
namespace operations::pool {

namespace {

float get_pool_pad_value(Pool2DType pool_type) {
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: return -std::numeric_limits<float>::infinity();
        case Pool2DType::AVG_POOL2D: return 0.;
    }
}

}  // namespace

template <Pool2DType pool_type>
Tensor Pool2DOp<pool_type>::invoke(uint8_t queue_id,
                                    const Tensor& input_tensor,
                                    uint32_t batch_size,
                                    uint32_t input_h, uint32_t input_w,
                                    uint32_t channels,
                                    std::array<uint32_t, 2> kernel_size,
                                    std::array<uint32_t, 2> stride,
                                    std::array<uint32_t, 2> padding,
                                    std::array<uint32_t, 2> dilation,
                                    const std::optional<const MemoryConfig> memory_config,
                                    const std::optional<const TensorMemoryLayout> applied_shard_scheme) {
    sliding_window::SlidingWindowConfig sliding_window_config{
            .batch_size = batch_size,
            .input_hw = {input_h, input_w},
            .window_hw = {kernel_size.at(0), kernel_size.at(1)},
            .stride_hw = {stride.at(0), stride.at(1)},
            .pad_hw = {padding.at(0), padding.at(1)},
            .dilation_hw = {dilation.at(0), dilation.at(1)}
    };
    auto output_shape = sliding_window_config.get_output_shape();   // last dim/width is 0
    auto input_tensor_sharded = input_tensor;

    // maxpool output is row major
    bool is_out_tiled = false;
    bool is_in_tiled = input_tensor.dtype() == DataType::BFLOAT8_B; // input tiled for bfp8_b

    sliding_window::ParallelConfig parallel_config;
    MemoryConfig out_memory_config = input_tensor_sharded.memory_config();
    uint32_t num_cores_nhw = 0;
    uint32_t num_cores_c = 0;

    TensorMemoryLayout shard_layout = TensorMemoryLayout::HEIGHT_SHARDED; // default to height sharding
    if (!out_memory_config.shard_spec.has_value()) {
        // Input is not sharded. Perform sharding.
        if (applied_shard_scheme.has_value()) {
            TT_FATAL((applied_shard_scheme.value() == TensorMemoryLayout::HEIGHT_SHARDED) ||
                     (applied_shard_scheme.value() == TensorMemoryLayout::WIDTH_SHARDED) ||
                     (applied_shard_scheme.value() == TensorMemoryLayout::BLOCK_SHARDED),
                     "Only height, width, or block sharding strategies are supported.");
            shard_layout = applied_shard_scheme.value();
        }
        parallel_config = conv::conv2d::determine_parallel_config(
                                            shard_layout,
                                            batch_size,
                                            channels,
                                            output_shape[1],
                                            output_shape[2],
                                            channels,
                                            input_tensor.device(),
                                            ShardOrientation::ROW_MAJOR,
                                            false);
        num_cores_nhw = conv::conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);
        num_cores_c = conv::conv2d::get_num_cores_channels_from_parallel_config(parallel_config);
        auto sharded_mem_config = conv::conv2d::create_sharded_memory_config_from_parallel_config(input_tensor_sharded.shape(), parallel_config, is_in_tiled ? tt::constants::TILE_HEIGHT : 1);
        input_tensor_sharded = ttnn::to_memory_config(input_tensor_sharded, sharded_mem_config, std::nullopt); // this converts interleaved to sharded
        out_memory_config = input_tensor_sharded.memory_config();
    } else {
        // input is already sharded, use it as is
        const auto shard_grid = out_memory_config.shard_spec.value().grid;
        const auto shard_scheme = out_memory_config.memory_layout;
        const auto shard_orientation = out_memory_config.shard_spec.value().orientation;
        TT_FATAL(!applied_shard_scheme.has_value(), "A sharding scheme should not be specified for a sharded input tensor.");
        TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");
        parallel_config.grid = shard_grid;
        parallel_config.shard_scheme = shard_scheme;
        parallel_config.shard_orientation = shard_orientation;
        num_cores_nhw = conv::conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);
        num_cores_c = conv::conv2d::get_num_cores_channels_from_parallel_config(parallel_config);
    }

    // update the shard spec to match the output shape
    auto shard_spec = out_memory_config.shard_spec.value();
    uint32_t output_shard_width_padded = input_tensor.dtype() == DataType::BFLOAT8_B ? tt::round_up(channels / num_cores_c, tt::constants::TILE_WIDTH) : tt::round_up(channels / num_cores_c * tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype())), tt::constants::TILE_WIDTH);
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    uint32_t output_nhw_padded = tt::round_up(output_nhw, num_cores_nhw * (is_out_tiled ? tt::constants::TILE_HEIGHT : 1));
    uint32_t output_shard_height_padded = output_nhw_padded / num_cores_nhw;
    log_debug(tt::LogOp, "output_nhw: {}, output_nhw_padded: {}, output_shard_height_padded: {}, output_shard_width_padded: {}", output_nhw, output_nhw_padded, output_shard_height_padded, output_shard_width_padded);
    out_memory_config.shard_spec = ShardSpec{shard_spec.grid, {output_shard_height_padded, output_shard_width_padded}, ShardOrientation::ROW_MAJOR, false};

    sliding_window_config = sliding_window::SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_h, input_w},
            .window_hw = {kernel_size.at(0), kernel_size.at(1)},
            .stride_hw = {stride.at(0), stride.at(1)},
            .pad_hw = {padding.at(0), padding.at(1)},
            .dilation_hw = {dilation.at(0), dilation.at(1)},
            .num_cores_nhw = num_cores_nhw,
            .num_cores_c = num_cores_c,
            .core_range_set = parallel_config.grid,
            .snap_to_tile = false
    };

    // call the halo uop
    const float pad_value = get_pool_pad_value(pool_type);
    auto haloed_tensor = ttnn::halo(
        queue_id,
        input_tensor_sharded,
        sliding_window_config,
        *reinterpret_cast<const uint32_t*>(&pad_value) >> 16,
        false,
        parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
        0,
        input_tensor_sharded.memory_config(),
        is_out_tiled);

    auto output_tensor = ttnn::prim::pool2d(
        queue_id,
        haloed_tensor,
        sliding_window_config,
        pool_type,
        DataType::BFLOAT16,      // input_tensor.dtype(), // currently only bfp16 output is supported
        out_memory_config);

    if (memory_config.has_value() && memory_config.value() != out_memory_config) {
        output_tensor = ttnn::to_memory_config(output_tensor, memory_config.value(), std::nullopt);
    }

    return output_tensor;
}

template class Pool2DOp<Pool2DType::MAX_POOL2D>;
template class Pool2DOp<Pool2DType::AVG_POOL2D>;

}  // namespace operations::pool
}  // namespace ttnn
