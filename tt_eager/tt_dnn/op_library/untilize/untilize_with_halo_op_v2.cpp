// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/sharding_utilities.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tensor/owned_buffer_functions.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

using range_t = std::array<int32_t, 2>;
const int32_t NEIGHBORHOOD_DIST = 2;    // => ncores to left and ncores to right

namespace untilize_with_halo_helpers {

range_t calculate_in_range(const range_t& out_range, const PoolConfig& pc) {
    // given out stick range, calculate corresponding window's center stick input coords
    range_t in_range;
    // start of the range
    {
        uint32_t out_w_i = out_range[0] % pc.out_w;
        uint32_t out_h_i = out_range[0] / pc.out_w;
        uint32_t in_w_i = out_w_i * pc.stride_w;
        uint32_t in_h_i = out_h_i * pc.stride_h;
        in_range[0] = in_h_i * pc.in_w + in_w_i;
    }
    // end of the range
    {
        uint32_t out_w_i = out_range[1] % pc.out_w;
        uint32_t out_h_i = out_range[1] / pc.out_w;
        // corresponding window's center stick input coords:
        uint32_t in_w_i = out_w_i * pc.stride_w;
        uint32_t in_h_i = out_h_i * pc.stride_h;
        in_range[1] = in_h_i * pc.in_w + in_w_i;
    }
    return in_range;
}

std::map<CoreCoord, CoreCoord> left_neighbor_core, right_neighbor_core;
void init_neighbor_core_xy_mapping(CoreCoord grid_size, bool is_twod = false) {
    TT_ASSERT(grid_size.x == 12 && grid_size.y == 9);   // grayskull
    if (is_twod) {
        // 2d decomposition case (block sharded)
        // left-right neighbors are calculated along the x dim
        // first the left neighbors (x = 0 has no left neighbor)
        for (int32_t x = 1; x < grid_size.x; ++ x) {
            int32_t left_x = x - 1;
            for (int32_t y = 0; y < grid_size.y; ++ y) {
                CoreCoord core = {(uint32_t) x, (uint32_t) y};
                left_neighbor_core[core] = {(uint32_t) left_x, (uint32_t) y};
            }
        }
        // then the neighbors (x = grid_size.x - 1 has no left neighbor)
        for (int32_t x = 0; x < grid_size.x - 1; ++ x) {
            int32_t right_x = x + 1;
            for (int32_t y = 0; y < grid_size.y; ++ y) {
                CoreCoord core = {(uint32_t) x, (uint32_t) y};
                right_neighbor_core[core] = {(uint32_t) right_x, (uint32_t) y};
            }
        }
    } else {
        // default 1d distribution case (height sharded)
        for (int32_t y = 0; y < grid_size.y; ++ y) {
            for (int32_t x = 0; x < grid_size.x; ++ x) {
                CoreCoord core = {(uint32_t) x, (uint32_t) y};
                // calculate left neighbor
                int32_t left_x = x - 1, left_y = y;
                if (left_x < 0) {
                    left_x = grid_size.x - 1;
                    left_y -= 1;
                }
                if (left_y < 0) {
                    // there is no left neighbor
                } else {
                    left_neighbor_core[core] = {(uint32_t) left_x, (uint32_t) left_y};
                }
                // calculate right neighbor
                int32_t right_x = x + 1, right_y = y;
                if (right_x == grid_size.x) {
                    right_x = 0;
                    right_y += 1;
                }
                if (right_y == grid_size.y) {
                    // there is no right neighbor
                } else {
                    right_neighbor_core[core] = {(uint32_t) right_x, (uint32_t) right_y};
                }
            }
        }
    }
}

} // namespace untilize_with_halo_helpers

void UntilizeWithHaloV2::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& local_pad_start_and_size = input_tensors.at(1);
    const auto& ll_data_start_and_size = input_tensors.at(2);
    const auto& l_data_start_and_size = input_tensors.at(3);
    const auto& local_data_start_and_size = input_tensors.at(4);
    const auto& r_data_start_and_size = input_tensors.at(5);
    const auto& rr_data_start_and_size = input_tensors.at(6);

    // validate input data tensor
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor should be TILE for untilize");
    TT_FATAL(input_tensor.volume() % TILE_HW == 0);
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    // validate all other config tensors
    for (auto tensor : {local_pad_start_and_size,
                        ll_data_start_and_size,
                        l_data_start_and_size,
                        local_data_start_and_size,
                        r_data_start_and_size,
                        rr_data_start_and_size}) {
        TT_FATAL(tensor.buffer() != nullptr, "Input tensors need to be allocated buffers on device");
        TT_FATAL(tensor.memory_config().is_sharded());
        TT_FATAL(tensor.layout() == Layout::ROW_MAJOR);
        TT_FATAL(tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
    }
}

std::vector<Shape> UntilizeWithHaloV2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.shape();
    Shape output_shape = input_shape;
    // pad_h, pad_w
    // calculate the sizes (num sticks) for each of the 7 sections (5 local, 2 halo)
    // output num sticks:
    // local sections:
    // 1. (partial first row width + pad_w)
    // 2. (out_w + pad_w * 2) * (num full rows partial top image)
    // 3. (out_w + pad_w * 2) * (pad_h + out_h) * num full images
    // 4. (out_w + pad_w * 2) * (pad_h + num full rows partial bottom image)
    // 5. (partial last row width + pad_w)
    // halo sections on local core:
    // A. left halo: out_w + pad_w + 1
    // B. right halo: out_w + pad_w + 1
    // corresponding halo sections on neighbors
    // Aa. left left halo:
    // Ab. left halo:
    // Ba. right halo:
    // Bb. right right halo:

    uint32_t in_nhw = this->in_b * this->in_h * this->in_w;
    uint32_t nbatch = input_shape[0];

    // get ncores from shard shape and input shape
    auto shard_shape = input.shard_spec().value().shard_shape;
    uint32_t ncores = in_nhw / shard_shape[0];
    if (input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(input.shard_spec().value().shard_grid.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
    }

    uint32_t total_nsticks = ncores_ * max_out_nsticks_per_core_;

    // output_shape[0] remains same
    // output_shape[1] remains same
    // output_shape[2] changes
    // output_shape[3] remains same
    if (stride_ == 1) {
        output_shape[2] = total_nsticks;
    } else {
        total_nsticks = ncores * (max_out_nsticks_per_core_ + 2);   // TODO [AS]: Need to debug why using exact number (without + 2) makes it fail.
        output_shape[2] = (uint32_t) ceil((float) total_nsticks / output_shape[0]);
    }

    if (1) {
        log_debug(LogOp, "output_shape: {} {} {} {}", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
        log_debug(LogOp, "max_out_nsticks_per_core: {}", max_out_nsticks_per_core_);
        log_debug(LogOp, "derived ncores: {}", ncores);
    }

    return {output_shape};
}

std::vector<Tensor> UntilizeWithHaloV2::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();
    auto shard_spec = input_tensor.shard_spec().value();
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    uint32_t ncores = input_tensor.shape()[0] * input_tensor.shape()[2] / shard_spec.shard_shape[0];
    if (input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto core_range = *(input_tensor.shard_spec().value().shard_grid.ranges().begin());
        ncores = core_range.end.x - core_range.start.x + 1;
    }
    shard_spec.shard_shape[0] = output_shape[0] * div_up(output_shape[2], ncores);
    shard_spec.halo = true;
    // log_debug(LogOp, "derived ncores: {}", ncores);
    return {create_sharded_device_tensor(output_shape, output_dtype, Layout::ROW_MAJOR, input_tensor.device(), this->output_mem_config, shard_spec)};
}

operation::ProgramWithCallbacks UntilizeWithHaloV2::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    switch (stride_) {
        case 1:
            log_debug(LogOp, "Using stride 1 kernel");
            return { untilize_with_halo_multi_core_s1(input_tensor_a, output_tensor, pad_val_, this->in_b, this->in_h, this->in_w, this->max_out_nsticks_per_core_) };
        case 2:
            log_debug(LogOp, "Using stride 2 kernel");
            return { untilize_with_halo_multi_core_s2(input_tensor_a, output_tensor, pad_val_, in_b, in_h, in_w, max_out_nsticks_per_core_, pc_) };
        default:
            TT_ASSERT(false, "Unsupported stride value");
    };
    return {};
}

Tensor untilize_with_halo_v2(const Tensor& input_tensor,
                             const Tensor& local_pad_start_and_size,
                             const Tensor& ll_data_start_and_size,
                             const Tensor& l_data_start_and_size,
                             const Tensor& local_data_start_and_size,
                             const Tensor& r_data_start_and_size,
                             const Tensor& rr_data_start_and_size,
                             const std::vector<uint32_t>& local_pad_nsegments_per_core,
                             const std::vector<uint32_t>& ll_data_nsegments_per_core,
                             const std::vector<uint32_t>& l_data_nsegments_per_core,
                             const std::vector<uint32_t>& local_data_nsegments_per_core,
                             const std::vector<uint32_t>& r_data_nsegments_per_core,
                             const std::vector<uint32_t>& rr_data_nsegments_per_core,
                             const std::vector<std::tuple<uint32_t, uint32_t>>& resharded_start_and_end,
                             const uint32_t pad_val,
                             const MemoryConfig& mem_config) {
    TT_ASSERT(input_tensor.memory_config().is_sharded());
    TT_ASSERT(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);

    auto input_shape = input_tensor.shape();
    auto input_shard_shape = input_tensor.shard_spec().value().shard_shape;
    uint32_t ncores_height = local_data_nsegments_per_core.size();
    // NOTE: for HEIGHT_SHARDED, ncores_height == ncores
    //       for BLOCK_SHARDED, ncores_height is just the ncores along height dim (last tensor dim is split along width)

    // Calculate the max output nsticks across all coresfrom the resharded global indices
    uint32_t max_out_nsticks_per_core = 0;
    for (auto [shard_start, shard_end] : resharded_start_and_end) { // NOTE: start and end are inclusive
        uint32_t shard_nsticks = shard_end - shard_start + 1;
        max_out_nsticks_per_core = std::max(max_out_nsticks_per_core, shard_nsticks);
    }
    log_debug("max out nsticks across all shards = {}", max_out_nsticks_per_core);

    return operation::run_without_autoformat(UntilizeWithHaloV2{
                                                pad_val,
                                                ncores_height,
                                                max_out_nsticks_per_core,
                                                local_pad_nsegments_per_core,
                                                ll_data_nsegments_per_core,
                                                l_data_nsegments_per_core,
                                                local_data_nsegments_per_core,
                                                r_data_nsegments_per_core,
                                                rr_data_nsegments_per_core,
                                                resharded_start_and_end,
                                                mem_config},
                                             {input_tensor,
                                              local_pad_start_and_size,
                                              ll_data_start_and_size,
                                              l_data_start_and_size,
                                              local_data_start_and_size,
                                              r_data_start_and_size,
                                              rr_data_start_and_size})
                                            .at(0);

}

}  // namespace tt_metal

}  // namespace tt
