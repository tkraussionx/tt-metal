// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/downsample/downsample_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


void DownsampleV2::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to downsample need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to downsample need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Can only downsample tile major data");

    TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
    TT_FATAL(input_tensor_a.memory_config().is_sharded());
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
}

std::pair<uint32_t, uint32_t> get_num_cores_height_width_sliced(CoreRangeSet all_cores, TensorMemoryLayout memory_layout, ShardOrientation shard_orientation) {
    TT_FATAL(memory_layout ==  TensorMemoryLayout::HEIGHT_SHARDED || memory_layout ==  TensorMemoryLayout::BLOCK_SHARDED);
    if (memory_layout ==  TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR);
    } else {
        TT_FATAL(shard_orientation == ShardOrientation::COL_MAJOR);
        TT_FATAL(all_cores.ranges().size() == 1);
    }
    uint32_t num_cores = all_cores.num_cores();
    auto first_core_range = *all_cores.ranges().begin();
    uint32_t num_cores_height_sliced = memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ? num_cores : first_core_range.end.x+1;
    uint32_t num_cores_width_sliced = memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ? 1 : first_core_range.end.y+1; // width is not sliced when height sharded
    return {num_cores_height_sliced, num_cores_width_sliced};
}

std::vector<Shape> DownsampleV2::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.shape()[0] == 1 && input_tensor_a.shape()[1] == 1);
    uint32_t input_height = input_tensor_a.shape()[2];
    auto [img_batch_size, img_height, img_width, img_stride_h, img_stride_w] = this->downsample_params_;
    TT_FATAL(input_height >= img_batch_size * img_height * img_width);
    uint32_t output_height_unpadded = img_batch_size * ceil( (double) img_height / (double) img_stride_h) * ceil( (double) img_width / (double) img_stride_w);
    uint32_t output_height = round_up(output_height_unpadded, TILE_HEIGHT);
    uint32_t output_width = input_tensor_a.shape()[3];
    auto output_padding = Padding({{0, 0}, {0, 0}, {0, (output_height - output_height_unpadded)}, {0, 0}}, Padding::PadValue::Any);
    auto output_tensor_shape = Shape({1, 1, output_height, output_width}, output_padding);
    return {output_tensor_shape};
}

std::vector<Tensor> DownsampleV2::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    auto [num_cores_height_sliced, num_cores_width_sliced] = get_num_cores_height_width_sliced(input_tensor.shard_spec().value().shard_grid,
                                            input_tensor.memory_config().memory_layout,
                                            input_tensor.shard_spec().value().shard_orientation);
    TT_FATAL(num_cores_height_sliced == this->ncores_nhw_);
    uint32_t output_shard_height = round_up(output_shape[2], num_cores_height_sliced * TILE_HEIGHT) / num_cores_height_sliced;
    uint32_t output_shard_width = round_up(output_shape[3], num_cores_width_sliced * TILE_WIDTH) / num_cores_width_sliced;
    return {create_sharded_device_tensor(output_shape, this->output_dtype, Layout::TILE, input_tensor.device(), input_tensor.memory_config(), ShardSpec{input_tensor.shard_spec().value().shard_grid, std::array<uint32_t, 2>{{output_shard_height, output_shard_width}}, input_tensor.shard_spec().value().shard_orientation})};
}

operation::ProgramWithCallbacks DownsampleV2::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& a = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);
    tt::DataFormat untilized_cb_data_format = DataFormat::Float16_b;
    uint32_t untilized_single_tile_size = tt_metal::detail::TileSize(untilized_cb_data_format);
    auto [img_batch_size, img_height, img_width, img_stride_h, img_stride_w] = downsample_params_;
    tt_metal::Buffer *src0_buffer = a.buffer();

    TT_FATAL(a.shape()[0] == 1 && a.shape()[1] == 1);
    TT_FATAL(output.shape()[0] == 1 && output.shape()[1] == 1);

    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Sanity check of output size
    TT_FATAL(output.volume() % TILE_HW == 0);
    uint32_t unpadded_input_volume = img_batch_size * img_height * img_width;
    TT_FATAL(a.volume() >= unpadded_input_volume);
    uint32_t unpadded_output_volume = ceil((double) unpadded_input_volume / (double) (img_stride_h * img_stride_w));
    TT_FATAL(output.volume() >= unpadded_output_volume);


    uint32_t ncores_x_full_grid = device->compute_with_storage_grid_size().x;
    auto [num_cores_height_sliced, num_cores_width_sliced] = get_num_cores_height_width_sliced(a.shard_spec().value().shard_grid,
                                        a.memory_config().memory_layout,
                                        a.shard_spec().value().shard_orientation);
    TT_FATAL(num_cores_height_sliced == ncores_nhw_);
    uint32_t num_cores = num_cores_height_sliced * num_cores_width_sliced;
    auto all_cores = a.shard_spec().value().shard_grid;
    auto memory_layout = a.memory_config().memory_layout;
    TT_FATAL(all_cores == output.shard_spec().value().shard_grid);
    TT_FATAL(memory_layout == output.memory_config().memory_layout);
    TT_FATAL(memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(all_cores.ranges().size() == 1);
    } else {
        TT_FATAL(num_cores_width_sliced == 1);
    }
    uint32_t num_cores_x = memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ? ncores_x_full_grid : num_cores_height_sliced;

    auto core_range = all_cores;

    uint32_t input_height = a.shape()[2]; // input height == flattened face of input image, multiple images are stacked in H dim
    uint32_t input_width = a.shape()[3]; // input width == input image # of channels
    uint32_t output_height = output.shape()[2]; // output height == flattened face of output image, multiple images are stacked in H dim
    uint32_t output_width = output.shape()[3];
    TT_FATAL(input_width == output_width);

    uint32_t input_height_unpadded = img_batch_size * img_height * img_width;
    TT_FATAL(input_height >= input_height_unpadded);
    uint32_t output_height_unpadded = img_batch_size * std::ceil((double) (img_height * img_width) / (double) (img_stride_h * img_stride_w));
    TT_FATAL(output_height >= output_height_unpadded);

    uint32_t input_shard_height = a.shard_spec().value().shard_shape[0];
    TT_FATAL(input_shard_height * num_cores_height_sliced >= input_height); // last core shard size may be padded
    uint32_t input_height_padded = input_shard_height * num_cores_height_sliced;
    uint32_t input_shard_width = a.shard_spec().value().shard_shape[1];
    TT_FATAL(input_shard_width * num_cores_width_sliced == input_width);

    uint32_t input_height_padding = input_height_padded - input_height_unpadded;
    TT_FATAL(input_height_padding < input_shard_height); // last core has padding
    uint32_t last_core_input_shard_height_unpadded = input_shard_height - input_height_padding;


    uint32_t output_shard_height = output.shard_spec().value().shard_shape[0];
    uint32_t output_shard_width = output.shard_spec().value().shard_shape[1];

    TT_FATAL(output_shard_width == input_shard_width);
    TT_FATAL(output_shard_height * num_cores_height_sliced >= output_height); // last core shard size may be padded
    uint32_t output_height_padded = output_shard_height * num_cores_height_sliced;
    uint32_t output_height_padding = output_height_padded - output_height_unpadded;
    TT_FATAL(output_height_padding < output_shard_height);
    uint32_t last_core_output_shard_height_unpadded = output_shard_height - output_height_padding;

    TT_FATAL(input_shard_width % TILE_WIDTH == 0);
    uint32_t num_input_tiles_in_row = input_shard_width / TILE_WIDTH;
    TT_FATAL(input_shard_height % TILE_HEIGHT == 0);
    uint32_t num_rows_of_input_tiles = input_shard_height / TILE_HEIGHT;

    uint32_t num_output_tiles_in_row = num_input_tiles_in_row;
    TT_FATAL(output_shard_height % TILE_HEIGHT == 0);
    uint32_t num_rows_of_output_tiles = output_shard_height / TILE_HEIGHT;

    // CB for sharded input buffer (tiled layout)
    uint32_t input_cb_index = CB::c_in0;
    uint32_t num_input_tiles = num_input_tiles_in_row * num_rows_of_input_tiles;
    tt_metal::CircularBufferConfig input_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
		.set_page_size(input_cb_index, input_single_tile_size);
    input_cb_config = input_cb_config.set_globally_allocated_address(*a.buffer());
    auto input_cb = tt_metal::CreateCircularBuffer(program, core_range, input_cb_config);
    log_debug(LogOp, "CB {}: PS = {} NP = {}", input_cb_index, input_single_tile_size, num_input_tiles);

    // CB to store untilized resharded data. This will contain untilized data from current core and neighbour cores.
    // 1 row of tiles is untilized at a time. So, set the page size to be TILE_HEIGHT * row size bytes
    // a part or all of the local sharded input would be untilized into this CB
    // # of rows of local sharded input that would be part of the untilized resharded CB may not be divisible by TILE_HEIGHT
    // For each core, we have config variables - local_data_src_start_offset and local_data_start_and_size. These define how local data is mapped from untilized sharded input to untilized resharded intermediate buffer.
    // Determine the size of the untilized resharded intermediate buffer to adequately fit the max number of rows across all cores
    // Local data is directly untilized into resharded buffer. Untilize is done on a block with 1 row of tiles. If height is not divisible by tile height, untilize produces garbage rows. Resharded buffer needs to be sized up for local data's garbage rows.
    // The data from left and right cores, contain valid rows only and no garbage rows because the data is already untilized in remote left/right cores.
    // There are 4 scenarios that can happen for each core -
    // 1. All of the local sharded input needs to be untilized into the untilized resharded CB.
    //          In this case, local data source offset would be 0 and size of local data (# of rows) to move to resharded buffer == input shard height
    //          input shard height is already divisible by TILE_HEIGHT as input is tiled layout. No special padding required here.
    // 2. The top part of the local sharded input needs to be untilized into the untilized resharded CB.
    //          Here, local data source start offset == 0 and size of local data (# of rows) to move to resharded buffer != input shard height
    //          if size of local data to move to resharded buffer is not divisible by TILE_HEIGHT, there will be garbage rows in the end (post untilize)
    //          We have to size up untilized resharded buffer to include garbage rows in the end.
    //          In this case, if this is the last core, it means tilized sharded input has padded height. For sharded tilized data across cores, only the last core has padding.
    //          We don't have to worry about the padded height overwriting other data in CB because this is the last core and there is no data at the end of the resharded CB coming in from the next core.
    //          Otherwise if this is not the last core, we are sending data to the right core. It is okay to write garbage rows after local data in the resharded buffer. There is no data at the end of the resharded CB coming in from the next core.
    // 3. The bottom part of the local sharded input needs to be untilized into the untilized resharded CB.
    //          Here, local data source start offset != 0 and local data size (# of rows) to move to resharded buffer + local data source start offset == input shard height
    //          If the local data source start offset is not divisible by TILE_HEIGHT, we have to untilize garbage rows before the start offset. The start offset needs to be shifted up to nearest 32.
    //          In this case, we know that this core is not receiving any data from the left core that would be on top of the untilized resharded buffer. So, the local data destination start offset must be 0.
    //          Then, we have to ensure that the untilized resharded buffer is sized up for this garbage rows in the beginning
    // 4. The middle part of the local sharded input needs to be untilized into the untilized resharded CB.
    //          Here, the core is sending data to both left and right cores and not receiving any data from neighbour cores.
    //          If the local data source start offset is not divisible by TILE_HEIGHT, we have to untilize garbage rows before the start offset. The start offset needs to be shifted up to nearest 32.
    //          In this case, we know that this core is not receiving any data from the left core that would be on top of the untilized resharded buffer. So, the local data destination start offset must be 0.
    //          If the local data size with the shifted up start offset is not divisible by TILE HEIGHT, there will garbage rows in the end post untilize. This is also okay because there is no data coming from right core at the end of untilized resharded buffer.
    //          Thus, for this scenario, we have to ensure that the untilized resharded buffer is sized up for the garbage rows in the beginning and in the end
    uint32_t max_untilized_resharded_buffer_size_nrows_with_garbage = 0;
    vector<int32_t> local_data_source_start_offset_to_untilize = local_data_src_start_offset_;
    vector<int32_t> local_data_size_nrows_with_garbage;
    for(uint32_t i = 0; i < ncores_nhw_; i++) {
        int32_t local_data_source_start_offset = local_data_src_start_offset_[i];
        TT_FATAL(local_data_source_start_offset >= 0);
        int32_t local_data_dest_start_offset = local_data_start_and_size_[i].first;
       local_data_size_nrows_with_garbage.push_back(local_data_start_and_size_[i].second);
        if (local_data_source_start_offset == 0) {
            if (local_data_size_nrows_with_garbage[i] != input_shard_height && local_data_size_nrows_with_garbage[i] % TILE_HEIGHT != 0) {
                    // Scenario 2 (garbage in the end)
                    if (i < ncores_nhw_ - 1) {
                        // sanity check that next core does not have any data to send to current core
                        TT_FATAL(l_data_start_and_size_[i+1].first == -1 && l_data_start_and_size_[i+1].second == -1);
                    }
                    local_data_size_nrows_with_garbage[i] = div_up(local_data_size_nrows_with_garbage[i], TILE_HEIGHT);
                    TT_FATAL(local_data_size_nrows_with_garbage[i] <= input_shard_height);
            }
        } else {
            if (local_data_source_start_offset % TILE_HEIGHT != 0) {
                // Scenario 3 (garbage in the beginning)
                local_data_source_start_offset_to_untilize[i] = (local_data_source_start_offset / TILE_HEIGHT) * TILE_HEIGHT;
                local_data_size_nrows_with_garbage[i] = local_data_size_nrows_with_garbage[i] + (local_data_source_start_offset - local_data_source_start_offset_to_untilize[i]);
                TT_FATAL(local_data_size_nrows_with_garbage[i] <= input_shard_height);
                // sanity check that this is the top portion of the resharded buffer and no data is coming in from previous core
                TT_FATAL(local_data_dest_start_offset == 0);
                if (i > 0) {
                    TT_FATAL(r_data_start_and_size_[i-1].first == -1 && r_data_start_and_size_[i-1].second == -1);
                }
            }
            if (local_data_size_nrows_with_garbage[i] % TILE_HEIGHT != 0) {
                // Scenario 4 (garbage in the end)
                local_data_size_nrows_with_garbage[i] = div_up(local_data_size_nrows_with_garbage[i], TILE_HEIGHT);
                TT_FATAL(local_data_size_nrows_with_garbage[i] <= input_shard_height);
                if (i < ncores_nhw_ - 1) {
                    // sanity check that next core does not have any data to send to current core
                    TT_FATAL(l_data_start_and_size_[i+1].first == -1 && l_data_start_and_size_[i+1].second == -1);
                }
            }
        }
        uint32_t data_from_left_core_size_nrows = 0;
        uint32_t data_from_right_core_size_nrows = 0;
        if (i > 0) {
            if (r_data_start_and_size_[i-1].first != -1) {
                TT_FATAL(r_data_start_and_size_[i-1].second != -1);
                data_from_left_core_size_nrows = r_data_start_and_size_[i-1].second;
                TT_FATAL(local_data_dest_start_offset >= data_from_left_core_size_nrows + r_data_start_and_size_[i-1].first);
            }
        }
        if (i < ncores_nhw_ - 1) {
            if (l_data_start_and_size_[i+1].first != -1) {
                TT_FATAL(l_data_start_and_size_[i+1].second != -1);
                data_from_right_core_size_nrows = l_data_start_and_size_[i+1].second;
                TT_FATAL(local_data_size_nrows_with_garbage[i] == local_data_start_and_size_[i].second);
                TT_FATAL(local_data_dest_start_offset >= l_data_start_and_size_[i+1].first + data_from_left_core_size_nrows);
            }
        }
        uint32_t untilized_resharded_buffer_size_nrows_with_garbage = data_from_left_core_size_nrows + local_data_size_nrows_with_garbage[i] + data_from_right_core_size_nrows;
        if (max_untilized_resharded_buffer_size_nrows_with_garbage < untilized_resharded_buffer_size_nrows_with_garbage) {
            max_untilized_resharded_buffer_size_nrows_with_garbage = untilized_resharded_buffer_size_nrows_with_garbage;
        }
    }
    TT_FATAL(max_untilized_resharded_buffer_size_nrows_with_garbage >= max_resharded_untilized_nsticks_per_core_);
    uint32_t untilized_resharded_cb_index = CB::c_intermed1;
    TT_FATAL(max_untilized_resharded_buffer_size_nrows_with_garbage % TILE_HEIGHT == 0);
    uint32_t untilized_datum_size = datum_size(untilized_cb_data_format);
    uint32_t input_shard_width_bytes = input_shard_width * untilized_datum_size;
    tt_metal::CircularBufferConfig untilize_resharded_cb_config = tt_metal::CircularBufferConfig(max_untilized_resharded_buffer_size_nrows_with_garbage * input_shard_width_bytes, {{untilized_resharded_cb_index, untilized_cb_data_format}})
		.set_page_size(untilized_resharded_cb_index, input_shard_width_bytes);
    auto untilized_resharded_cb = tt_metal::CreateCircularBuffer(program, core_range, untilize_resharded_cb_config);
    log_debug(LogOp, "CB {}: PS = {} NP = {}", untilized_resharded_cb_index, input_shard_width_bytes, max_untilized_resharded_buffer_size_nrows_with_garbage);

    // CB to store untilized data to be sent to left core
    uint32_t untilized_data_for_left_cb_index = CB::c_intermed2;
    uint32_t max_untilized_rows_for_left_across_cores = 0;
    TT_FATAL(l_data_start_and_size_.size() == ncores_nhw_ * 2);
    for (uint32_t i = 1; i < l_data_start_and_size_.size(); i+=2) {
        max_untilized_rows_for_left_across_cores = l_data_start_and_size_[i] > max_untilized_rows_for_left_across_cores ? l_data_start_and_size_[i] : max_untilized_rows_for_left_across_cores;
    }
    max_untilized_rows_for_left_across_cores = div_up(max_untilized_rows_for_left_across_cores, TILE_HEIGHT);
    tt_metal::CircularBufferConfig untilized_data_for_left_cb_config = tt_metal::CircularBufferConfig(max_untilized_rows_for_left_across_cores * input_shard_width_bytes, {{untilized_data_for_left_cb_index, untilized_cb_data_format}})
		.set_page_size(untilized_data_for_left_cb_index, input_shard_width_bytes);
    auto untilized_data_for_left_cb = tt_metal::CreateCircularBuffer(program, core_range, untilized_data_for_left_cb_config);
    log_debug(LogOp, "CB {}: PS = {} NP = {}", untilized_data_for_left_cb_index, input_shard_width_bytes, max_untilized_rows_for_left_across_cores);


    // CB to store untilized data to be sent to right core
    uint32_t untilized_data_for_right_cb_index = CB::c_intermed3;
    uint32_t max_untilized_rows_for_right_across_cores = 0;
    TT_FATAL(r_data_start_and_size_.size() == ncores_nhw_ * 2);
    for (uint32_t i = 1; i < r_data_start_and_size_.size(); i+=2) {
        max_untilized_rows_for_right_across_cores = r_data_start_and_size_[i] > max_untilized_rows_for_right_across_cores ? r_data_start_and_size_[i] : max_untilized_rows_for_right_across_cores;
    }
    max_untilized_rows_for_right_across_cores = div_up(max_untilized_rows_for_right_across_cores, TILE_HEIGHT);
    tt_metal::CircularBufferConfig untilized_data_for_right_cb_config = tt_metal::CircularBufferConfig(max_untilized_rows_for_right_across_cores * input_shard_width_bytes, {{untilized_data_for_right_cb_index, untilized_cb_data_format}})
		.set_page_size(untilized_data_for_right_cb_index, input_shard_width_bytes);
    auto untilized_data_for_right_cb = tt_metal::CreateCircularBuffer(program, core_range, untilized_data_for_right_cb_config);
    log_debug(LogOp, "CB {}: PS = {} NP = {}", untilized_data_for_right_cb_index, input_shard_width_bytes, max_untilized_rows_for_right_across_cores);

    // CB to store resharded downsampled CB
    uint32_t output_shard_width_bytes = output_shard_width * datum_size(untilized_cb_data_format);
    uint32_t untilize_resharded_downsampled_cb_index = CB::c_intermed3;
    tt_metal::CircularBufferConfig untilize_resharded_downsampled_cb_config = tt_metal::CircularBufferConfig(output_shard_height * output_shard_width_bytes, {{untilize_resharded_downsampled_cb_index, untilized_cb_data_format}})
		.set_page_size(untilize_resharded_downsampled_cb_index, output_shard_width_bytes);
    auto untilize_resharded_downsampled_cb = tt_metal::CreateCircularBuffer(program, core_range, untilize_resharded_downsampled_cb_config);

    // CB to store final tilized resharded downsampled CB
    uint32_t final_tilize_output_cb_index = CB::c_out0;
    uint32_t num_output_tiles = num_output_tiles_in_row * num_rows_of_output_tiles;
    uint32_t num_tiles_final_tilize_output_cb = num_output_tiles; // final output cb size == output size per core
    tt_metal::CircularBufferConfig final_tilize_output_cb_config = tt_metal::CircularBufferConfig(num_tiles_final_tilize_output_cb * output_single_tile_size, {{final_tilize_output_cb_index, output_cb_data_format}})
		.set_page_size(final_tilize_output_cb_index, output_single_tile_size);
    final_tilize_output_cb_config = final_tilize_output_cb_config.set_globally_allocated_address(*output.buffer());
    auto final_tilize_output_cb = tt_metal::CreateCircularBuffer(program, core_range, final_tilize_output_cb_config);

    uint32_t log_base_2_of_input_shard_width_bytes = (uint32_t) std::log2((float) input_shard_width_bytes);
    uint32_t stride_h_x_image_width = img_stride_h * img_width;
    std::vector<uint32_t> data_movement_kernel_1_compile_time_args = {
        (std::uint32_t) untilized_resharded_cb_index,
        (std::uint32_t) final_tilize_output_cb_index,
        (std::uint32_t) datum_size(untilized_cb_data_format),
        (std::uint32_t) input_shard_width_bytes,
        log_base_2_of_input_shard_width_bytes,
    };

    // Writer to downsample - drops rows from untilized cb
    tt_metal::KernelHandle downsample_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_writer_kernel.cpp",
        core_range,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        input_cb_index,
        halo_prev_input_cb_index,
        halo_next_input_cb_index,
        untilize_cb_index,
        untilize_downsampled_cb_index,
        final_tilize_output_cb_index,
        num_input_tiles_in_row,
        num_rows_of_output_tiles,
        num_output_tiles_in_row,
    };
    string compute_kernel = "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_compute_kernel.cpp";
    if (num_input_tiles_in_row <= MAX_PACK_UNTILIZE_WIDTH) {
        compute_kernel = "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_fast_pack_untilize_compute_kernel.cpp";
    }
    auto downsample_compute_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core_range,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    // track img h, img w, across cores
    ImgTrackingVars v;
    CoreCoord prev_core = {0,0};
    bool input_flat_h_is_of_current_core = true;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        //cout << "i=" << i << endl;

        uint32_t input_end_flat_h = input_shard_height - 1;
        uint32_t output_end_flat_h = output_shard_height - 1;

        bool halo_prev_read_enabled = false;
        DownsampleReadPatternParams halo_prev_read_pattern_params;
        uint32_t halo_prev_noc_x = 0;
        uint32_t halo_prev_noc_y = 0;
        uint32_t halo_prev_start_addr = 0;
        uint32_t halo_prev_addr_offset = 0;
        uint32_t halo_prev_num_tiles = 0;
        uint32_t halo_prev_size_bytes = 0;
        uint32_t halo_prev_input_num_rows_of_tiles = 0;
        uint32_t halo_prev_read_pattern_offset = 0;

        bool halo_next_read_enabled = false;
        DownsampleReadPatternParams halo_next_read_pattern_params;
        uint32_t halo_next_noc_x = 0;
        uint32_t halo_next_noc_y = 0;
        uint32_t halo_next_start_addr = 0;
        uint32_t halo_next_addr_offset = 0;
        uint32_t halo_next_num_tiles = 0;
        uint32_t halo_next_size_bytes = 0;
        uint32_t halo_next_input_num_rows_of_tiles = 0;
        uint32_t halo_next_read_pattern_offset = 0;
        uint32_t local_read_pattern_offset = 0;
        uint32_t current_core_input_end_flat_h = input_end_flat_h;
        uint32_t next_core_input_end_flat_h = input_end_flat_h;
        if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            if (i % num_cores_x == 0) {
                // first core in row
                // reset
                v.input_flat_h = 0;
                v.output_flat_h = 0;
            } else if (i % num_cores_x == num_cores_x - 1) {
                // set unpadded height as end index for last core in row
                current_core_input_end_flat_h = last_core_input_shard_height_unpadded - 1;
                output_end_flat_h = last_core_output_shard_height_unpadded - 1;
            } else if (i % num_cores_x == num_cores_x - 2) {
                next_core_input_end_flat_h = last_core_input_shard_height_unpadded - 1;
            }
        } else if (i == num_cores - 1) {
            // for height sharding, set unpadded height as end index for last core
            current_core_input_end_flat_h = last_core_input_shard_height_unpadded - 1;
            output_end_flat_h = last_core_output_shard_height_unpadded - 1;
        } else if (i == num_cores - 2) {
            next_core_input_end_flat_h = last_core_input_shard_height_unpadded - 1;
        }
        if (v.input_flat_h != 0 && !input_flat_h_is_of_current_core) {
            // halo region of previous core
            TT_FATAL(i != 0);
            if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(prev_core.y == core.y); // for block sharding case, prev core is left core
            }
            halo_prev_read_enabled = true;
            TT_FATAL(v.input_flat_h < input_shard_height);
            // get halo start tile address from height idx
            uint32_t halo_prev_start_tile_id_h = v.input_flat_h / TILE_HEIGHT;
            TT_FATAL(input_shard_height - v.input_flat_h <= TILE_HEIGHT * halo_prev_input_cb_max_rows_of_tiles); // halo input cb is hardcoded to store only 4 rows of tiles for now. TODO: allocate bigger CB or read in blocks
            // get halo size
            halo_prev_size_bytes = (input_shard_height - (halo_prev_start_tile_id_h * TILE_HEIGHT)) * input_shard_width / TILE_HW * input_single_tile_size;
            TT_FATAL(halo_prev_size_bytes % input_single_tile_size == 0);
            halo_prev_num_tiles = halo_prev_size_bytes / input_single_tile_size;
            TT_FATAL(halo_prev_num_tiles <= num_halo_prev_cb_input_tiles);
            TT_FATAL(halo_prev_num_tiles % num_input_tiles_in_row == 0);
            halo_prev_input_num_rows_of_tiles = halo_prev_num_tiles / num_input_tiles_in_row;
            halo_prev_addr_offset = num_input_tiles_in_row * halo_prev_start_tile_id_h * input_single_tile_size;
            halo_prev_start_addr = GetCircularBufferConfig(program, input_cb).globally_allocated_address().value();

            TT_FATAL((halo_prev_start_addr + halo_prev_addr_offset) % 32 == 0); // read address should be 32 byte aligned
            auto halo_noc_coords = device->worker_core_from_logical_core(prev_core);
            halo_prev_noc_x = halo_noc_coords.x;
            halo_prev_noc_y = halo_noc_coords.y;
            TT_FATAL(v.input_flat_h >= halo_prev_start_tile_id_h * TILE_HEIGHT);
            halo_prev_read_pattern_offset = v.input_flat_h - (halo_prev_start_tile_id_h * TILE_HEIGHT);
            local_read_pattern_offset = halo_prev_input_num_rows_of_tiles * TILE_HEIGHT;
            halo_prev_read_pattern_params = generate_downsample_read_pattern(v, img_height, img_width, img_stride_h, img_stride_w, input_end_flat_h, output_end_flat_h, true, false);
        }
        // local core
        TT_FATAL(v.output_flat_h < output_shard_height);
        uint32_t local_start_h = v.input_flat_h;
        DownsampleReadPatternParams local_read_pattern_params = generate_downsample_read_pattern(v, img_height, img_width, img_stride_h, img_stride_w, current_core_input_end_flat_h, output_end_flat_h, false, false);
        TT_FATAL(v.output_flat_h <= output_shard_height);
        uint32_t local_end_h_exclusive = v.input_flat_h == 0 ? input_shard_height : v.input_flat_h;
        uint32_t local_num_rows = local_end_h_exclusive - local_start_h;
        TT_FATAL(local_num_rows > 0);
        uint32_t local_input_num_rows_of_tiles = std::ceil( (double) local_num_rows / (double) TILE_HEIGHT);
        uint32_t local_input_offset_rows_of_tiles = local_start_h / TILE_HEIGHT;
        if (local_start_h != 0) {
            TT_FATAL(local_read_pattern_offset == 0);
            local_read_pattern_offset = local_start_h % TILE_HEIGHT;
        }
        if (v.input_flat_h != 0) {
            input_flat_h_is_of_current_core = false;
        } else {
            input_flat_h_is_of_current_core = true; // updating flag for next core
        }
        TT_FATAL(local_input_num_rows_of_tiles <= num_rows_of_input_tiles);

        if (v.output_flat_h != 0) {
            // need to read halo from next core
            TT_FATAL(i != num_cores - 1);
            TT_FATAL(v.input_flat_h == 0);
            CoreCoord next_core = {(i+1)% num_cores_x, (i+1) / num_cores_x};
            if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(next_core.y == core.y); // for block sharding case, next core is right core
            }
            halo_next_read_enabled = true;
            halo_next_read_pattern_params = generate_downsample_read_pattern(v, img_height, img_width, img_stride_h, img_stride_w, next_core_input_end_flat_h, output_end_flat_h, false, true);
            TT_FATAL(v.output_flat_h == 0);
            TT_FATAL(v.input_flat_h != 0 && v.input_flat_h < input_shard_height);
            TT_FATAL(v.input_flat_h <= TILE_HEIGHT * halo_next_input_cb_max_rows_of_tiles); // halo next input cb is hardcoded to store only 5 rows of tiles for now. TODO: allocate bigger CB or read in blocks
            uint32_t halo_next_end_tile_id_h = v.input_flat_h / TILE_HEIGHT;
            // get halo size
            halo_next_size_bytes = (halo_next_end_tile_id_h+1) * TILE_HEIGHT * input_shard_width / TILE_HW * input_single_tile_size;
            TT_FATAL(halo_next_size_bytes % input_single_tile_size == 0);
            halo_next_num_tiles = halo_next_size_bytes / input_single_tile_size;
            TT_FATAL(halo_next_num_tiles <= num_halo_next_cb_input_tiles);
            TT_FATAL(halo_next_num_tiles % num_input_tiles_in_row == 0);
            halo_next_input_num_rows_of_tiles = halo_next_num_tiles / num_input_tiles_in_row;
            halo_next_addr_offset = 0;
            halo_next_start_addr = GetCircularBufferConfig(program, input_cb).globally_allocated_address().value();
            TT_FATAL((halo_next_start_addr + halo_next_addr_offset) % 32 == 0); // read address should be 32 byte aligned
            auto halo_noc_coords = device->worker_core_from_logical_core(next_core);
            halo_next_noc_x = halo_noc_coords.x;
            halo_next_noc_y = halo_noc_coords.y;
            TT_FATAL(halo_prev_input_num_rows_of_tiles == 0);
            halo_next_read_pattern_offset = local_input_num_rows_of_tiles * TILE_HEIGHT;
        }
        TT_FATAL(v.output_flat_h == 0);

        // Compile runtime args
        vector<uint32_t> compile_rt_kernel_args = {
            local_input_num_rows_of_tiles,
            local_input_offset_rows_of_tiles,
            halo_prev_read_enabled,
            halo_prev_input_num_rows_of_tiles,
            halo_next_read_enabled,
            halo_next_input_num_rows_of_tiles,
        };

        tt_metal::SetRuntimeArgs(
            program,
            downsample_compute_kernel_id,
            core,
            compile_rt_kernel_args
        );

        // Writer runtime args
        vector<uint32_t> writer_kernel_args = {
            (uint32_t) img_height,
            (uint32_t) img_width,
            (uint32_t) img_stride_h,
            (uint32_t) img_stride_w,

            // halo prev args
            halo_prev_read_enabled,
            halo_prev_noc_x,
            halo_prev_noc_y,
            halo_prev_num_tiles,
            halo_prev_start_addr,
            halo_prev_addr_offset,
            halo_prev_size_bytes,

            // halo prev read pattern args
            halo_prev_read_pattern_offset,
            halo_prev_read_pattern_params.top_partial_middle_aligned_row_width,
            halo_prev_read_pattern_params.skip_top_partial_middle_aligned_row,
            halo_prev_read_pattern_params.top_partial_right_aligned_row_width,
            halo_prev_read_pattern_params.skip_top_partial_right_aligned_row,
            halo_prev_read_pattern_params.num_rows_top_partial_image,
            halo_prev_read_pattern_params.num_skip_rows_top_partial_image,
            halo_prev_read_pattern_params.num_full_images,
            halo_prev_read_pattern_params.num_rows_bottom_partial_image,
            halo_prev_read_pattern_params.num_skip_rows_bottom_partial_image,
            halo_prev_read_pattern_params.bottom_partial_left_aligned_row_width,
            halo_prev_read_pattern_params.skip_bottom_partial_left_aligned_row,

            // local read pattern args
            local_read_pattern_offset,
            local_read_pattern_params.top_partial_middle_aligned_row_width,
            local_read_pattern_params.skip_top_partial_middle_aligned_row,
            local_read_pattern_params.top_partial_right_aligned_row_width,
            local_read_pattern_params.skip_top_partial_right_aligned_row,
            local_read_pattern_params.num_rows_top_partial_image,
            local_read_pattern_params.num_skip_rows_top_partial_image,
            local_read_pattern_params.num_full_images,
            local_read_pattern_params.num_rows_bottom_partial_image,
            local_read_pattern_params.num_skip_rows_bottom_partial_image,
            local_read_pattern_params.bottom_partial_left_aligned_row_width,
            local_read_pattern_params.skip_bottom_partial_left_aligned_row,

            // halo next core args
            halo_next_read_enabled,
            halo_next_noc_x,
            halo_next_noc_y,
            halo_next_num_tiles,
            halo_next_start_addr,
            halo_next_addr_offset,
            halo_next_size_bytes,

            // halo next read pattern args
            halo_next_read_pattern_offset,
            halo_next_read_pattern_params.top_partial_middle_aligned_row_width,
            halo_next_read_pattern_params.skip_top_partial_middle_aligned_row,
            halo_next_read_pattern_params.top_partial_right_aligned_row_width,
            halo_next_read_pattern_params.skip_top_partial_right_aligned_row,
            halo_next_read_pattern_params.num_rows_top_partial_image,
            halo_next_read_pattern_params.num_skip_rows_top_partial_image,
            halo_next_read_pattern_params.num_full_images,
            halo_next_read_pattern_params.num_rows_bottom_partial_image,
            halo_next_read_pattern_params.num_skip_rows_bottom_partial_image,
            halo_next_read_pattern_params.bottom_partial_left_aligned_row_width,
            halo_next_read_pattern_params.skip_bottom_partial_left_aligned_row,

            halo_prev_input_num_rows_of_tiles + local_input_num_rows_of_tiles + halo_next_input_num_rows_of_tiles,
            num_input_tiles_in_row,
            num_output_tiles,

            (uint32_t) false
        };

        tt_metal::SetRuntimeArgs(
            program,
            downsample_writer_kernel_id,
            core,
            writer_kernel_args
        );
        prev_core = core;
    }

    auto override_runtime_args_callback = [
        input_cb=input_cb,
        final_tilize_output_cb=final_tilize_output_cb,
        downsample_writer_kernel_id=downsample_writer_kernel_id,
        num_cores=num_cores,
        num_cores_x=num_cores_x
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, input_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, final_tilize_output_cb, *dst_buffer);
        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i % num_cores_x, i / num_cores_x};
            auto &runtime_args = GetRuntimeArgs(program, downsample_writer_kernel_id, core);
            runtime_args[8] = src_buffer->address();
            runtime_args[39] = src_buffer->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

Tensor downsample(const Tensor &input_tensor_a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> output_dtype) {
    return operation::run_without_autoformat(Downsample{downsample_params, output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a}).at(0);
}

struct DownsampleReadPatternParams {
    uint32_t top_partial_middle_aligned_row_width;
    uint32_t skip_top_partial_middle_aligned_row;
    uint32_t top_partial_right_aligned_row_width;
    uint32_t skip_top_partial_right_aligned_row;
    uint32_t num_rows_top_partial_image;
    uint32_t num_skip_rows_top_partial_image;
    uint32_t num_full_images;
    uint32_t num_rows_bottom_partial_image;
    uint32_t num_skip_rows_bottom_partial_image;
    uint32_t bottom_partial_left_aligned_row_width;
    uint32_t skip_bottom_partial_left_aligned_row;
};

struct ImgTrackingVars {
    uint32_t img_h = 0;
    uint32_t img_w = 0;
    uint32_t next_img_h = 0; // img_h after stride
    uint32_t next_img_w = 0;
    uint32_t input_flat_h = 0; // index within sharded input
    uint32_t output_flat_h = 0; // index within sharded output
};

DownsampleReadPatternParams generate_downsample_read_pattern(ImgTrackingVars & v, uint32_t img_height, uint32_t img_width, uint32_t img_stride_h, uint32_t img_stride_w, uint32_t input_end_flat_h, uint32_t output_end_flat_h, bool current_region_is_halo_prev_core, bool current_region_is_halo_next_core) {
    // Sanity checks at the start for local data
    TT_FATAL(v.next_img_h >= v.img_h);
    TT_FATAL(v.next_img_w == v.img_w); // assumption that the start is picked and not skipped by stride
    TT_FATAL(v.img_h < img_height);
    TT_FATAL(v.next_img_w < img_width);
    if (current_region_is_halo_prev_core) {
        //cout << "GENERATING READ PATTERN FOR HALO REGION FROM PREVIOUS CORE" << endl;
        TT_FATAL(!current_region_is_halo_next_core);
        TT_FATAL(v.input_flat_h != 0);
        TT_FATAL(v.output_flat_h == 0);
    } else if (current_region_is_halo_next_core) {
        //cout << "GENERATING READ PATTERN FOR HALO REGION FROM NEXT CORE" << endl;
        TT_FATAL(!current_region_is_halo_prev_core);
        TT_FATAL(v.input_flat_h == 0);
        TT_FATAL(v.output_flat_h != 0);
    } else {
        //cout << "GENERATING READ PATTERN FOR LOCAL REGION" << endl;
    }

    //cout << "img_h=" << v.img_h << ", img_w=" << v.img_w << ", next_img_h=" << v.next_img_h << ", next_img_w=" << v.img_w << endl;
    //cout << "v.input_flat_h=" << v.input_flat_h << ", input_end_flat_h=" << input_end_flat_h << ", v.output_flat_h=" << v.output_flat_h << ", output_end_flat_h=" << output_end_flat_h << endl;

    TT_FATAL(v.input_flat_h < input_end_flat_h);
    TT_FATAL(v.output_flat_h < output_end_flat_h);

    uint32_t output_img_height = std::ceil ( (double) img_height / (double) img_stride_h);
    uint32_t output_img_width = std::ceil ( (double) img_width / (double) img_stride_w);
    bool found_halo_for_next_core = false;

    uint32_t top_partial_middle_aligned_row_width = 0;
    uint32_t skip_top_partial_middle_aligned_row = 1;
    uint32_t top_partial_right_aligned_row_width = 0;
    uint32_t skip_top_partial_right_aligned_row = 1;
    uint32_t num_rows_top_partial_image = 0;
    uint32_t num_skip_rows_top_partial_image = 0;
    uint32_t num_full_images = 0;
    uint32_t num_rows_bottom_partial_image = 0;
    uint32_t num_skip_rows_bottom_partial_image = 0;
    uint32_t bottom_partial_left_aligned_row_width = 0;
    uint32_t skip_bottom_partial_left_aligned_row = 1;
    if (v.img_w != 0) {
        // Check if its right aligned or middle aligned (special corner case for halo)
        if (v.input_flat_h + img_width - v.img_w <= input_end_flat_h+1) {
            // top partial right aligned
            top_partial_right_aligned_row_width = img_width - v.img_w;
            skip_top_partial_right_aligned_row = (v.next_img_h == v.img_h) ? 0 : 1;
            v.input_flat_h += top_partial_right_aligned_row_width;
            if (!skip_top_partial_right_aligned_row) {
                v.output_flat_h += std::ceil((double) top_partial_right_aligned_row_width / (double) img_stride_w);
                TT_FATAL(v.output_flat_h <= output_end_flat_h);
            }
            v.img_w = 0;
            v.next_img_w = 0;
            if (v.img_h == img_height - 1) {
                v.img_h = 0;
                v.next_img_h = 0;
            } else {
                v.img_h += 1;
                if (v.next_img_h < v.img_h) {
                    v.next_img_h += img_stride_h;
                }
            }
        } else {
            // special corner case for halo region
            // middle aligned
            TT_FATAL(input_end_flat_h - v.input_flat_h + 1 < img_width);
            TT_FATAL(current_region_is_halo_prev_core || current_region_is_halo_next_core);
            // top partial middle aligned
            top_partial_middle_aligned_row_width = input_end_flat_h - v.input_flat_h + 1;
            skip_top_partial_middle_aligned_row = (v.next_img_h == v.img_h) ? 0 : 1;
            v.input_flat_h += top_partial_middle_aligned_row_width;
            if (!skip_top_partial_middle_aligned_row) {
                v.output_flat_h += std::ceil((double) top_partial_middle_aligned_row_width / (double) img_stride_w);
                TT_FATAL(v.output_flat_h <= output_end_flat_h);
            }
            uint32_t img_w_start = v.img_w;
            while(v.img_w < img_w_start + top_partial_middle_aligned_row_width) {
                v.img_w += 1;
                if (v.next_img_w < v.img_w) {
                    v.next_img_w += img_stride_w;
                }
            }
            TT_FATAL(v.img_w < img_width-1);
        }
    }
    TT_FATAL(v.next_img_w == v.img_w);
    TT_FATAL(v.output_flat_h <= output_end_flat_h);
    TT_FATAL(v.next_img_h >= v.img_h);
    if (v.img_w != 0) {
        // special case for halo
        TT_FATAL(v.input_flat_h == input_end_flat_h+1);
    }
    TT_FATAL(v.img_h < img_height && v.img_w < img_width);

    uint32_t num_rows_remaining_of_current_image = (v.img_h == 0) ? 0 : img_height - v.img_h;
    if (num_rows_remaining_of_current_image > 0) {
        uint32_t num_rows_to_skip = v.next_img_h - v.img_h;
        uint32_t output_h_from_remaining_rows_of_current_image = std::ceil( (double) (num_rows_remaining_of_current_image - num_rows_to_skip) / (double) img_stride_h ) * output_img_width;
        bool output_for_partial_top_image = v.output_flat_h + output_h_from_remaining_rows_of_current_image <= output_end_flat_h+1;
        bool input_for_partial_top_image = v.input_flat_h + (num_rows_remaining_of_current_image * img_width) <= input_end_flat_h+1;
        if (output_for_partial_top_image && input_for_partial_top_image) {
            // Top partial image section
            num_rows_top_partial_image = img_height - v.img_h;
            num_skip_rows_top_partial_image = v.next_img_h - v.img_h;
            // Sanity check
            TT_FATAL((v.img_h + num_rows_top_partial_image == img_height));
            v.img_h = 0;
            v.next_img_h = 0;
            v.input_flat_h += (num_rows_top_partial_image * img_width);
            v.output_flat_h += output_h_from_remaining_rows_of_current_image;
            TT_FATAL(v.input_flat_h <= input_end_flat_h+1);
        }
    TT_FATAL(v.output_flat_h <= output_end_flat_h+1);
    }
    TT_FATAL(v.img_h < img_height && v.img_w < img_width);

    if (v.img_h == 0 && v.img_w == 0) {
        // Check for full images
        while(1) {
            bool output_for_current_full_image = v.output_flat_h + (output_img_height * output_img_width) <= output_end_flat_h+1;
            bool input_for_current_full_image = v.input_flat_h + (img_height * img_width) <= input_end_flat_h+1;
            if (!output_for_current_full_image || !input_for_current_full_image) {
                break;
            }
            v.input_flat_h += (img_height * img_width);
            v.img_h = 0;
            v.img_w = 0;
            v.next_img_h = 0;
            v.next_img_w = 0;
            num_full_images += 1;
            v.output_flat_h +=  (output_img_height * output_img_width);
        }
        TT_FATAL(v.img_h == 0 && v.img_w == 0 && v.next_img_h == 0 && v.next_img_w == 0);
    }

    // Sanity check
    TT_FATAL(v.input_flat_h <= input_end_flat_h+1);
    TT_FATAL(v.output_flat_h <= output_end_flat_h+1);

    bool found_first_unskipped_row_in_bottom_partial_imgage = false;
    // check for bottom partial image rows
    while (1) {
        bool output_for_bottom_partial_image_row = (v.next_img_h == v.img_h) ? (v.output_flat_h + output_img_width <= output_end_flat_h+1) : true; // true for skipped row
        bool input_for_bottom_partial_image_row = v.input_flat_h + img_width <= input_end_flat_h+1;
        if (!output_for_bottom_partial_image_row || !input_for_bottom_partial_image_row) {
            break;
        }
        if (!found_first_unskipped_row_in_bottom_partial_imgage) {
            if (v.next_img_h == v.img_h) {
                found_first_unskipped_row_in_bottom_partial_imgage = true;
            } else {
                TT_FATAL(v.next_img_h > v.img_h);
                num_skip_rows_bottom_partial_image += 1;
            }
        }
        v.input_flat_h += img_width;
        if (v.next_img_h == v.img_h) {
            v.output_flat_h += output_img_width;
        }
        v.img_w = 0;
        v.next_img_w = 0;
        TT_FATAL(v.img_h < img_height - 1); // this is supposed to be a bottom partial image
        v.img_h += 1;
        if (v.next_img_h < v.img_h) {
            v.next_img_h += img_stride_h;
            TT_FATAL(v.next_img_h <= img_height); // odd heights and odd size sharding with stride > 1 not supported
            if (v.next_img_h == img_height && v.img_h == img_height) {
                v.next_img_h = 0;
                v.img_h = 0;
                break;
            }
        }
        num_rows_bottom_partial_image += 1;
    }

    // Sanity check
    TT_FATAL(v.input_flat_h <= input_end_flat_h+1);
    TT_FATAL(v.output_flat_h <= output_end_flat_h+1);
    TT_FATAL(v.img_h < img_height && v.img_w < img_width);

    // check if there is a bottom partial left aligned row
    if (v.input_flat_h <= input_end_flat_h && v.output_flat_h <= output_end_flat_h) {
        TT_FATAL(v.img_w == 0 && v.next_img_w == 0);
        // bottom partial left aligned row width can be split between 2 cores
        uint32_t input_remaining = input_end_flat_h - v.input_flat_h + 1;
        uint32_t output_remaining = output_end_flat_h - v.output_flat_h + 1;
        TT_FATAL(output_remaining < output_img_width || input_remaining < img_width);  // there must be a partial width either on input side or output side
        bottom_partial_left_aligned_row_width = input_remaining;
        if (output_remaining < output_img_width) {
            bottom_partial_left_aligned_row_width = std::min(input_remaining, output_remaining * img_stride_w);
        }
        // sanity
        TT_FATAL(bottom_partial_left_aligned_row_width < img_width);
        TT_FATAL(v.next_img_h >= v.img_h);
        skip_bottom_partial_left_aligned_row = (v.next_img_h == v.img_h) ? 0 : 1;
        while(v.img_w < bottom_partial_left_aligned_row_width) {
            v.img_w += 1;
            if (v.next_img_w < v.img_w) {
                v.next_img_w += img_stride_w;
                TT_FATAL(v.next_img_w < img_width); // odd widths and odd size sharding with stride > 1 not supported
            }
        }
        TT_FATAL(v.img_w == bottom_partial_left_aligned_row_width && v.next_img_w >= v.img_w);
        v.input_flat_h += bottom_partial_left_aligned_row_width;
        if (!skip_bottom_partial_left_aligned_row) {
            v.output_flat_h += std::ceil( (double) bottom_partial_left_aligned_row_width / (double) img_stride_w);
        }
    }
    TT_FATAL(v.img_h < img_height && v.img_w < img_width);

    if (0) {
        log_debug(LogOp, "   top_partial_middle_aligned_row_width: {}", top_partial_middle_aligned_row_width);
        log_debug(LogOp, "   skip_top_partial_middle_aligned_row: {}", skip_top_partial_middle_aligned_row);
        log_debug(LogOp, "   top_partial_right_aligned_row_width: {}", top_partial_right_aligned_row_width);
        log_debug(LogOp, "   skip_top_partial_right_aligned_row: {}", skip_top_partial_right_aligned_row);
        log_debug(LogOp, "   num_rows_top_partial_image: {}", num_rows_top_partial_image);
        log_debug(LogOp, "   num_skip_rows_top_partial_image: {}", num_skip_rows_top_partial_image);
        log_debug(LogOp, "   num_full_images: {}", num_full_images);
        log_debug(LogOp, "   num_rows_bottom_partial_image: {}", num_rows_bottom_partial_image);
        log_debug(LogOp, "   num_skip_rows_bottom_partial_image: {}", num_skip_rows_bottom_partial_image);
        log_debug(LogOp, "   bottom_partial_left_aligned_row_width: {}", bottom_partial_left_aligned_row_width);
        log_debug(LogOp, "   skip_bottom_partial_left_aligned_row: {}", skip_bottom_partial_left_aligned_row);
        log_debug(LogOp, "   v.input_flat_h: {}", v.input_flat_h);
        log_debug(LogOp, "   v.output_flat_h: {}", v.output_flat_h);
        log_debug(LogOp, "   input_end_flat_h: {}", input_end_flat_h);
        log_debug(LogOp, "   output_end_flat_h: {}", output_end_flat_h);
        //cout << "img_h=" << v.img_h << ", img_w=" << v.img_w << ", next_img_h=" << v.next_img_h << ", next_img_w=" << v.img_w << endl;
    }

    // Sanity check
    TT_FATAL(v.input_flat_h <= input_end_flat_h+1);
    TT_FATAL(v.output_flat_h <= output_end_flat_h+1);

    if (v.input_flat_h == input_end_flat_h + 1) {
        v.input_flat_h = 0;
    }
    if (v.output_flat_h == output_end_flat_h + 1) {
        v.output_flat_h = 0;
    }
    return DownsampleReadPatternParams{.top_partial_middle_aligned_row_width=top_partial_middle_aligned_row_width,
                                    .skip_top_partial_middle_aligned_row=skip_top_partial_middle_aligned_row,
                                    .top_partial_right_aligned_row_width=top_partial_right_aligned_row_width,
                                    .skip_top_partial_right_aligned_row=skip_top_partial_right_aligned_row,
                                    .num_rows_top_partial_image=num_rows_top_partial_image,
                                    .num_skip_rows_top_partial_image=num_skip_rows_top_partial_image,
                                    .num_full_images=num_full_images,
                                    .num_rows_bottom_partial_image=num_rows_bottom_partial_image,
                                    .num_skip_rows_bottom_partial_image=num_skip_rows_bottom_partial_image,
                                    .bottom_partial_left_aligned_row_width=bottom_partial_left_aligned_row_width,
                                    .skip_bottom_partial_left_aligned_row=skip_bottom_partial_left_aligned_row};
}

operation::ProgramWithCallbacks downsample_single_core(const Tensor &a, std::array<uint32_t, 5> downsample_params, Tensor& output) {

    tt_metal::Program program = tt_metal::CreateProgram();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);
    tt::DataFormat untilized_cb_data_format = DataFormat::Float16_b;
    uint32_t untilized_single_tile_size = tt_metal::detail::TileSize(untilized_cb_data_format);
    auto [img_batch_size, img_height, img_width, img_stride_h, img_stride_w] = downsample_params;
    tt_metal::Buffer *src0_buffer = a.buffer();

    TT_FATAL(a.shape()[0] == 1 && a.shape()[1] == 1);
    TT_FATAL(output.shape()[0] == 1 && output.shape()[1] == 1);

    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Sanity check of output size
    TT_FATAL(output.volume() % TILE_HW == 0);
    uint32_t unpadded_input_volume = img_batch_size * img_height * img_width;
    TT_FATAL(a.volume() >= unpadded_input_volume);
    uint32_t unpadded_output_volume = ceil((double) unpadded_input_volume / (double) (img_stride_h * img_stride_w));
    TT_FATAL(output.volume() >= unpadded_output_volume);


    uint32_t ncores_x_full_grid = device->compute_with_storage_grid_size().x;
    auto [num_cores_height_sliced, num_cores_width_sliced] = get_num_cores_height_width_sliced(a.shard_spec().value().shard_grid,
                                        a.memory_config().memory_layout,
                                        a.shard_spec().value().shard_orientation);
    uint32_t num_cores = num_cores_height_sliced * num_cores_width_sliced;
    auto all_cores = a.shard_spec().value().shard_grid;
    auto memory_layout = a.memory_config().memory_layout;
    TT_FATAL(all_cores == output.shard_spec().value().shard_grid);
    TT_FATAL(memory_layout == output.memory_config().memory_layout);
    TT_FATAL(memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || memory_layout == TensorMemoryLayout::BLOCK_SHARDED);
    if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(all_cores.ranges().size() == 1);
    } else {
        TT_FATAL(num_cores_width_sliced == 1);
    }
    uint32_t num_cores_x = memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ? ncores_x_full_grid : num_cores_height_sliced;

    auto core_range = all_cores;

    uint32_t input_height = a.shape()[2]; // input height == flattened face of input image, multiple images are stacked in H dim
    uint32_t input_width = a.shape()[3]; // input width == input image # of channels
    uint32_t output_height = output.shape()[2]; // output height == flattened face of output image, multiple images are stacked in H dim
    uint32_t output_width = output.shape()[3];
    TT_FATAL(input_width == output_width);

    uint32_t input_height_unpadded = img_batch_size * img_height * img_width;
    TT_FATAL(input_height >= input_height_unpadded);
    uint32_t output_height_unpadded = img_batch_size * std::ceil((double) (img_height * img_width) / (double) (img_stride_h * img_stride_w));
    TT_FATAL(output_height >= output_height_unpadded);

    uint32_t input_shard_height = a.shard_spec().value().shard_shape[0];
    TT_FATAL(input_shard_height * num_cores_height_sliced >= input_height); // last core shard size may be padded
    uint32_t input_height_padded = input_shard_height * num_cores_height_sliced;
    uint32_t input_shard_width = a.shard_spec().value().shard_shape[1];
    TT_FATAL(input_shard_width * num_cores_width_sliced == input_width);

    uint32_t input_height_padding = input_height_padded - input_height_unpadded;
    TT_FATAL(input_height_padding < input_shard_height); // last core has padding
    uint32_t last_core_input_shard_height_unpadded = input_shard_height - input_height_padding;


    uint32_t output_shard_height = output.shard_spec().value().shard_shape[0];
    uint32_t output_shard_width = output.shard_spec().value().shard_shape[1];

    TT_FATAL(output_shard_width == input_shard_width);
    TT_FATAL(output_shard_height * num_cores_height_sliced >= output_height); // last core shard size may be padded
    uint32_t output_height_padded = output_shard_height * num_cores_height_sliced;
    uint32_t output_height_padding = output_height_padded - output_height_unpadded;
    TT_FATAL(output_height_padding < output_shard_height);
    uint32_t last_core_output_shard_height_unpadded = output_shard_height - output_height_padding;


    uint32_t input_shard_width_bytes = input_shard_width * datum_size(untilized_cb_data_format);

    TT_FATAL(input_shard_width % TILE_WIDTH == 0);
    uint32_t num_input_tiles_in_row = input_shard_width / TILE_WIDTH;
    TT_FATAL(input_shard_height % TILE_HEIGHT == 0);
    uint32_t num_rows_of_input_tiles = input_shard_height / TILE_HEIGHT;

    uint32_t num_output_tiles_in_row = num_input_tiles_in_row;
    TT_FATAL(output_shard_height % TILE_HEIGHT == 0);
    uint32_t num_rows_of_output_tiles = output_shard_height / TILE_HEIGHT;

    uint32_t input_cb_index = CB::c_in0;
    uint32_t num_input_tiles = num_input_tiles_in_row * num_rows_of_input_tiles;
    tt_metal::CircularBufferConfig input_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
		.set_page_size(input_cb_index, input_single_tile_size);
    input_cb_config = input_cb_config.set_globally_allocated_address(*a.buffer());
    auto input_cb = tt_metal::CreateCircularBuffer(program, core_range, input_cb_config);
    log_debug(LogOp, "CB {}: PS = {} NP = {}", input_cb_index, input_single_tile_size, num_input_tiles);

    // CB to store halo data
    // hardcode to store 1 row of tiles
    uint32_t halo_prev_input_cb_index = CB::c_in1;
    uint32_t halo_prev_input_cb_max_rows_of_tiles = 4;
    uint32_t num_halo_prev_cb_input_tiles = num_input_tiles_in_row  * halo_prev_input_cb_max_rows_of_tiles;
    tt_metal::CircularBufferConfig halo_prev_input_cb_config = tt_metal::CircularBufferConfig(num_halo_prev_cb_input_tiles * input_single_tile_size, {{halo_prev_input_cb_index, input_cb_data_format}})
		.set_page_size(halo_prev_input_cb_index, input_single_tile_size);
    auto halo_prev_input_cb = tt_metal::CreateCircularBuffer(program, core_range, halo_prev_input_cb_config);

    uint32_t halo_next_input_cb_index = CB::c_in2;
    uint32_t halo_next_input_cb_max_rows_of_tiles = 33; // TODO: Remove hardcoding
    uint32_t num_halo_next_cb_input_tiles = num_input_tiles_in_row  * halo_next_input_cb_max_rows_of_tiles;
    tt_metal::CircularBufferConfig halo_next_input_cb_config = tt_metal::CircularBufferConfig(num_halo_next_cb_input_tiles * input_single_tile_size, {{halo_next_input_cb_index, input_cb_data_format}})
		.set_page_size(halo_next_input_cb_index, input_single_tile_size);
    auto halo_next_input_cb = tt_metal::CreateCircularBuffer(program, core_range, halo_next_input_cb_config);
    // CB to store reader pattern array
    // read pattern array size == output_height
    uint32_t reader_pattern_array_size = output_shard_height;
    uint32_t reader_pattern_array_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig reader_pattern_array_cb_config = tt_metal::CircularBufferConfig(reader_pattern_array_size * 4, {{reader_pattern_array_cb_index, DataFormat::Float16_b}})
		.set_page_size(reader_pattern_array_cb_index, 4);
    auto reader_pattern_array_cb = tt_metal::CreateCircularBuffer(program, core_range, reader_pattern_array_cb_config);
    // untilized CB has size - [32, full width]
    uint32_t untilize_cb_index = CB::c_intermed2;
    uint32_t num_tiles_untilize_cb = num_input_tiles_in_row;
    tt_metal::CircularBufferConfig untilize_cb_config = tt_metal::CircularBufferConfig(num_tiles_untilize_cb * untilized_single_tile_size, {{untilize_cb_index, untilized_cb_data_format}})
		.set_page_size(untilize_cb_index, untilized_single_tile_size);
    auto untilize_cb = tt_metal::CreateCircularBuffer(program, core_range, untilize_cb_config);
    log_debug(LogOp, "CB {}: PS = {} NP = {}", untilize_cb_index, untilized_single_tile_size, num_tiles_untilize_cb);

    uint32_t num_output_tiles = num_output_tiles_in_row * num_rows_of_output_tiles;
    uint32_t untilize_downsampled_cb_index = CB::c_intermed3;
    uint32_t num_tiles_untilize_downsampled_cb = num_output_tiles; // untilize downsampled cb size == output size per core
    tt_metal::CircularBufferConfig untilize_downsampled_cb_config = tt_metal::CircularBufferConfig(num_tiles_untilize_downsampled_cb * untilized_single_tile_size, {{untilize_downsampled_cb_index, untilized_cb_data_format}})
		.set_page_size(untilize_downsampled_cb_index, untilized_single_tile_size);
    auto untilize_downsampled_cb = tt_metal::CreateCircularBuffer(program, core_range, untilize_downsampled_cb_config);

    uint32_t final_tilize_output_cb_index = CB::c_out0;
    uint32_t num_tiles_final_tilize_output_cb = num_output_tiles; // final output cb size == output size per core
    tt_metal::CircularBufferConfig final_tilize_output_cb_config = tt_metal::CircularBufferConfig(num_tiles_final_tilize_output_cb * output_single_tile_size, {{final_tilize_output_cb_index, output_cb_data_format}})
		.set_page_size(final_tilize_output_cb_index, output_single_tile_size);
    final_tilize_output_cb_config = final_tilize_output_cb_config.set_globally_allocated_address(*output.buffer());
    auto final_tilize_output_cb = tt_metal::CreateCircularBuffer(program, core_range, final_tilize_output_cb_config);

    uint32_t log_base_2_of_conv_act_size_c_bytes = (uint32_t) std::log2((float) input_shard_width_bytes);
    uint32_t stride_h_x_image_width = img_stride_h * img_width;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) untilize_cb_index,
        (std::uint32_t) untilize_downsampled_cb_index,
        (std::uint32_t) final_tilize_output_cb_index,
        (std::uint32_t) reader_pattern_array_cb_index,
        (std::uint32_t) datum_size(untilized_cb_data_format),
        (std::uint32_t) input_shard_width_bytes,
        (std::uint32_t) halo_prev_input_cb_index,
        (std::uint32_t) halo_next_input_cb_index,
        log_base_2_of_conv_act_size_c_bytes,
        stride_h_x_image_width
    };

    // Writer to downsample - drops rows from untilized cb
    tt_metal::KernelHandle downsample_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_writer_kernel.cpp",
        core_range,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        input_cb_index,
        halo_prev_input_cb_index,
        halo_next_input_cb_index,
        untilize_cb_index,
        untilize_downsampled_cb_index,
        final_tilize_output_cb_index,
        num_input_tiles_in_row,
        num_rows_of_output_tiles,
        num_output_tiles_in_row,
    };
    string compute_kernel = "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_compute_kernel.cpp";
    if (num_input_tiles_in_row <= MAX_PACK_UNTILIZE_WIDTH) {
        compute_kernel = "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_fast_pack_untilize_compute_kernel.cpp";
    }
    auto downsample_compute_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core_range,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    // track img h, img w, across cores
    ImgTrackingVars v;
    CoreCoord prev_core = {0,0};
    bool input_flat_h_is_of_current_core = true;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % num_cores_x, i / num_cores_x};
        //cout << "i=" << i << endl;

        uint32_t input_end_flat_h = input_shard_height - 1;
        uint32_t output_end_flat_h = output_shard_height - 1;

        bool halo_prev_read_enabled = false;
        DownsampleReadPatternParams halo_prev_read_pattern_params;
        uint32_t halo_prev_noc_x = 0;
        uint32_t halo_prev_noc_y = 0;
        uint32_t halo_prev_start_addr = 0;
        uint32_t halo_prev_addr_offset = 0;
        uint32_t halo_prev_num_tiles = 0;
        uint32_t halo_prev_size_bytes = 0;
        uint32_t halo_prev_input_num_rows_of_tiles = 0;
        uint32_t halo_prev_read_pattern_offset = 0;

        bool halo_next_read_enabled = false;
        DownsampleReadPatternParams halo_next_read_pattern_params;
        uint32_t halo_next_noc_x = 0;
        uint32_t halo_next_noc_y = 0;
        uint32_t halo_next_start_addr = 0;
        uint32_t halo_next_addr_offset = 0;
        uint32_t halo_next_num_tiles = 0;
        uint32_t halo_next_size_bytes = 0;
        uint32_t halo_next_input_num_rows_of_tiles = 0;
        uint32_t halo_next_read_pattern_offset = 0;
        uint32_t local_read_pattern_offset = 0;
        uint32_t current_core_input_end_flat_h = input_end_flat_h;
        uint32_t next_core_input_end_flat_h = input_end_flat_h;
        if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            if (i % num_cores_x == 0) {
                // first core in row
                // reset
                v.input_flat_h = 0;
                v.output_flat_h = 0;
            } else if (i % num_cores_x == num_cores_x - 1) {
                // set unpadded height as end index for last core in row
                current_core_input_end_flat_h = last_core_input_shard_height_unpadded - 1;
                output_end_flat_h = last_core_output_shard_height_unpadded - 1;
            } else if (i % num_cores_x == num_cores_x - 2) {
                next_core_input_end_flat_h = last_core_input_shard_height_unpadded - 1;
            }
        } else if (i == num_cores - 1) {
            // for height sharding, set unpadded height as end index for last core
            current_core_input_end_flat_h = last_core_input_shard_height_unpadded - 1;
            output_end_flat_h = last_core_output_shard_height_unpadded - 1;
        } else if (i == num_cores - 2) {
            next_core_input_end_flat_h = last_core_input_shard_height_unpadded - 1;
        }
        if (v.input_flat_h != 0 && !input_flat_h_is_of_current_core) {
            // halo region of previous core
            TT_FATAL(i != 0);
            if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(prev_core.y == core.y); // for block sharding case, prev core is left core
            }
            halo_prev_read_enabled = true;
            TT_FATAL(v.input_flat_h < input_shard_height);
            // get halo start tile address from height idx
            uint32_t halo_prev_start_tile_id_h = v.input_flat_h / TILE_HEIGHT;
            TT_FATAL(input_shard_height - v.input_flat_h <= TILE_HEIGHT * halo_prev_input_cb_max_rows_of_tiles); // halo input cb is hardcoded to store only 4 rows of tiles for now. TODO: allocate bigger CB or read in blocks
            // get halo size
            halo_prev_size_bytes = (input_shard_height - (halo_prev_start_tile_id_h * TILE_HEIGHT)) * input_shard_width / TILE_HW * input_single_tile_size;
            TT_FATAL(halo_prev_size_bytes % input_single_tile_size == 0);
            halo_prev_num_tiles = halo_prev_size_bytes / input_single_tile_size;
            TT_FATAL(halo_prev_num_tiles <= num_halo_prev_cb_input_tiles);
            TT_FATAL(halo_prev_num_tiles % num_input_tiles_in_row == 0);
            halo_prev_input_num_rows_of_tiles = halo_prev_num_tiles / num_input_tiles_in_row;
            halo_prev_addr_offset = num_input_tiles_in_row * halo_prev_start_tile_id_h * input_single_tile_size;
            halo_prev_start_addr = GetCircularBufferConfig(program, input_cb).globally_allocated_address().value();

            TT_FATAL((halo_prev_start_addr + halo_prev_addr_offset) % 32 == 0); // read address should be 32 byte aligned
            auto halo_noc_coords = device->worker_core_from_logical_core(prev_core);
            halo_prev_noc_x = halo_noc_coords.x;
            halo_prev_noc_y = halo_noc_coords.y;
            TT_FATAL(v.input_flat_h >= halo_prev_start_tile_id_h * TILE_HEIGHT);
            halo_prev_read_pattern_offset = v.input_flat_h - (halo_prev_start_tile_id_h * TILE_HEIGHT);
            local_read_pattern_offset = halo_prev_input_num_rows_of_tiles * TILE_HEIGHT;
            halo_prev_read_pattern_params = generate_downsample_read_pattern(v, img_height, img_width, img_stride_h, img_stride_w, input_end_flat_h, output_end_flat_h, true, false);
        }
        // local core
        TT_FATAL(v.output_flat_h < output_shard_height);
        uint32_t local_start_h = v.input_flat_h;
        DownsampleReadPatternParams local_read_pattern_params = generate_downsample_read_pattern(v, img_height, img_width, img_stride_h, img_stride_w, current_core_input_end_flat_h, output_end_flat_h, false, false);
        TT_FATAL(v.output_flat_h <= output_shard_height);
        uint32_t local_end_h_exclusive = v.input_flat_h == 0 ? input_shard_height : v.input_flat_h;
        uint32_t local_num_rows = local_end_h_exclusive - local_start_h;
        TT_FATAL(local_num_rows > 0);
        uint32_t local_input_num_rows_of_tiles = std::ceil( (double) local_num_rows / (double) TILE_HEIGHT);
        uint32_t local_input_offset_rows_of_tiles = local_start_h / TILE_HEIGHT;
        if (local_start_h != 0) {
            TT_FATAL(local_read_pattern_offset == 0);
            local_read_pattern_offset = local_start_h % TILE_HEIGHT;
        }
        if (v.input_flat_h != 0) {
            input_flat_h_is_of_current_core = false;
        } else {
            input_flat_h_is_of_current_core = true; // updating flag for next core
        }
        TT_FATAL(local_input_num_rows_of_tiles <= num_rows_of_input_tiles);

        if (v.output_flat_h != 0) {
            // need to read halo from next core
            TT_FATAL(i != num_cores - 1);
            TT_FATAL(v.input_flat_h == 0);
            CoreCoord next_core = {(i+1)% num_cores_x, (i+1) / num_cores_x};
            if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(next_core.y == core.y); // for block sharding case, next core is right core
            }
            halo_next_read_enabled = true;
            halo_next_read_pattern_params = generate_downsample_read_pattern(v, img_height, img_width, img_stride_h, img_stride_w, next_core_input_end_flat_h, output_end_flat_h, false, true);
            TT_FATAL(v.output_flat_h == 0);
            TT_FATAL(v.input_flat_h != 0 && v.input_flat_h < input_shard_height);
            TT_FATAL(v.input_flat_h <= TILE_HEIGHT * halo_next_input_cb_max_rows_of_tiles); // halo next input cb is hardcoded to store only 5 rows of tiles for now. TODO: allocate bigger CB or read in blocks
            uint32_t halo_next_end_tile_id_h = v.input_flat_h / TILE_HEIGHT;
            // get halo size
            halo_next_size_bytes = (halo_next_end_tile_id_h+1) * TILE_HEIGHT * input_shard_width / TILE_HW * input_single_tile_size;
            TT_FATAL(halo_next_size_bytes % input_single_tile_size == 0);
            halo_next_num_tiles = halo_next_size_bytes / input_single_tile_size;
            TT_FATAL(halo_next_num_tiles <= num_halo_next_cb_input_tiles);
            TT_FATAL(halo_next_num_tiles % num_input_tiles_in_row == 0);
            halo_next_input_num_rows_of_tiles = halo_next_num_tiles / num_input_tiles_in_row;
            halo_next_addr_offset = 0;
            halo_next_start_addr = GetCircularBufferConfig(program, input_cb).globally_allocated_address().value();
            TT_FATAL((halo_next_start_addr + halo_next_addr_offset) % 32 == 0); // read address should be 32 byte aligned
            auto halo_noc_coords = device->worker_core_from_logical_core(next_core);
            halo_next_noc_x = halo_noc_coords.x;
            halo_next_noc_y = halo_noc_coords.y;
            TT_FATAL(halo_prev_input_num_rows_of_tiles == 0);
            halo_next_read_pattern_offset = local_input_num_rows_of_tiles * TILE_HEIGHT;
        }
        TT_FATAL(v.output_flat_h == 0);

        // Compile runtime args
        vector<uint32_t> compile_rt_kernel_args = {
            local_input_num_rows_of_tiles,
            local_input_offset_rows_of_tiles,
            halo_prev_read_enabled,
            halo_prev_input_num_rows_of_tiles,
            halo_next_read_enabled,
            halo_next_input_num_rows_of_tiles,
        };

        tt_metal::SetRuntimeArgs(
            program,
            downsample_compute_kernel_id,
            core,
            compile_rt_kernel_args
        );

        // Writer runtime args
        vector<uint32_t> writer_kernel_args = {
            (uint32_t) img_height,
            (uint32_t) img_width,
            (uint32_t) img_stride_h,
            (uint32_t) img_stride_w,

            // halo prev args
            halo_prev_read_enabled,
            halo_prev_noc_x,
            halo_prev_noc_y,
            halo_prev_num_tiles,
            halo_prev_start_addr,
            halo_prev_addr_offset,
            halo_prev_size_bytes,

            // halo prev read pattern args
            halo_prev_read_pattern_offset,
            halo_prev_read_pattern_params.top_partial_middle_aligned_row_width,
            halo_prev_read_pattern_params.skip_top_partial_middle_aligned_row,
            halo_prev_read_pattern_params.top_partial_right_aligned_row_width,
            halo_prev_read_pattern_params.skip_top_partial_right_aligned_row,
            halo_prev_read_pattern_params.num_rows_top_partial_image,
            halo_prev_read_pattern_params.num_skip_rows_top_partial_image,
            halo_prev_read_pattern_params.num_full_images,
            halo_prev_read_pattern_params.num_rows_bottom_partial_image,
            halo_prev_read_pattern_params.num_skip_rows_bottom_partial_image,
            halo_prev_read_pattern_params.bottom_partial_left_aligned_row_width,
            halo_prev_read_pattern_params.skip_bottom_partial_left_aligned_row,

            // local read pattern args
            local_read_pattern_offset,
            local_read_pattern_params.top_partial_middle_aligned_row_width,
            local_read_pattern_params.skip_top_partial_middle_aligned_row,
            local_read_pattern_params.top_partial_right_aligned_row_width,
            local_read_pattern_params.skip_top_partial_right_aligned_row,
            local_read_pattern_params.num_rows_top_partial_image,
            local_read_pattern_params.num_skip_rows_top_partial_image,
            local_read_pattern_params.num_full_images,
            local_read_pattern_params.num_rows_bottom_partial_image,
            local_read_pattern_params.num_skip_rows_bottom_partial_image,
            local_read_pattern_params.bottom_partial_left_aligned_row_width,
            local_read_pattern_params.skip_bottom_partial_left_aligned_row,

            // halo next core args
            halo_next_read_enabled,
            halo_next_noc_x,
            halo_next_noc_y,
            halo_next_num_tiles,
            halo_next_start_addr,
            halo_next_addr_offset,
            halo_next_size_bytes,

            // halo next read pattern args
            halo_next_read_pattern_offset,
            halo_next_read_pattern_params.top_partial_middle_aligned_row_width,
            halo_next_read_pattern_params.skip_top_partial_middle_aligned_row,
            halo_next_read_pattern_params.top_partial_right_aligned_row_width,
            halo_next_read_pattern_params.skip_top_partial_right_aligned_row,
            halo_next_read_pattern_params.num_rows_top_partial_image,
            halo_next_read_pattern_params.num_skip_rows_top_partial_image,
            halo_next_read_pattern_params.num_full_images,
            halo_next_read_pattern_params.num_rows_bottom_partial_image,
            halo_next_read_pattern_params.num_skip_rows_bottom_partial_image,
            halo_next_read_pattern_params.bottom_partial_left_aligned_row_width,
            halo_next_read_pattern_params.skip_bottom_partial_left_aligned_row,

            halo_prev_input_num_rows_of_tiles + local_input_num_rows_of_tiles + halo_next_input_num_rows_of_tiles,
            num_input_tiles_in_row,
            num_output_tiles,

            (uint32_t) false
        };

        tt_metal::SetRuntimeArgs(
            program,
            downsample_writer_kernel_id,
            core,
            writer_kernel_args
        );
        prev_core = core;
    }

    auto override_runtime_args_callback = [
        input_cb=input_cb,
        final_tilize_output_cb=final_tilize_output_cb,
        downsample_writer_kernel_id=downsample_writer_kernel_id,
        num_cores=num_cores,
        num_cores_x=num_cores_x
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, input_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, final_tilize_output_cb, *dst_buffer);
        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i % num_cores_x, i / num_cores_x};
            auto &runtime_args = GetRuntimeArgs(program, downsample_writer_kernel_id, core);
            runtime_args[8] = src_buffer->address();
            runtime_args[39] = src_buffer->address();
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
