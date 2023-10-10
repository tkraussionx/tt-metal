// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/downsample/downsample_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


void Downsample::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to downsample need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to downsample need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only downsample tile major data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
    TT_ASSERT(input_tensor_a.memory_config().is_sharded() && this->output_mem_config.is_sharded());
}

std::vector<Shape> Downsample::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.shape()[0] == 1 && input_tensor_a.shape()[1] == 1);
    uint32_t input_height = input_tensor_a.shape()[2];
    auto [input_height_size_z, input_height_size_y, input_height_size_x, height_y_stride, height_x_stride] = this->downsample_params;
    TT_ASSERT(input_height == input_height_size_z * input_height_size_y * input_height_size_x);
    uint32_t output_height = (input_height_size_z * ceil(input_height_size_y / height_y_stride) * ceil(input_height_size_x / height_x_stride));
    uint32_t output_width = input_tensor_a.shape()[3];
    return {Shape({1, 1, output_height, output_width})};
}

std::vector<Tensor> Downsample::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto input_shard_spec = input_tensor.shard_spec().value();
    auto input_shard_height = input_shard_spec.shard_shape[0];
    TT_ASSERT(input_shard_height % (this->downsample_params[3] * this->downsample_params[4]) == 0);
    uint32_t output_shard_height = input_shard_height / (this->downsample_params[3] * this->downsample_params[4]);
    auto output_shard_width = input_shard_spec.shard_shape[1];
    auto output_shard_grid = input_shard_spec.shard_grid;
    return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), Layout::TILE, input_tensor.device(), this->output_mem_config, ShardSpec{output_shard_grid, std::array<uint32_t, 2>{{output_shard_height, output_shard_width}}})};
}

operation::ProgramWithCallbacks Downsample::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {downsample_single_core(input_tensor_a, downsample_params, output_tensor)};
}

tt::stl::reflection::Attributes Downsample::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
        {"downsample_params", this->downsample_params}
    };
}

Tensor downsample(const Tensor &input_tensor_a, std::array<uint32_t, 5> downsample_params, const MemoryConfig& mem_config) {
    return operation::run_without_autoformat(Downsample{mem_config, downsample_params}, {input_tensor_a}).at(0);
}

operation::ProgramWithCallbacks downsample_single_core(const Tensor &a, std::array<uint32_t, 5> downsample_params, Tensor& output) {

    tt_metal::Program program = tt_metal::Program();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);
    auto [input_height_size_z, input_height_size_y, input_height_size_x, height_y_stride, height_x_stride] = downsample_params;
    tt_metal::Buffer *src0_buffer = a.buffer();

    TT_ASSERT(a.shape()[0] == 1 && a.shape()[1] == 1);
    TT_ASSERT(output.shape()[0] == 1 && output.shape()[1] == 1);

    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    // Sanity check of output size
    TT_ASSERT(output.volume() % TILE_HW == 0);
    TT_ASSERT(ceil(a.volume() / (height_y_stride * height_x_stride)) == output.volume());


    uint32_t ncores_x_full_grid = device->compute_with_storage_grid_size().x;
    uint32_t ncores_y_full_grid = device->compute_with_storage_grid_size().y;
    auto all_cores = a.shard_spec().value().shard_grid;
    TT_ASSERT(all_cores == output.shard_spec().value().shard_grid);
    uint32_t num_cores = 0;
    for (const auto& core_range : all_cores.ranges()) {
        num_cores += core_range.size();
    }
    uint32_t ncores = num_cores;
    auto core_range = all_cores;

    uint32_t input_height = a.shape()[2]; // input height == flattened face of input image, multiple images are stacked in H dim
    uint32_t input_width = a.shape()[3]; // input width == input image # of channels
    uint32_t output_height = output.shape()[2]; // output height == flattened face of output image, multiple images are stacked in H dim
    uint32_t output_width = output.shape()[3];
    TT_ASSERT(input_width == output_width);

    uint32_t input_shard_height = a.shard_spec().value().shard_shape[0];
    uint32_t input_shard_width = a.shard_spec().value().shard_shape[1];
    TT_ASSERT(input_shard_width == input_width); // tensor is sharded across height dim only

    uint32_t output_shard_height = output.shard_spec().value().shard_shape[0];
    uint32_t output_shard_width = output.shard_spec().value().shard_shape[1];
    TT_ASSERT(output_shard_width == output_width);

    uint32_t input_width_bytes = input_width * a.element_size();

    TT_ASSERT(input_width % TILE_WIDTH == 0);
    uint32_t num_input_tiles_in_row = input_width / TILE_WIDTH;
    TT_ASSERT(input_shard_height % TILE_HEIGHT == 0);
    uint32_t num_rows_of_input_tiles = input_shard_height / TILE_HEIGHT;

    TT_ASSERT(output_width % TILE_WIDTH == 0);
    uint32_t num_output_tiles_in_row = output_width / TILE_WIDTH;
    TT_ASSERT(output_shard_height % TILE_HEIGHT == 0);
    uint32_t num_rows_of_output_tiles = output_shard_height / TILE_HEIGHT;

    uint32_t input_cb_index = CB::c_in0;
    uint32_t num_input_tiles = num_input_tiles_in_row * num_rows_of_input_tiles;
    tt_metal::CircularBufferConfig input_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{input_cb_index, cb_data_format}})
		.set_page_size(input_cb_index, single_tile_size);
    input_cb_config = input_cb_config.set_globally_allocated_address(a.buffer()->address());
    auto input_cb = tt_metal::CreateCircularBuffer(program, core_range, input_cb_config);
    cout << "input cb created with - " << num_input_tiles << " tiles" << std::endl;
    // CB to store reader pattern array
    // read pattern array size == output_height
    uint32_t reader_pattern_array_size = output_height;
    cout << "output_height=" << output_height << endl;
    uint32_t reader_pattern_array_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig reader_pattern_array_cb_config = tt_metal::CircularBufferConfig(reader_pattern_array_size * 4, {{reader_pattern_array_cb_index, DataFormat::Float16_b}})
		.set_page_size(reader_pattern_array_cb_index, 4);
    auto reader_pattern_array_cb = tt_metal::CreateCircularBuffer(program, core_range, reader_pattern_array_cb_config);
    cout << "reader pattern cb created with - " << reader_pattern_array_size * 4 << " bytes" << std::endl;

    // untilized CB has size - [32, full width]
    uint32_t untilize_cb_index = CB::c_intermed1;
    uint32_t num_tiles_untilize_cb = num_input_tiles_in_row;
    tt_metal::CircularBufferConfig untilize_cb_config = tt_metal::CircularBufferConfig(num_tiles_untilize_cb * single_tile_size, {{untilize_cb_index, cb_data_format}})
		.set_page_size(untilize_cb_index, single_tile_size);
    auto untilize_cb = tt_metal::CreateCircularBuffer(program, core_range, untilize_cb_config);

    uint32_t num_output_tiles =  output.volume() / TILE_HW;
    assert(num_output_tiles == num_output_tiles_in_row * num_rows_of_output_tiles);
    uint32_t untilize_downsampled_cb_index = CB::c_intermed2;
    uint32_t num_tiles_untilize_downsampled_cb = num_output_tiles; // untilize downsampled cb size == output size
    tt_metal::CircularBufferConfig untilize_downsampled_cb_config = tt_metal::CircularBufferConfig(num_tiles_untilize_downsampled_cb * single_tile_size, {{untilize_downsampled_cb_index, cb_data_format}})
		.set_page_size(untilize_downsampled_cb_index, single_tile_size);
    auto untilize_downsampled_cb = tt_metal::CreateCircularBuffer(program, core_range, untilize_downsampled_cb_config);

    uint32_t final_tilize_output_cb_index = CB::c_out0;
    uint32_t num_tiles_final_tilize_output_cb = output.volume() / TILE_HW; // final output cb size == output size
    tt_metal::CircularBufferConfig final_tilize_output_cb_config = tt_metal::CircularBufferConfig(num_tiles_final_tilize_output_cb * single_tile_size, {{final_tilize_output_cb_index, cb_data_format}})
		.set_page_size(final_tilize_output_cb_index, single_tile_size);
    final_tilize_output_cb_config = final_tilize_output_cb_config.set_globally_allocated_address(output.buffer()->address());
    auto final_tilize_output_cb = tt_metal::CreateCircularBuffer(program, core_range, final_tilize_output_cb_config);

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) untilize_cb_index,
        (std::uint32_t) untilize_downsampled_cb_index,
        (std::uint32_t) final_tilize_output_cb_index,
        (std::uint32_t) reader_pattern_array_cb_index,
        (std::uint32_t) a.element_size(),
        (std::uint32_t) input_width_bytes,
    };

    // Writer to downsample - drops rows from untilized cb
    tt_metal::KernelID downsample_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_writer_kernel.cpp",
        core_range,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        input_cb_index,
        untilize_cb_index,
        untilize_downsampled_cb_index,
        final_tilize_output_cb_index,
        num_rows_of_input_tiles,
        num_input_tiles_in_row,
        num_rows_of_output_tiles,
        num_output_tiles_in_row,
    };

    auto downsample_compute_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/downsample/kernels/downsample_compute_kernel.cpp",
        core_range,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    // track img h, img w, n (img id), across cores
    uint32_t img_h = 0;
    uint32_t img_w = 0;
    uint32_t img_id = 0;
    uint32_t img_height = input_height_size_y;
    uint32_t img_width = input_height_size_x;

    uint32_t img_stride_h = height_y_stride;
    uint32_t img_stride_w = height_x_stride;

    // next img h, w read after striding
    uint32_t next_img_h = 0;
    uint32_t next_img_w = 0;

    // input flattened height
    uint32_t input_flat_h = 0;
    uint32_t current_core_end_flat_h = input_shard_height - 1;
    // !!ASSUMPTION!! in determining core coordinate is that all 12 cores in x dim are used
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i % ncores_x_full_grid, i / ncores_x_full_grid};
        cout << "i=" << i << endl;
        // Sanity checks at the start
        TT_ASSERT(next_img_h >= img_h);
        TT_ASSERT(next_img_w >= img_w);
        TT_ASSERT(input_flat_h < current_core_end_flat_h);
        TT_ASSERT(next_img_h < img_height);
        TT_ASSERT(next_img_w < img_width);

        uint32_t num_rows_top_partial_image = 0;
        uint32_t num_skip_rows_top_partial_image = 0;
        uint32_t num_full_images = 0;
        uint32_t num_rows_bottom_partial_image = 0;
        uint32_t bottom_partial_left_aligned_row_width = 0;
        uint32_t skip_bottom_partial_left_aligned_row = 1;

        // Top partial right aligned row section
        uint32_t top_partial_right_aligned_row_width = (img_w == 0) ? 0 : img_width - img_w;
        uint32_t skip_top_partial_right_aligned_row = (top_partial_right_aligned_row_width == 0) ? 1 : (next_img_h == img_h) ? 0 : 1;
        if (top_partial_right_aligned_row_width > 0) {
            img_w = 0;
            if (img_h == img_height - 1) {
                img_h = 0;
                next_img_h = 0;
            } else {
                img_h += 1;
                if (next_img_h < img_h) {
                    next_img_h += img_stride_h;
                }
            }
            input_flat_h += top_partial_right_aligned_row_width;
        }
        TT_ASSERT(input_flat_h < current_core_end_flat_h); // sharded height is at least 32
        TT_ASSERT(next_img_h >= img_h);
        TT_ASSERT(img_w == 0);

        uint32_t num_rows_remaining_of_current_image = (img_h == 0) ? 0 : img_height - img_h;
        if (num_rows_remaining_of_current_image > 0 && (input_flat_h += (num_rows_remaining_of_current_image * img_width) <= current_core_end_flat_h+1)) {
            // Top partial image section
            num_rows_top_partial_image = img_height - img_h;
            num_skip_rows_top_partial_image = next_img_h - img_h;
            // Sanity check
            TT_ASSERT((img_h + img_height == num_rows_top_partial_image));
            img_h = 0;
            next_img_h = 0;
            input_flat_h += (num_rows_top_partial_image * img_width);
            TT_ASSERT(input_flat_h <= current_core_end_flat_h+1);
        }

        while(input_flat_h + (img_height * img_width) <= current_core_end_flat_h+1) {
            input_flat_h += (img_height * img_width);
            img_h = 0;
            img_w = 0;
            next_img_h = 0;
            num_full_images += 1;
        }

        // Sanity check
        TT_ASSERT(img_h == 0 && img_w == 0 && next_img_h == 0);
        TT_ASSERT(input_flat_h <= current_core_end_flat_h+1);

        while (input_flat_h + img_width <= current_core_end_flat_h+1) {
            input_flat_h += img_width;
            img_w = 0;
            if (img_h == img_height - 1) {
                img_h = 0;
                next_img_h = 0;
            } else {
                img_h += 1;
                if (next_img_h < img_h) {
                    next_img_h += img_stride_h;
                }
            }
            num_rows_bottom_partial_image += 1;
        }
        if (input_flat_h < current_core_end_flat_h) {
            TT_ASSERT(img_w == 0);
            bottom_partial_left_aligned_row_width = current_core_end_flat_h - input_flat_h + 1;
            TT_ASSERT(bottom_partial_left_aligned_row_width < img_width);
            TT_ASSERT(next_img_h >= img_h);
            skip_bottom_partial_left_aligned_row = (next_img_h == img_h) ? 0 : 1;
            img_w = bottom_partial_left_aligned_row_width;
            input_flat_h += bottom_partial_left_aligned_row_width;
        }
        // Writer runtime args
        vector<uint32_t> writer_kernel_args = {
            (uint32_t) input_height_size_y,
            (uint32_t) input_height_size_x,
            (uint32_t) height_y_stride,
            (uint32_t) height_x_stride,
            top_partial_right_aligned_row_width,
            skip_top_partial_right_aligned_row,
            num_rows_top_partial_image,
            num_skip_rows_top_partial_image,
            num_full_images,
            num_rows_bottom_partial_image,
            bottom_partial_left_aligned_row_width,
            skip_bottom_partial_left_aligned_row,

            num_rows_of_input_tiles,
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
        TT_ASSERT(input_flat_h == current_core_end_flat_h+1);
        current_core_end_flat_h += input_shard_height;
    }

    auto override_runtime_args_callback = [
        input_cb=input_cb,
        final_tilize_output_cb=final_tilize_output_cb
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        auto& input_cb_config = GetCircularBufferConfig(program, input_cb);
        input_cb_config.set_globally_allocated_address(src_buffer->address());
        auto& final_tilize_output_cb_config = GetCircularBufferConfig(program, final_tilize_output_cb);
        final_tilize_output_cb_config.set_globally_allocated_address(dst_buffer->address());
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
