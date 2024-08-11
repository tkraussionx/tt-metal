// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_binary_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

namespace ttnn::operations::moreh_eltwise {
MorehBinaryDeviceOperation::Fusion::cached_program_t
MorehBinaryDeviceOperation::Fusion::create(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &tensor_return_value) {
  using namespace tt;
  using namespace tt::tt_metal;

  const auto &input_tensor0 = tensor_args.input_tensor0;
  const auto &input_tensor1 = tensor_args.input_tensor1;
  auto &output_tensor = tensor_return_value;

  auto input0_buffer = input_tensor0.buffer();
  auto input1_buffer = input_tensor1.buffer();
  auto output_buffer = output_tensor.buffer();

  auto device = input_tensor0.device();
  uint32_t num_tiles = input_tensor0.volume() / tt::constants::TILE_HW;

  ////////////////////////////////////////////////
  Program program = CreateProgram();
  auto grid = device->compute_with_storage_grid_size();
  const auto [num_cores_unused, all_cores_unused, core_group_1, core_group_2,
              num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
      split_work_to_cores(grid, num_tiles);
  auto all_cores_in_grid = CoreRange({0, 0}, {grid.x - 1, grid.y - 1});
  auto target_cores = all_cores_in_grid;

  /////////////////////////////////////////////////////////////////////////////////
  // Allocate circular buffer 0 and 1.
  /////////////////////////////////////////////////////////////////////////////////

  auto create_cb = [&program, &target_cores](auto t, CB id) {
    auto cb_num_tiles = 1;
    auto data_format =
        tt::tt_metal::datatype_to_dataformat_converter(t.get_dtype());
    auto page_size = tt::tt_metal::detail::TileSize(data_format);
    auto config =
        CircularBufferConfig(cb_num_tiles * page_size, {{id, data_format}})
            .set_page_size(id, page_size);
    CreateCircularBuffer(program, target_cores, config);
  };

  auto input0_cb = CB::c_in0;
  auto input1_cb = CB::c_in1;
  auto output_cb = CB::c_out0;

  create_cb(input_tensor0, input0_cb);
  create_cb(input_tensor1, input1_cb);
  create_cb(output_tensor, output_cb);

  /////////////////////////////////////////////////////////////////////////////////
  // Create kernels
  /////////////////////////////////////////////////////////////////////////////////
  const uint32_t input0_is_dram =
      static_cast<uint32_t>(input0_buffer->buffer_type() == BufferType::DRAM);
  const uint32_t input1_is_dram =
      static_cast<uint32_t>(input1_buffer->buffer_type() == BufferType::DRAM);
  const uint32_t output_is_dram =
      static_cast<uint32_t>(output_buffer->buffer_type() == BufferType::DRAM);

  auto reader_kernel_id = CreateKernel(
      program,
      "ttnn/cpp/ttnn/operations/moreh_eltwise/moreh_binary/device/kernels/"
      "reader.cpp",
      target_cores,
      ReaderDataMovementConfig({input0_is_dram, input1_is_dram}, {}));

  auto writer_kernel_id = CreateKernel(
      program,
      "ttnn/cpp/ttnn/operations/moreh_eltwise/moreh_binary/device/kernels/"
      "writer.cpp",
      target_cores, WriterDataMovementConfig({output_is_dram}, {}));

  auto compute_kernel_id =
      CreateKernel(program,
                   "ttnn/cpp/ttnn/operations/moreh_eltwise/moreh_binary/device/"
                   "kernels/fusion.cpp",
                   target_cores,
                   ComputeConfig{
                       .compile_args = {},
                       .defines = {},
                   });
  /////////////////////////////////////////////////////////////////////////////////
  // Set runtime args
  /////////////////////////////////////////////////////////////////////////////////
  {
    uint32_t tile_offset = 0;
    using FloatBits = union {
      float f;
      uint32_t bits;
    };

    FloatBits slope0 = {.f = operation_attributes.scalar0};
    FloatBits slope1 = {.f = operation_attributes.scalar1};

    for (uint32_t x = 0; x < grid.x; ++x) {
      for (uint32_t y = 0; y < grid.y; ++y) {
        auto core = CoreCoord{x, y};
        uint32_t num_tiles_per_core = 0;

        if (core_group_1.core_coord_in_core_ranges(core)) {
          num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
          num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
          num_tiles_per_core = 0;
        }

        const std::vector<uint32_t> reader_runtime_args{
            input0_buffer->address(),
            input1_buffer->address(),
            static_cast<uint32_t>(input0_cb),
            static_cast<uint32_t>(input1_cb),
            num_tiles_per_core,
            tile_offset};
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            output_buffer->address(), static_cast<uint32_t>(output_cb),
            num_tiles_per_core, tile_offset};

        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        const std::vector<uint32_t> compute_runtime_args = {
            static_cast<uint32_t>(input0_cb),
            static_cast<uint32_t>(input1_cb),
            static_cast<uint32_t>(output_cb),
            slope0.bits,
            slope1.bits,
            num_tiles_per_core};

        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_tiles_per_core;
      }
    }
  }

  return {std::move(program),
          {.reader_kernel_id = reader_kernel_id,
           .writer_kernel_id = writer_kernel_id,
           .compute_kernel_id = compute_kernel_id,
           .grid = grid}};
}

void MorehBinaryDeviceOperation::Fusion::override_runtime_arguments(
    cached_program_t &cached_program,
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &tensor_return_value) {
      // TODO
      // Update runtime arguemtns
    }

} // namespace ttnn::operations::moreh_eltwise
