// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_relu_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

namespace ttnn::operations::moreh_eltwise {
MorehReluDeviceOperation::MultiCore::cached_program_t
MorehReluDeviceOperation::MultiCore::create(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &tensor_return_value) {
  using namespace tt;
  using namespace tt::tt_metal;

  const auto &input_tensor = tensor_args.input_tensor;
  auto &output_tensor = tensor_return_value;

  auto device_buffer0 = input_tensor.buffer();
  auto device_buffer1 = output_tensor.buffer();

  auto device = input_tensor.device();
  uint32_t num_tiles = input_tensor.volume() / tt::constants::TILE_HW;

  ////////////////////////////////////////////////
  Program program = CreateProgram();
  // Multi core part
  auto grid = device->compute_with_storage_grid_size();
  const auto [num_cores_unused, all_cores_unused, core_group_1, core_group_2,
              num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
      split_work_to_cores(grid, num_tiles);
  auto all_cores_in_grid = CoreRange({0, 0}, {grid.x - 1, grid.y - 1});
  auto target_cores = all_cores_in_grid;

  /////////////////////////////////////////////////////////////////////////////////
  // Allocate circular buffer 0 and 1.
  /////////////////////////////////////////////////////////////////////////////////
  auto cb_num_tiles = 2;
  auto cb0_id = CB::c_in0;
  auto cb0_data_format =
      tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
  auto page_size = tt::tt_metal::detail::TileSize(cb0_data_format);

  auto cb0_config = CircularBufferConfig(cb_num_tiles * page_size,
                                         {{cb0_id, cb0_data_format}})
                        .set_page_size(cb0_id, page_size);
  CreateCircularBuffer(program, target_cores, cb0_config);

  auto cb1_id = CB::c_out0;
  auto cb1_data_format =
      tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
  auto cb1_config = CircularBufferConfig(cb_num_tiles * page_size,
                                         {{cb1_id, cb1_data_format}})
                        .set_page_size(cb1_id, page_size);
  CreateCircularBuffer(program, target_cores, cb1_config);

  /////////////////////////////////////////////////////////////////////////////////
  // Create kernels
  /////////////////////////////////////////////////////////////////////////////////
  const uint32_t device_buffer0_is_dram =
      static_cast<uint32_t>(device_buffer0->buffer_type() == BufferType::DRAM);
  const uint32_t device_buffer1_is_dram =
      static_cast<uint32_t>(device_buffer1->buffer_type() == BufferType::DRAM);

  auto reader_kernel_id = CreateKernel(
      program,
      "ttnn/cpp/ttnn/operations/moreh_eltwise/moreh_relu/device/kernels/reader.cpp",
      target_cores, ReaderDataMovementConfig({device_buffer0_is_dram}, {}));

  auto writer_kernel_id = CreateKernel(
      program,
      "ttnn/cpp/ttnn/operations/moreh_eltwise/moreh_relu/device/kernels/writer.cpp",
      target_cores, WriterDataMovementConfig({device_buffer1_is_dram}, {}));

  auto compute_kernel_id = CreateKernel(
      program,
      "ttnn/cpp/ttnn/operations/moreh_eltwise/moreh_relu/device/kernels/compute.cpp",
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
    for (uint32_t x = 0; x < grid.x; ++x) {
      for (uint32_t y = 0; y < grid.y; ++y) {
        auto core = CoreCoord{x, y};
        uint32_t num_tiles_per_core = 0;

        // TODO. set num_tiles_per_core properly according to which group the
        // core belongs to.(group1? group2? or neither?)
        if (core_group_1.core_coord_in_core_ranges(core)) {
          num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
          num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
          num_tiles_per_core = 0;
        }

        /* TODO. set runtime args with tile_offset.*/
        const std::vector<uint32_t> reader_runtime_args{
            device_buffer0->address(), static_cast<uint32_t>(cb0_id),
            num_tiles_per_core, tile_offset};
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            device_buffer1->address(), static_cast<uint32_t>(cb1_id),
            num_tiles_per_core, tile_offset};

        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        // TODO. set runtime args of compute kernel.
        // compute kernel does not need to know tile offset.
        const std::vector<uint32_t> compute_runtime_args = {
            static_cast<uint32_t>(cb0_id), static_cast<uint32_t>(cb1_id),
            num_tiles_per_core};

        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_tiles_per_core;
      }
    }
  }

  return {std::move(program),
          {.reader_kernel_id = reader_kernel_id,
           .writer_kernel_id = writer_kernel_id,
           .compute_kernel_id = compute_kernel_id}};
}

void MorehReluDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t &cached_program,
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &tensor_return_value) {
  // auto& program = cached_program.program;
  // auto& unary_reader_kernel_id =
  // cached_program.shared_variables.unary_reader_kernel_id; auto&
  // unary_writer_kernel_id =
  // cached_program.shared_variables.unary_writer_kernel_id; auto& num_cores =
  // cached_program.shared_variables.num_cores; auto& num_cores_y =
  // cached_program.shared_variables.num_cores_y;

  // const auto& input_tensor = tensor_args.input_tensor;
  // auto& output_tensor = tensor_return_value;

  // auto src_buffer = input_tensor.buffer();
  // auto dst_buffer = output_tensor.buffer();

  // for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
  //     CoreCoord core = {i / num_cores_y, i % num_cores_y};

  //     {
  //         auto& runtime_args = GetRuntimeArgs(program,
  //         unary_reader_kernel_id, core); runtime_args[0] =
  //         src_buffer->address();
  //     }

  //     {
  //         auto& runtime_args = GetRuntimeArgs(program,
  //         unary_writer_kernel_id, core); runtime_args[0] =
  //         dst_buffer->address();
  //     }
  // }
}

} // namespace ttnn::operations::moreh_eltwise
