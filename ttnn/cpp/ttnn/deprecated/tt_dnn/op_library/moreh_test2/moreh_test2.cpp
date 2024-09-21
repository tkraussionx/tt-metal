// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_test2/moreh_test2_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"

using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_test2_impl(
    const Tensor &input1,
    const Tensor &cond,
    const Tensor &input2,
    const Tensor &output,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device{input1.device()};
    auto program{CreateProgram()};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t num_tiles{input1.volume() / tt::constants::TILE_HW};
    tt::DataFormat input1_data_format{tt_metal::datatype_to_dataformat_converter(input1.get_dtype())};
    tt::DataFormat cond_data_format{tt_metal::datatype_to_dataformat_converter(cond.get_dtype())};
    tt::DataFormat input2_data_format{tt_metal::datatype_to_dataformat_converter(input2.get_dtype())};

    log_info(LogOp, "input1 data format {}", input1_data_format);
    log_info(LogOp, "cond data format {}", cond);
    log_info(LogOp, "input2 data format {}", input2_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(input1.device()->arch(), compute_kernel_config);
    log_info(
        LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] = tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    log_info(LogOp, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_info(LogOp, "num_tiles_per_core_group_1 : {}", num_tiles_per_core_group_1);
    log_info(LogOp, "num_tiles_per_core_group_2 : {}", num_tiles_per_core_group_2);
    log_info(LogOp, "num_tiles {}", num_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{2};        // input1, double buffer
    const uint32_t in1_t{2};        // cond, double buffer
    const uint32_t in2_t{2};        // input2, double buffer
    const uint32_t out0_t{2};       // output, double buffer

    log_info(LogOp, "tile size for CB in0_t {}", in0_t);
    log_info(LogOp, "tile size for CB in1_t {}", in1_t);
    log_info(LogOp, "tile size for CB in2_t {}", in2_t);
    log_info(LogOp, "tile size for CB out0_t {}", out0_t);

    CreateCircularBuffer(
        program,
        all_cores,
        input1_data_format,
        {
            {CB::c_in0, in0_t},
        });
    CreateCircularBuffer(
        program,
        all_cores,
        cond_data_format,
        {
            {CB::c_in1, in1_t},
        });
    CreateCircularBuffer(
        program,
        all_cores,
        input2_data_format,
        {
            {CB::c_in2, in2_t},
        });
    CreateCircularBuffer(
        program,
        all_cores,
        input1_data_format,
        {
            {CB::c_out0, out0_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(is_dram(input1)),
        static_cast<uint32_t>(is_dram(cond)),
        static_cast<uint32_t>(is_dram(input2))};

    const std::string reader_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_test2/kernels/reader_test2.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(is_dram(output))};
    const std::string writer_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_test2/kernels/writer_test2.cpp";
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_tiles_per_core_group_1};

    std::map<string, string> compute_defines{
        {"FP32_DEST_ACC_EN", "1"},
    };
    const bool is_fp32 = input1.get_dtype() == DataType::FLOAT32;
    const auto compute_kernel_file = "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/moreh_test2/kernels/moreh_test2.cpp";
    const auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_tiles_per_core_group_1, compute_args_group_1},
        compute_defines,
        /*math_fidelity=*/MathFidelity::HiFi4,
        /*fp32_dest_acc_en=*/is_fp32,
        /*math_approx_mode=*/false,
        /*preserve_fp32_precision=*/is_fp32);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_tiles_per_core_group_2};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_tiles_per_core_group_2, compute_args_group_2},
            compute_defines,
            /*math_fidelity=*/MathFidelity::HiFi4,
            /*fp32_dest_acc_en=*/is_fp32,
            /*math_approx_mode=*/false,
            /*preserve_fp32_precision=*/is_fp32);
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // input1, cond, input2, output
    const auto input1_addr{input1.buffer()->address()};
    const auto cond_addr{cond.buffer()->address()};
    const auto input2_addr{input2.buffer()->address()};
    const auto output_addr{output.buffer()->address()};
    log_info(LogOp, "input1_addr {}", input1_addr);
    log_info(LogOp, "cond_addr {}", cond_addr);
    log_info(LogOp, "input2_addr {}", input2_addr);
    log_info(LogOp, "output_addr {}", output_addr);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_to_be_used; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, {input1_addr, cond_addr, input2_addr, num_output_tiles_per_core, num_tiles_written});

        tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {output_addr, num_output_tiles_per_core, num_tiles_written});

        num_tiles_written += num_output_tiles_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        log_info(LogOp, "{}:{} args_callback ", __func__, __LINE__);
        const auto *input1_buffer = input_tensors.at(0).buffer();
        const auto *cond_buffer = input_tensors.at(1).buffer();
        const auto *input2_buffer = input_tensors.at(2).buffer();
        const auto *output_buffer = output_tensors.at(0).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = input1_buffer->address();
                runtime_args[1] = cond_buffer->address();
                runtime_args[2] = input2_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = output_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
