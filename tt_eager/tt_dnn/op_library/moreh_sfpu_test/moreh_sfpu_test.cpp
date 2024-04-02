// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_sfpu_test/moreh_sfpu_test.hpp"

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

void MorehSFPUTest::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {}

std::vector<Shape> MorehSFPUTest::compute_output_shapes(const std::vector<Tensor> &input_tensors) const { return {}; }

std::vector<Tensor> MorehSFPUTest::create_output_tensors(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    return {output_tensors.at(0).value()};
}

operation::ProgramWithCallbacks MorehSFPUTest::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = output_tensors.at(0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device{input.device()};
    auto program{CreateProgram()};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t num_tiles{1};
    tt::DataFormat input_data_format{tt_metal::datatype_to_dataformat_converter(input.get_dtype())};
    tt::DataFormat output_data_format{tt_metal::datatype_to_dataformat_converter(output.get_dtype())};
    log_debug(LogOp, fmt::format("input_data_format : {}", input_data_format).c_str());
    log_debug(LogOp, fmt::format("output_data_format : {}", output_data_format).c_str());

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord(core_grid.x_, num_cores_y);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] = tt_metal::split_work_to_cores(core_grid_coord, num_tiles);

    log_debug(LogOp, fmt::format("num_cores_to_be_used: {}", num_cores_to_be_used).c_str());
    log_debug(LogOp, fmt::format("num_tiles_per_core_group_1 : {}", num_tiles_per_core_group_1).c_str());
    log_debug(LogOp, fmt::format("num_tiles_per_core_group_2 : {}", num_tiles_per_core_group_2).c_str());
    log_debug(LogOp, fmt::format("num_tiles {}", num_tiles).c_str());

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{2};
    const uint32_t in1_t{2};
    const uint32_t intermed0_t{2};
    const uint32_t intermed1_t{2};
    const uint32_t out0_t{2};

    CreateCircularBuffer(
        program,
        all_cores,
        input_data_format,
        {
            {CB::c_in0, in0_t},
            {CB::c_intermed0, intermed0_t},
        });

    CreateCircularBuffer(
        program,
        all_cores,
        output_data_format,
        {
            {CB::c_out0, out0_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::string reader_kernel_file("tt_eager/tt_dnn/op_library/moreh_sfpu_test/kernels/reader_sfpu_test.cpp");
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores);

    const std::string writer_kernel_file("tt_eager/tt_dnn/op_library/moreh_sfpu_test/kernels/writer_sfpu_test.cpp");
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{};
    std::map<string, string> compute_defines;
    // compute_defines["SFPU_OP_IDENTITY_INCLUDE"] = "1";
    std::string test_define = "SFPU_OP_TEST_CASE_" + std::to_string(this->test_case);
    compute_defines[test_define] = "1";

    bool fp32_dest_acc = (input_data_format == tt::DataFormat::UInt32) ? (true) : (false);
    const auto compute_kernel_file = "tt_eager/tt_dnn/op_library/moreh_sfpu_test/kernels/moreh_sfpu_test.cpp";
    const auto compute_kernel_1_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {core_group_1, num_tiles_per_core_group_1, compute_args_group_1},
        compute_defines, MathFidelity::HiFi4, fp32_dest_acc);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_tiles_per_core_group_2, compute_args_group_2},
            compute_defines, MathFidelity::HiFi4, fp32_dest_acc);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr{input.buffer()->address()};
    const auto output_addr{output.buffer()->address()};
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores_to_be_used; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_tiles_per_core_group_1;
            SetRuntimeArgs(program, compute_kernel_1_id, core, {});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_tiles_per_core = num_tiles_per_core_group_2;
            SetRuntimeArgs(program, compute_kernel_2_id.value(), core, {});
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, {input_addr, num_output_tiles_per_core, num_tiles_written});

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
                                                   const std::vector<Tensor> &output_tensors) {};

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

Tensor moreh_sfpu_test(const Tensor &input, const Tensor &output, uint32_t test_case) {
    return operation::run(MorehSFPUTest{.test_case = test_case}, {input}, {}, {output}).at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
