// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_copy/moreh_copy_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {}  // namespace

operation::ProgramWithCallbacks moreh_copy_impl(const Tensor &input, Tensor &output) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto num_tiles = output.volume() / TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord single_core = {0, 0};

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;

    const uint32_t out0_t = 1;

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    CreateCircularBuffer(
        program,
        single_core,
        cb_data_format,
        {
            {CB::c_in0, in0_t},
            {CB::c_out0, out0_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file = "tt_eager/tt_dnn/op_library/moreh_copy/kernels/dataflow/reader_moreh_copy.cpp";
    const auto writer_kernel_file = "tt_eager/tt_dnn/op_library/moreh_copy/kernels/dataflow/writer_moreh_copy.cpp";

    const std::vector<uint32_t> reader_compile_time_args{num_tiles, static_cast<uint32_t>(is_dram(input))};
    const std::vector<uint32_t> writer_compile_time_args{num_tiles, static_cast<uint32_t>(is_dram(output))};

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, single_core, reader_compile_time_args);
    const auto writer_kernels_id =
        CreateWriteKernel(program, writer_kernel_file, single_core, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file = "tt_eager/tt_dnn/op_library/moreh_copy/kernels/compute/moreh_copy_kernel.cpp";

    const std::vector<uint32_t> compute_compile_time_args{num_tiles};

    const auto compute_kernels_id =
        CreateComputeKernel(program, compute_kernel_file, {single_core, num_tiles, compute_compile_time_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();
    const auto output_addr = output.buffer()->address();

    const std::vector<uint32_t> reader_runtime_args{input_addr};
    SetRuntimeArgs(program, reader_kernels_id, single_core, reader_runtime_args);

    const std::vector<uint32_t> writer_runtime_args{output_addr};
    SetRuntimeArgs(program, writer_kernels_id, single_core, writer_runtime_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [](const Program &program,
                                             const std::vector<Buffer *> &input_buffers,
                                             const std::vector<Buffer *> &output_buffers) {
        //     auto input_buffer = input_buffers.at(0);
        //     auto gamma_buffer = input_buffers.at(1);
        //     auto beta_buffer = input_buffers.at(2);

        //     auto ouput_buffer = output_buffers.at(0);
        //     auto mean_buffer = output_buffers.at(1);
        //     auto rstd_buffer = output_buffers.at(2);

        //     for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
        //         CoreCoord core = {i / num_cores_y, i % num_cores_y};

        //         {
        //             auto runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
        //             runtime_args[0] = input_buffer->address();
        //             if (gamma_buffer != nullptr) {
        //                 runtime_args[2] = gamma_buffer->address();
        //             }
        //             if (beta_buffer != nullptr) {
        //                 runtime_args[5] = beta_buffer->address();
        //             }
        //             SetRuntimeArgs(program, reader_kernels_id, core, runtime_args);
        //         }

        //         {
        //             auto runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
        //             runtime_args[0] = ouput_buffer->address();
        //             if (mean_buffer != nullptr) {
        //                 runtime_args[2] = mean_buffer->address();
        //             }
        //             if (rstd_buffer != nullptr) {
        //                 runtime_args[5] = rstd_buffer->address();
        //             }
        //             SetRuntimeArgs(program, writer_kernels_id, core, runtime_args);
        //         }
        //     }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
