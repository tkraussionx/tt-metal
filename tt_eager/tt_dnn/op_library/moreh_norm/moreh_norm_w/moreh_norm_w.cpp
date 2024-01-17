// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {
std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    const bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}
}  // namespace

operation::ProgramWithCallbacks moreh_norm_w_impl(const Tensor &input, float p, int64_t dim, const Tensor &output) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = input.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.shape();

    const auto N = input_shape[0];
    const auto C = input_shape[1];
    const auto H = input_shape[2];
    const auto W = input_shape[3];

    const auto Ht = H / TILE_HEIGHT;
    const auto Wt = W / TILE_WIDTH;

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord = {.x = core_grid.x_, .y = num_cores_y};

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = tt_metal::split_work_to_cores(core_grid_coord, N * C * Ht);

    std::cout << "num_cores_to_be_used: " << num_cores_to_be_used << std::endl;
    std::cout << "num_rows_per_core_group_1: " << num_rows_per_core_group_1 << std::endl;
    std::cout << "num_rows_per_core_group_2: " << num_rows_per_core_group_2 << std::endl;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    // const auto single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    const uint32_t in0_t{1};  // input
    const uint32_t in1_t{1};  // one
    const uint32_t in2_t{1};  // decimal

    const uint32_t out0_t{1};  // output

    // uint32_t im0_t{1};  // x
    // uint32_t im1_t{1};  // |x|^p
    // uint32_t im2_t{1};  // Sum(|x|^p)

    const uint32_t im0_t{1};  // |x|
    const uint32_t im1_t{1};  // log(|x|)
    const uint32_t im2_t{1};  // exp(log(|x|) * decimal)
    const uint32_t im3_t{1};  // |x|^p
    const uint32_t im4_t{1};  // |x|^p * exp(log(|x|) * decimal) == |x + decimal|^p
    const uint32_t im5_t{1};  // Add(|x + decimal|^p)
    const uint32_t im6_t{1};  // Sum(|x + decimal|^p)

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},    // input
            {CB::c_in1, in1_t},    // one
            {CB::c_in2, in2_t},    // decimal
            {CB::c_out0, out0_t},  // output
            {CB::c_intermed0, im0_t},
            {CB::c_intermed1, im1_t},
            {CB::c_intermed2, im2_t},
            {CB::c_intermed3, im3_t},
            {CB::c_intermed4, im4_t},
            {CB::c_intermed5, im5_t},
            {CB::c_intermed6, im6_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // const std::vector<uint32_t> reader_compile_time_args{};
    // const std::vector<uint32_t> writer_compile_time_args{};

    const auto reader_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_w/kernels/"
        "reader_moreh_norm_w.cpp";
    const auto writer_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_w/kernels/"
        "writer_moreh_norm_w.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores, {});
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores, {});

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    const auto compute_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_w/kernels/"
        "moreh_norm_w_kernel.cpp";

    // const std::vector<uint32_t> compute_args_group_1{};

    const auto compute_kernels_id_1 = CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_rows_per_core_group_1, {}}, compute_defines);

    KernelHandle compute_kernels_id_2;
    if (!core_group_2.ranges().empty()) {
        // const std::vector<uint32_t> compute_args_group_2{};

        compute_kernels_id_2 = CreateComputeKernel(
            program, compute_kernel_file, {core_group_2, num_rows_per_core_group_2, {}}, compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();
    const auto output_addr = output.buffer()->address();

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        KernelHandle compute_kernel_id;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
            compute_kernel_id = compute_kernels_id_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
            compute_kernel_id = compute_kernels_id_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            *reinterpret_cast<uint32_t *>(&decimal),
            num_rows_per_core,
            Wt,
            tile_offset};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            output_addr, static_cast<uint32_t>(is_dram(output)), num_rows_per_core, Wt, tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{
            num_rows_per_core, Wt, static_cast<uint32_t>(floored_p), static_cast<uint32_t>(p_is_negative)};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_rows_per_core * Wt;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback =
        [](const Program &program, const std::vector<Buffer *> &, const std::vector<Buffer *> &) {};

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
