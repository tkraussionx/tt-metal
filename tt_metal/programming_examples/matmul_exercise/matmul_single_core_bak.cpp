// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

void golden_matmul(
    vector<bfloat16>& a,
    vector<bfloat16>& b,
    vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool transpose = false) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    vector<bfloat16> c_bf(M * N, 0);
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            idx_c = j + (i * N);
            idx_a = i * K;
            idx_b = transpose ? j * K : j;
            c_f = 0;
            for (uint32_t k_m = 0; k_m < K; k_m++) {
                float_tmp = a[idx_a].to_float() * (b[idx_b].to_float());
                c_f += float_tmp;
                idx_a += 1;
                idx_b += transpose ? 1 : N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

void matmul_single_core(
    vector<bfloat16>& a,
    vector<bfloat16>& b,
    vector<bfloat16>& output,
    bool transpose_b,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    Device* device) {
    /*
     * Setup program to execute along with its buffers and kernels to use
     * Core range is just single core
     */
    CommandQueue& cq = device->command_queue();
    Program program{};
    CoreRange core({0, 0}, {0, 0});

    /*
     * EXtracting Matrix dimensions from input/output vectors
     */
    // C = A*B
    // MN = MK*KN
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    /*
     * Create DRAM Buffers for input and output vectors
     * Writing data from input vectors to source buffers
     */
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_tile_size = 2 * 32 * 32;

    uint32_t dram_buffer_A_size =
        single_tile_size * Mt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size =
        single_tile_size * Nt * Kt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size =
        single_tile_size * Mt * Nt;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    /* DRAM buffer size = input full size */
    /* limiting page_size = single tile size; to allow DRAM channels interleaving */

    tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = dram_buffer_A_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = dram_buffer_B_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = dram_buffer_C_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_A);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config_B);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_C);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    /*
     * Config of Circular Buffer in the device L1
     * input tiles count is = 2 because it's single tile process, and double-buffer
     */
    uint32_t src0_cb_index = CB::c_in0;  // 0
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1;  // 1
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    /*
     * Compile time arguments
     */
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    /*
     * Create Kernels (Reader, Writer, Compute)
     */
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_exercise/kernels/dataflow/reader_matmul_single_core.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_exercise/kernels/dataflow/writer_matmul_single_core.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        Mt,  // Mt
        Kt,  // Kt
        Nt,  // Nt
        uint32_t(transpose_b ? 1 : 0)};
    auto matmul_single_core_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_exercise/kernels/compute/matmul_single_core.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args});

    /*
     * Kernels - Runtime arguments
     */
    tt_metal::SetRuntimeArgs(
        program, reader_id, core, {src0_addr, src1_addr, Mt, Kt, Nt, Mt * Kt, Kt * Nt, uint32_t(transpose_b ? 1 : 0)});

    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Mt, Nt});

    /* Launch program & read in output buffer result into the host vector */
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
}

///////////////////////////////////////

int main(int argc, char** argv) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    uint32_t M = 32;
    uint32_t N = 32;
    uint32_t K = 32;
    bool transpose_b = false;
    bool skip_pcc = false;
    try {
        std::tie(M, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "-m", 32);
        std::tie(N, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "-n", 32);
        std::tie(K, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "-k", 32);
        std::tie(transpose_b, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--transpose-b");
        std::tie(skip_pcc, input_args) = test_args::has_command_option_and_remaining_args(input_args, "--skip-pcc");

        test_args::validate_remaining_args(input_args);
    } catch (const std::exception& e) {
        log_error(LogTest, "Command line arguments found exception", e.what());
    }
    log_info(tt::LogVerif, "M {} N {} K {} transpose_b {}", M, N, K, transpose_b);

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        Device* device = CreateDevice(device_id);

        /* Create source data */
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt;  // num_tiles of FP16_B
        uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt;  // num_tiles of FP16_B
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;  // num_tiles of FP16_B

        /* input vectors with various ranges of values */
        std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_native(dram_buffer_A_size, 1, 123);
        std::vector<bfloat16> src1_vec = create_random_vector_of_bfloat16_native(dram_buffer_B_size, 1, 12522);

        /* Golden Matmul running on CPU (Float)*/
        vector<bfloat16> golden_vec(M * N, 0);
        if (!skip_pcc) {
            golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K, transpose_b);
        }

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                cout << src0_vec[i * K + j] << " ";
            }
            cout << endl;
        }
        cout << endl;
        /* Input vector tilizing */
        tilize(src0_vec, M, K);
        tilize(src1_vec, K, N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                cout << src0_vec[i * K + j] << " ";
            }
            cout << endl;
        }

        /* Calling the MatMul host program. Read in result into a host vector */
        vector<bfloat16> result_vec(dram_buffer_C_size / sizeof(bfloat16));
        matmul_single_core(src0_vec, src1_vec, result_vec, transpose_b, M, N, K, device);
        untilize(result_vec, M, N);

        log_info(tt::LogVerif, "Output vector of size {}", result_vec.size());
        if (!skip_pcc) {
            float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
            log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
            TT_FATAL(pearson > 0.99, "PCC not high enough. Result PCC: {}, Expected PCC: 0.99", pearson);
        } else {
            log_info(tt::LogVerif, "Skip PCC");
        }

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
