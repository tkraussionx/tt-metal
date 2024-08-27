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
    vector<bfloat16> &a,
    vector<bfloat16> &b,
    vector<bfloat16> &output,
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
    vector<bfloat16> &a,
    vector<bfloat16> &b,
    vector<bfloat16> &output,
    bool transpose_b,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    Device *device) {
    /*
     * Setup program and command queue
     */
    Program program = CreateProgram();
    CoreCoord core = CoreCoord{0, 0};  // single core
    CommandQueue &cq = device->command_queue();

    /*
     * EXtracting Matrix dimensions from input/output vectors
     */
    constexpr uint32_t one_tile_size_bytes = 2 * 32 * 32;
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    /*
     * Create DRAM Buffers for input and output vectors
     */
    InterleavedBufferConfig dram_buf_A_conf{
        .device = device,
        .size = Mt * Kt * one_tile_size_bytes,
        .page_size = one_tile_size_bytes,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED};  // check buffer layout
    InterleavedBufferConfig dram_buf_B_conf{
        .device = device,
        .size = Kt * Nt * one_tile_size_bytes,
        .page_size = one_tile_size_bytes,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED};  // check buffer layout
    InterleavedBufferConfig dram_buf_C_conf{
        .device = device,
        .size = Mt * Nt * one_tile_size_bytes,
        .page_size = one_tile_size_bytes,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED};  // check buffer layout

    std::shared_ptr<tt::tt_metal::Buffer> dram_buf_A = tt::tt_metal::CreateBuffer(dram_buf_A_conf);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buf_B = tt::tt_metal::CreateBuffer(dram_buf_B_conf);
    std::shared_ptr<tt::tt_metal::Buffer> dram_buf_C = tt::tt_metal::CreateBuffer(dram_buf_C_conf);

    /*
     * Config of Circular Buffer in the device L1
     * input tiles count is = 2 because it's single tile process, and double-buffer
     */
    uint32_t total_cb_tiles = 2;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t cbA_index = CB::c_in0;
    uint32_t cbB_index = CB::c_in1;
    uint32_t cbC_index = CB::c_out0;

    // NOTICE this
    CircularBufferConfig cb_in_A_conf =
        CircularBufferConfig(total_cb_tiles * one_tile_size_bytes, {{cbA_index, cb_data_format}})
            .set_page_size(cbA_index, one_tile_size_bytes);
    CircularBufferConfig cb_in_B_conf =
        CircularBufferConfig(total_cb_tiles * one_tile_size_bytes, {{cbB_index, cb_data_format}})
            .set_page_size(cbB_index, one_tile_size_bytes);
    CircularBufferConfig cb_out_C_conf =
        CircularBufferConfig(total_cb_tiles * one_tile_size_bytes, {{cbC_index, cb_data_format}})
            .set_page_size(cbC_index, one_tile_size_bytes);

    CBHandle cb_in_A = CreateCircularBuffer(program, core, cb_in_A_conf);
    CBHandle cb_in_B = CreateCircularBuffer(program, core, cb_in_B_conf);
    CBHandle cb_in_C = CreateCircularBuffer(program, core, cb_out_C_conf);

    /*
     * Compile time arguments
     */
    // ???? why need compile-time args
    uint32_t is_buffer_A_DRAM = 1;
    uint32_t is_buffer_B_DRAM = 1;
    uint32_t is_buffer_C_DRAM = 1;
    uint32_t tranpose_b = 0;

    std::vector<uint32_t> reader_compile_time_args{is_buffer_A_DRAM, is_buffer_B_DRAM};
    std::vector<uint32_t> writer_compile_time_args{is_buffer_C_DRAM};
    std::vector<uint32_t> compute_compile_time_args{Mt, Kt, Nt, tranpose_b};

    /*
     * Create Kernels (Reader, Writer, Compute)
     */
    KernelHandle reader_id = CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_exercise/kernels/dataflow/reader_matmul_single_core.cpp",
        core,
        ReaderDataMovementConfig(reader_compile_time_args));  // use default value for R-RSCV and NOC
    KernelHandle writer_id = CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_exercise/kernels/dataflow/writer_matmul_single_core.cpp",
        core,
        WriterDataMovementConfig(writer_compile_time_args));
    KernelHandle computer_id = CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_exercise/kernels/compute/matmul_single_core.cpp",
        core,
        ComputeConfig{.compile_args = compute_compile_time_args});

    /*
     * Kernels - Runtime arguments
     */
    uint32_t dram_A_base_addr = dram_buf_A->address();
    uint32_t dram_B_base_addr = dram_buf_B->address();
    uint32_t dram_C_base_addr = dram_buf_C->address();
    std::vector<uint32_t> reader_runtime_args{
        dram_A_base_addr,
        dram_B_base_addr,
        Mt,
        Kt,
        Nt,
        Mt * Kt,
        Kt * Nt,
    };
    std::vector<uint32_t> writer_runtime_args{
        dram_C_base_addr,
        Mt,
        Nt,
    };
    SetRuntimeArgs(program, reader_id, core, reader_runtime_args);
    SetRuntimeArgs(program, writer_id, core, writer_runtime_args);

    /* Launch program & read in output buffer result into the host vector */
    EnqueueWriteBuffer(cq, dram_buf_A, a.data(), false);
    EnqueueWriteBuffer(cq, dram_buf_B, b.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dram_buf_C, output.data(), true);
}

int main(int argc, char **argv) {
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
    } catch (const std::exception &e) {
        log_error(LogTest, "Command line arguments found exception", e.what());
    }
    log_info(tt::LogVerif, "M {} N {} K {} transpose_b {}", M, N, K, transpose_b);

    try {
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);

        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        constexpr uint32_t one_tile_size_bytes = 2 * 32 * 32;
        uint32_t dram_buf_A_size_bytes = one_tile_size_bytes * Mt * Kt;
        uint32_t dram_buf_B_size_bytes = one_tile_size_bytes * Nt * Kt;
        uint32_t dram_buf_C_size_bytes = one_tile_size_bytes * Mt * Nt;

        vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_native(dram_buf_A_size_bytes, 1, 123);
        vector<bfloat16> src1_vec = create_random_vector_of_bfloat16_native(dram_buf_B_size_bytes, 1, 12522);
        vector<bfloat16> golden_vec(M * N, 0);
        if (!skip_pcc) {
            golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K, transpose_b);
        }

        // Check mem layout of A, B matrix to four 16x16 tiles layout
        tilize(src0_vec, M, K);
        tilize(src1_vec, K, N);

        /* Calling the MatMul host program. Read in result into a host vector */
        vector<bfloat16> result_vec(dram_buf_C_size_bytes / sizeof(bfloat16));
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

    } catch (const std::exception &e) {
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
