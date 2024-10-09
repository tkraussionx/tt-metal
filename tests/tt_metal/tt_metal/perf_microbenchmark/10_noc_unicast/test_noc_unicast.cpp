// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <string>

#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the performance of the noc_async_write unicast operation.
// The source Tensix core writes to the L1 of the target Tensix core and then
// increments as semaphore. The target Tensix core waits for the semaphore to
// be incremented by the source core, and the time taken for the target core to
// receive all the data + semaphore is measured.
//
// Usage example:
//   ./test_noc_adjacent
//     --src-core-x <source core x>
//     --src-core-y <source core y>
//     --target-core-x <target core x>
//     --target-core-y <target core y>
//     --num-tiles <number of tiles for each transfer>
//     --tiles-per-transfer <number of transfers>
//     --noc-index <NOC index to use>
//     --num-tests <count of tests>
//     --use-device-profiler (set to use device profiler for measurement)
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    std::vector<double> measured_bandwidth;

    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);

    uint32_t src_core_x = 0;
    uint32_t src_core_y = 0;
    uint32_t target_core_x = 1;
    uint32_t target_core_y = 0;
    uint32_t num_tiles = 512;
    uint32_t num_transfers = 1;
    uint32_t noc_index = 0;
    uint32_t num_tests = 10;
    bool use_device_profiler = false;
    bool bypass_check = false;
    try {
        std::tie(src_core_x, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--src-core-x", 0);
        std::tie(src_core_y, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--src-core-y", 0);

        std::tie(target_core_x, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--target-core-x", 1);
        std::tie(target_core_y, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--target-core-y", 0);

        std::tie(num_tiles, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tiles", 512);

        std::tie(num_transfers, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-transfers", 1);

        std::tie(noc_index, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--noc-index", 0);

        std::tie(num_tests, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);

        std::tie(use_device_profiler, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--use-device-profiler");

        std::tie(bypass_check, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

        test_args::validate_remaining_args(input_args);
    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Command line arguments found exception", e.what());
    }

    if (num_tiles > 512) {
        log_error(
            LogTest,
            "Number of tiles to transfer ({}) must be less than 512 to fit in L1.",
            num_tiles
        );
    }

    if (use_device_profiler) {
#if !defined(TRACY_ENABLE)
        log_error(
            LogTest,
            "Metal library and test code should be build with "
            "profiler option using ./scripts/build_scripts/build_with_profiler_opt.sh");
#endif
        auto device_profiler = getenv("TT_METAL_DEVICE_PROFILER");
        TT_FATAL(
            device_profiler,
            "Before running the program, do one of the following in a shell: "
            "either export the environment variable by executing export TT_METAL_DEVICE_PROFILER=1, "
            "or run the program with TT_METAL_DEVICE_PROFILER=1 prefixed to the command");
    }

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device* device = tt_metal::CreateDevice(device_id);

        int clock_freq_mhz = get_tt_npu_clock(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord src_core = {src_core_x, src_core_y};
        CoreCoord target_core = {target_core_x, target_core_y};
        auto target_core_noc = device->worker_core_from_logical_core(target_core);
        CoreRangeSet all_cores({CoreRange(src_core), CoreRange(target_core)});

        // Circular buffer setup
        uint32_t cb_tiles = num_tiles;
        uint32_t single_tile_size = 2 * 1024; // float16

        uint32_t cb_src0_index = 0;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(cb_tiles * single_tile_size, {{cb_src0_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_src0_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

        // Semaphore setup
        auto noc_sem_id = tt_metal::CreateSemaphore(program, all_cores, 0);

        // Source kernel setup
        auto src_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/"
            "10_noc_unicast/kernels/src.cpp",
            src_core,
            tt_metal::DataMovementConfig{
                .processor = (noc_index == 0) ? tt_metal::DataMovementProcessor::RISCV_0
                                              : tt_metal::DataMovementProcessor::RISCV_1,
                .noc = (noc_index == 0) ? tt_metal::NOC::RISCV_0_default : tt_metal::NOC::RISCV_1_default});

        std::vector<uint32_t> src_kernel_runtime_args = {
            target_core_noc.x,
            target_core_noc.y,
            noc_sem_id,
            num_tiles,
            num_transfers,
        };
        tt_metal::SetRuntimeArgs(program, src_kernel, src_core, src_kernel_runtime_args);


        // Target kernel setup
        auto target_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/"
            "10_noc_unicast/kernels/target.cpp",
            target_core,
            tt_metal::DataMovementConfig{
                .processor = (noc_index == 0) ? tt_metal::DataMovementProcessor::RISCV_0
                                              : tt_metal::DataMovementProcessor::RISCV_1,
                .noc = (noc_index == 0) ? tt_metal::NOC::RISCV_0_default : tt_metal::NOC::RISCV_1_default});

        std::vector<uint32_t> target_kernel_runtime_args = {
            noc_sem_id,
            num_transfers,
        };
        tt_metal::SetRuntimeArgs(program, target_kernel, target_core, target_kernel_runtime_args);


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::detail::CompileProgram(device, program);

        log_info(LogTest, "Num tests {}", num_tests);
        for (uint32_t i = 0; i < num_tests; ++i) {
            auto t_begin = std::chrono::steady_clock::now();
            EnqueueProgram(device->command_queue(), program, false);
            Finish(device->command_queue());
            auto t_end = std::chrono::steady_clock::now();
            unsigned long elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            unsigned long elapsed_cc = clock_freq_mhz * elapsed_us;

            if (use_device_profiler) {
                elapsed_cc = get_t0_to_any_riscfw_end_cycle(device, program);
                elapsed_us = (double)elapsed_cc / clock_freq_mhz;
                log_info(LogTest, "Time elapsed using device profiler: {}us ({}cycles)", elapsed_us, elapsed_cc);
            }

            log_info(LogTest, "Time elapsed for {} tile transfers on NOC {}: {}us ({}cycles)", num_tiles * num_transfers, noc_index, elapsed_us, elapsed_cc);

            // total transfer amount per core = tile size * number of tiles * number of transfers
            // NOC bandwidth = total transfer amount per core / elapsed clock cycle
            measured_bandwidth.push_back((double)single_tile_size * num_tiles * num_transfers / elapsed_cc);

            log_info(LogTest, "Measured NOC bandwidth: {:.3f}B/cc", measured_bandwidth[i]);
        }

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    auto avg_measured_bandwidth = calculate_average(measured_bandwidth);
    if (pass && bypass_check == false) {
        // goal is 85% of theoretical peak using a single NOC channel
        // theoretical peak: 32bytes per clock cycle
        double target_bandwidth = 32.0 * 0.85;
        if (avg_measured_bandwidth < target_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The NOC bandwidth does not meet the criteria. "
                "Current: {:.3f}B/cc, goal: >={:.3f}B/cc",
                avg_measured_bandwidth,
                target_bandwidth);
        }
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
