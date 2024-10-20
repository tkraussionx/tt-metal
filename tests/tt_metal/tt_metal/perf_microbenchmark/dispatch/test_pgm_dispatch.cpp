// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 100;
constexpr uint32_t MIN_KERNEL_SIZE_BYTES = 32;  // overhead
constexpr uint32_t DEFAULT_KERNEL_SIZE_K = 1;
constexpr uint32_t MAX_CBS = 32;
constexpr uint32_t MAX_ARGS = 255;

//////////////////////////////////////////////////////////////////////////////////////////
// Test dispatch program performance
//
// Times dispatching program to M cores, N processors, of various sizes, CBs, runtime args
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

uint32_t iterations_g = DEFAULT_ITERATIONS;
uint32_t warmup_iterations_g = DEFAULT_WARMUP_ITERATIONS;
CoreRange workers_g = {{0, 0}, {0, 0}};
uint32_t kernel_size_g;
uint32_t fast_kernel_cycles_g;
uint32_t slow_kernel_cycles_g;
uint32_t nfast_kernels_g;
uint32_t n_cbs_g;
uint32_t n_args_g;
uint32_t n_common_args_g;
uint32_t n_sems_g;
uint32_t n_kgs_g;
bool brisc_enabled_g;
bool ncrisc_enabled_g;
bool trisc_enabled_g;
bool erisc_enabled_g;
uint32_t erisc_count_g;
bool lazy_g;
bool time_just_finish_g;
bool use_global_g;
bool use_trace_g;

void init(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  -w: warm-up iterations before starting timer (default {}), ", DEFAULT_WARMUP_ITERATIONS);
        log_info(LogTest, "  -i: iterations (default {})", DEFAULT_ITERATIONS);
        log_info(LogTest, "  -s: size of kernels in powers of 2 bytes (default {}, min {})", DEFAULT_KERNEL_SIZE_K * 1024, MIN_KERNEL_SIZE_BYTES);
        log_info(LogTest, "  -x: X end of inclusive core range (default {})", 0);
        log_info(LogTest, "  -y: Y end of inclusive core range (default {})", 0);
        log_info(LogTest, "  -c: number of CBs (default {}, max {})", 0, MAX_CBS);
        log_info(LogTest, "  -a: number of runtime args (default {}, max {})", 0, MAX_ARGS);
        log_info(LogTest, "  -ca: number of common runtime args multicast to all cores (default {}, max {})", 0, MAX_ARGS);
        log_info(LogTest, "  -S: number of semaphores (default {}, max {})", 0, NUM_SEMAPHORES);
        log_info(LogTest, " -kg: number of kernel groups (default 1)");
        log_info(LogTest, "  -g: use a 4 byte global variable (additional spans");
        log_info(LogTest, "  -rs:run \"slow\" kernels for exactly <n> cycles (default 0)");
        log_info(LogTest, "  -rf:run \"fast\" kernels for exactly <n> cycles (default 0)");
        log_info(LogTest, "  -nf:run <n> fast kernels between slow kernels (default 0)");
        log_info(LogTest, "  -b: disable brisc kernel (default enabled)");
        log_info(LogTest, "  -n: disable ncrisc kernel (default enabled)");
        log_info(LogTest, "  -t: disable trisc kernels (default enabled)");
        log_info(LogTest, "  -t: disable trisc kernels (default enabled)");
        log_info(LogTest, "  +e: enable erisc kernels (default disabled)");
        log_info(LogTest, " -ec: erisc count (default 1 if enabled)");
        log_info(LogTest, "  -f: time just the finish call (use w/ lazy mode) (default disabled)");
        log_info(LogTest, "  -z: enable dispatch lazy mode (default disabled)");
        log_info(LogTest, "  -tr: enable trace (default disabled)");
        exit(0);
    }

    uint32_t core_x = test_args::get_command_option_uint32(input_args, "-x", 0);
    uint32_t core_y = test_args::get_command_option_uint32(input_args, "-y", 0);
    warmup_iterations_g = test_args::get_command_option_uint32(input_args, "-w", DEFAULT_WARMUP_ITERATIONS);
    iterations_g = test_args::get_command_option_uint32(input_args, "-i", DEFAULT_ITERATIONS);
    kernel_size_g = test_args::get_command_option_uint32(input_args, "-s", DEFAULT_KERNEL_SIZE_K * 1024);
    n_cbs_g = test_args::get_command_option_uint32(input_args, "-c", 0);
    n_args_g = test_args::get_command_option_uint32(input_args, "-a", 0);
    n_common_args_g = test_args::get_command_option_uint32(input_args, "-ca", 0);
    n_sems_g = test_args::get_command_option_uint32(input_args, "-S", 0);
    n_kgs_g = test_args::get_command_option_uint32(input_args, "-kg", 1);
    lazy_g = test_args::has_command_option(input_args, "-z");
    use_global_g = test_args::has_command_option(input_args, "-g");
    time_just_finish_g = test_args::has_command_option(input_args, "-f");
    fast_kernel_cycles_g = test_args::get_command_option_uint32(input_args, "-rf", 0);
    slow_kernel_cycles_g = test_args::get_command_option_uint32(input_args, "-rs", 0);
    nfast_kernels_g = test_args::get_command_option_uint32(input_args, "-nf", 0);
    use_trace_g = test_args::has_command_option(input_args, "-tr");
    if (kernel_size_g < MIN_KERNEL_SIZE_BYTES) {
        log_fatal("Minimum kernel size is {} bytes", MIN_KERNEL_SIZE_BYTES);
        exit(0);
    }
    if (n_cbs_g > MAX_CBS) {
        log_fatal("CB count must be 0..{}", MAX_CBS);
        exit(0);
    }
    if (n_args_g > MAX_ARGS) {
        log_fatal("Runtime arg count must be 0..{}", MAX_ARGS);
        exit(0);
    }
    if (n_common_args_g > MAX_ARGS) {
        log_fatal("Common Runtime arg count must be 0..{}", MAX_ARGS);
        exit(0);
    }
    if (n_sems_g > NUM_SEMAPHORES) {
        log_fatal("Sem count must be 0..{}", NUM_SEMAPHORES);
        exit(0);
    }
    if (n_kgs_g > core_x + 1) {
        log_fatal("This test uses columns for kernel groups so number of kernel groups must be <= x core range");
        exit(0);
    }
    brisc_enabled_g = !test_args::has_command_option(input_args, "-b");
    ncrisc_enabled_g = !test_args::has_command_option(input_args, "-n");
    trisc_enabled_g = !test_args::has_command_option(input_args, "-t");
    erisc_enabled_g = test_args::has_command_option(input_args, "+e");
    erisc_count_g = test_args::get_command_option_uint32(input_args, "-ec", 1);

    workers_g = CoreRange({0, 0}, {core_x, core_y});

    if (nfast_kernels_g != 0 && slow_kernel_cycles_g <= fast_kernel_cycles_g) {
        log_error("The number of fast kernels is non-zero, but slow_kernel_ cycles ({}) is <= fast_kernel_cycles ({})",
                  slow_kernel_cycles_g, fast_kernel_cycles_g );
        log_error("For meaningful results, run multiple fast kernels between single slow kernels");
        exit(0);
    }
}

void set_runtime_args(tt_metal::Program& program, tt_metal::KernelHandle kernel_id, vector<uint32_t>& args, CoreRange kg) {
    for (int core_idx_y = kg.start_coord.y; core_idx_y <= kg.end_coord.y; core_idx_y++) {
        for (int core_idx_x = kg.start_coord.x; core_idx_x <= kg.end_coord.x; core_idx_x++) {
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
            tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
        }
    }
}

void initialize_program(Device *device, tt_metal::Program& program, uint32_t run_cycles) {

    program = tt_metal::CreateProgram();

    std::map<string, string> defines = {
        {"KERNEL_BYTES", std::to_string(kernel_size_g)}
    };
    if (run_cycles != 0) {
        defines.insert(std::pair<string, string>("KERNEL_RUN_TIME", std::to_string(run_cycles)));
    }
    if (use_global_g) {
        defines.insert(std::pair<string, string>("KERNEL_GLOBAL", "1"));
    }

    for (uint32_t i = 0; i < n_sems_g; i++) {
        tt_metal::CreateSemaphore(program, workers_g, 3);
    }

    vector<uint32_t> args;
    args.resize(n_args_g);
    vector<uint32_t> common_args;
    common_args.resize(n_common_args_g);

    for (int i = 0; i < n_cbs_g; i++) {
        tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(16, {{i, tt::DataFormat::Float16_b}})
            .set_page_size(i, 16);
        auto cb = tt_metal::CreateCircularBuffer(program, workers_g, cb_config);
    }

    // first kernel group is possibly wide, remaining kernel groups are 1 column each
    CoreRange kg = { workers_g.start_coord, { workers_g.end_coord.x - n_kgs_g + 1, workers_g.end_coord.y }};
    for (uint32_t i = 0; i < n_kgs_g; i++) {
        defines.insert(std::pair<string, string>(string("KG_") + std::to_string(i), ""));

        if (brisc_enabled_g) {
            auto dm0 = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                kg,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .defines = defines});
            set_runtime_args(program, dm0, args, kg);
            tt_metal::SetCommonRuntimeArgs(program, dm0, common_args);
        }

        if (ncrisc_enabled_g) {
            auto dm1 = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                kg,
                tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .defines = defines});
            set_runtime_args(program, dm1, args, kg);
            tt_metal::SetCommonRuntimeArgs(program, dm1, common_args);
        }

        if (trisc_enabled_g) {
            auto compute = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                kg,
                tt_metal::ComputeConfig{.defines = defines});
            set_runtime_args(program, compute, args, kg);
            tt_metal::SetCommonRuntimeArgs(program, compute, common_args);
        }

        kg.start_coord = { kg.end_coord.x + 1, kg.end_coord.y };
        kg.end_coord = kg.start_coord;
    }

    if (erisc_enabled_g) {
        auto erisc_cores = device->get_active_ethernet_cores();
        if (erisc_count_g > erisc_cores.size()) {
            log_fatal("Requested number of erisc cores {} exceeds actual erisc core count {}", erisc_count_g, erisc_cores.size());
            exit(0);
        }
        auto erisc_core = erisc_cores.begin();
        for (uint32_t i = 0; i < erisc_count_g; i++, erisc_core++) {
            auto eth_kernel = CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
                *erisc_core,
                tt::tt_metal::EthernetConfig {
                    .eth_mode = Eth::RECEIVER,
                    .noc = NOC::NOC_0,
                    .defines = defines,
                });
            tt_metal::SetRuntimeArgs(program, eth_kernel, *erisc_core, args);
            tt_metal::SetCommonRuntimeArgs(program, eth_kernel, common_args);
        }
    }

}

int main(int argc, char **argv) {
    init(argc, argv);

    tt::llrt::OptionsG.set_kernels_nullified(true);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device* device;
        if (use_trace_g) {
            device = tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, 900000000);
        } else {
            device = tt_metal::CreateDevice(device_id);
        }

        CommandQueue& cq = device->command_queue();

        tt_metal::Program program[2];
        initialize_program(device, program[0], slow_kernel_cycles_g);
        initialize_program(device, program[1], fast_kernel_cycles_g);

        // Cache stuff
        for (int i = 0; i < warmup_iterations_g; i++) {
            EnqueueProgram(cq, program[0], false);
            if (nfast_kernels_g > 0) {
                EnqueueProgram(cq, program[1], false);
            }
        }

        auto main_program_loop = [&]() {
            for (int i = 0; i < iterations_g; i++) {
                EnqueueProgram(cq, program[0], false);
                if (nfast_kernels_g > 0) {
                    EnqueueProgram(cq, program[1], false);
                }
            }
        };
        uint32_t tid = 0;
        if (use_trace_g) {
            tid = BeginTraceCapture(device, cq.id());
            main_program_loop();
            EndTraceCapture(device, cq.id(), tid);
            Finish(cq);
        }

        if (lazy_g) {
            // Does this do anything?
            tt_metal::detail::SetLazyCommandQueueMode(true);
        }

        auto start = std::chrono::system_clock::now();
        if (use_trace_g) {
            EnqueueTrace(cq, tid, false);
        } else {
            main_program_loop();
        }
        if (time_just_finish_g) {
            start = std::chrono::system_clock::now();
        }
        Finish(cq);
        auto end = std::chrono::system_clock::now();

        if (use_trace_g) {
            log_info(LogTest, "Running with trace enabled");
        }
        log_info(LogTest, "Warmup iterations: {}", warmup_iterations_g);
        log_info(LogTest, "Iterations: {}", iterations_g);
        log_info(LogTest, "Grid: ({}-{}) ({} cores)", workers_g.start_coord.str(), workers_g.end_coord.str(), workers_g.size());
        log_info(LogTest, "Kernel size: {}", kernel_size_g);
        if (nfast_kernels_g != 0) {
            log_info(LogTest, "Fast kernel cycles: {}", fast_kernel_cycles_g);
            log_info(LogTest, "Slow kernel cycles: {}", slow_kernel_cycles_g);
            log_info(LogTest, "{} fast kernels between slow kernels", nfast_kernels_g);
        } else {
            log_info(LogTest, "Kernel cycles: {}", slow_kernel_cycles_g);
        }
        log_info(LogTest, "KGs: {}", n_kgs_g);
        log_info(LogTest, "CBs: {}", n_cbs_g);
        log_info(LogTest, "UniqueRTArgs: {}", n_args_g);
        log_info(LogTest, "CommonRTArgs: {}", n_common_args_g);
        log_info(LogTest, "Sems: {}", n_sems_g);
        log_info(LogTest, "Lazy: {}", lazy_g);

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
        log_info(LogTest, "Ran in {}us per iteration", elapsed_seconds.count() * 1000 * 1000 / iterations_g);

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::llrt::OptionsG.set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
