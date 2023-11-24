// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "common/constants.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util_device_profiler.hpp"
#include "tt_numpy/functions.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;

Tensor diagonal(const Shape &shape, float value) {
    Tensor tensor = tt::numpy::zeros(shape);
    auto buffer = owned_buffer::get_as<bfloat16>(tensor);
    for (int i = 0; i < shape[0] * shape[1]; ++i) {
        for (int j = 0; j < std::min(shape[2], shape[3]); j++) {
            buffer[i * shape[2] * shape[3] + j * shape[3] + j] = bfloat16(value);
        }
    }
    return tensor;
}

static bool nearly_equal(float a, float b, float epsilon = 1e-5f, float abs_threshold = 1e-5f) {
    auto diff = std::abs(a - b);
    auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
    auto result = diff < std::max(abs_threshold, epsilon * norm);
    return result;
}

template <typename... Args>
static bool nearly_equal(bfloat16 a, bfloat16 b, Args... args) {
    return nearly_equal(a.to_float(), b.to_float(), args...);
}

inline bool compare(
    const Tensor &tensor_a,
    const Tensor &tensor_b,
    int B1,
    int B2,
    int M,
    int N,
    int K,
    int in0_B1,
    int in0_B2,
    bool print = false) {
    auto tensor_a_buffer = owned_buffer::get_as<bfloat16>(tensor_a);
    auto tensor_b_buffer = owned_buffer::get_as<bfloat16>(tensor_b);

    // debug print
    int print_cnt = 0;
    int print_cnt2 = 0;
    int count = 0;
    int print_limit = 10;

    int MN = M * N;
    int B2MN = B2 * MN;

    int MK = M * K;
    int in0_B2MK = in0_B2 * MK;

    for (int b1 = 0; b1 < B1; ++b1) {
        for (int b2 = 0; b2 < B2; ++b2) {
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    int a_b1 = (b1 >= in0_B1) ? (0) : (b1);
                    int a_b2 = (b2 >= in0_B2) ? (0) : (b2);

                    int a_index = a_b1 * in0_B2MK + a_b2 * MK + m * K + n;
                    int b_index = b1 * B2MN + b2 * MN + m * N + n;

                    if (n >= K) {
                        if (tensor_b_buffer[b_index] != 0) {
                            count++;
                            if (print && print_cnt++ < print_limit) {
                                log_error(
                                    LogTest,
                                    "(b1, b2, m, n) = ({}, {}, {}, {}), output {} should be zero.",
                                    b1,
                                    b2,
                                    m,
                                    n,
                                    tensor_b_buffer[b_index]);
                            }
                        }
                        continue;
                    }
                    if (not nearly_equal(tensor_a_buffer[a_index], tensor_b_buffer[b_index])) {
                        count++;
                        if (print && print_cnt2++ < print_limit) {
                            log_error(
                                LogTest,
                                "(b1, b2, m, n) = ({}, {}, {}, {}), activation {} != output {}",
                                b1,
                                b2,
                                m,
                                n,
                                tensor_a_buffer[a_index],
                                tensor_b_buffer[b_index]);
                        }
                    }
                }
            }
        }
    }

    if (count) {
        if (print) {
            log_error(LogTest, "{} diffs", count);
        }
        return false;
    }
    return true;
}

tuple<uint32_t, uint32_t, uint32_t> get_aligned_input_tile_num(uint32_t M, uint32_t N, uint32_t K) {
    auto align_to_tile = [](uint32_t value) -> uint32_t {
        return ((value + (constants::TILE_WIDTH - 1)) / constants::TILE_WIDTH) * constants::TILE_WIDTH;
    };

    TT_ASSERT(M != 0 && N != 0 && K != 0, "Matmul input size should not be zero");

    uint32_t M_aligned = align_to_tile(M);
    uint32_t N_aligned = align_to_tile(N);
    uint32_t K_aligned = align_to_tile(K);

    if (M % constants::TILE_WIDTH || N % constants::TILE_WIDTH || K % constants::TILE_WIDTH)
        log_info(LogTest, "M, N, K = {}, {}, {} are aligned to {}, {}, {}", M, N, K, M_aligned, N_aligned, K_aligned);

    uint32_t Mt = M_aligned / constants::TILE_WIDTH;
    uint32_t Nt = N_aligned / constants::TILE_WIDTH;
    uint32_t Kt = K_aligned / constants::TILE_WIDTH;
    return {Mt, Nt, Kt};
}

double get_tt_npu_rpeak_tflops(tt::ARCH arch, CoreCoord grid_size, int tt_npu_clock, bool bfp8_format) {
    constexpr double WH_FPU_BFP8_TFLOPS_PER_TENSIX = 2.05;
    constexpr double WH_FPU_BF16_TFLOPS_PER_TENSIX = 1.02;
    constexpr double GS_FPU_BFP8_TFLOPS_PER_TENSIX = 0.58;
    constexpr double GS_FPU_BF16_TFLOPS_PER_TENSIX = 0.58;

    double rpeak_tflops = 0.0f;
    double clock = static_cast<double>(tt_npu_clock) / 1000;
    uint32_t num_compute_core = grid_size.x * grid_size.y;
    if (arch == tt::ARCH::WORMHOLE || arch == tt::ARCH::WORMHOLE_B0) {

        double tflops_per_tensix = (bfp8_format) ? (WH_FPU_BFP8_TFLOPS_PER_TENSIX) : (WH_FPU_BF16_TFLOPS_PER_TENSIX);
        rpeak_tflops =
            tflops_per_tensix * static_cast<double>(num_compute_core) * static_cast<double>(clock);
    } else if (arch == tt::ARCH::GRAYSKULL) {
        double tflops_per_tensix = (bfp8_format) ? (GS_FPU_BFP8_TFLOPS_PER_TENSIX) : (GS_FPU_BF16_TFLOPS_PER_TENSIX);
        rpeak_tflops =
            GS_FPU_BFP8_TFLOPS_PER_TENSIX * static_cast<double>(num_compute_core) * static_cast<double>(clock);
    }

    log_debug(LogTest, "Rpeak {} TFLOPS", rpeak_tflops);
    return rpeak_tflops;
}

double calculateAverage(const std::vector<double> &vec, bool skip_first_run) {
    if (vec.empty()) {
        return 0.0;  // 벡터가 비어있을 경우 0을 반환
    }

    int index = (skip_first_run) ? (1) : (0);
    double sum = std::accumulate(vec.begin() + index, vec.end(), 0.0);
    double average = sum / (vec.size() - index);
    return average;
}

static OwnedBuffer create_owned_buffer_from_vector_of_floats(std::vector<float> &&data, DataType data_type) {
    switch (data_type) {
        case DataType::BFLOAT8_B: {
            auto uint32_vector = pack_fp32_vec_as_bfp8_tiles(data, /*row_major_input=*/false, /*is_exp_a=*/false);
            return owned_buffer::create<uint32_t>(std::move(uint32_vector));
        }
        case DataType::FLOAT32: {
            return owned_buffer::create<float>(std::move(data));
        }
        case DataType::BFLOAT16: {
            std::vector<bfloat16> bfloat16_data(data.size());
            std::transform(std::begin(data), std::end(data), std::begin(bfloat16_data), [](float value) {
                return bfloat16(value);
            });
            return owned_buffer::create<bfloat16>(std::move(bfloat16_data));
        }
        default: {
            TT_THROW("Cannot create a host buffer!");
        }
    }
}

std::vector<float> generate_fp32_random(uint32_t num_elems, int32_t rand_max_val = 100) {
    std::vector<float> vec(num_elems);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_val), std::mt19937(seed));
    for (uint32_t i = 0; i < num_elems; ++i) {
        vec.at(i) = static_cast<float>(rand_float());
    }
    return vec;
}

Tensor get_bfp8_tensor(std::vector<float> data, const Shape &shape, DataType data_type, Layout layout) {
    auto owned_buffer = create_owned_buffer_from_vector_of_floats(std::move(data), data_type);
    return Tensor(OwnedStorage{owned_buffer}, shape, data_type, layout);
}

Tensor get_tensor(tt_metal::Device *device, const Shape &shape, bool bfp8_format) {
    if (bfp8_format) {
        auto data = generate_fp32_random(shape[0] * shape[1] * shape[2] * shape[3]);
        return get_bfp8_tensor(data, shape, DataType::BFLOAT8_B, Layout::TILE).to(device);
    } else {
        return tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    }
}

int main(int argc, char **argv) {
    bool pass = true;

#if !defined(PROFILER)
    log_error(
        "In the slow dispatch mode, device profiler is used to measure the "
        "performance. Build the Metal library and test code with "
        "'ENABLE_PROFILER=1'");
    TT_ASSERT(false);
#endif

    double rmax = 0.0f;
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        uint32_t M;
        uint32_t N;
        uint32_t K;
        bool l1_output = false;
        bool skip_first_run = false;
        bool bfp8_format = false;
        uint32_t num_count = 10;
        std::tie(M, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--m", 11264);
        std::tie(N, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--n", 3072);
        std::tie(K, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--k", 768);
        std::tie(num_count, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-count", 10);
        std::tie(l1_output, input_args) = test_args::has_command_option_and_remaining_args(input_args, "--l1-output");
        std::tie(skip_first_run, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--skip-first-run");
        std::tie(bfp8_format, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--bfp8-format");
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [Mt, Nt, Kt] = get_aligned_input_tile_num(M, N, K);
        log_info(LogTest, "Input M, N, K = {}, {}, {} / {}, {}, {} tile(s)", M, N, K, Mt, Nt, Kt);
        Shape shapea = {1, 1, Mt * 32, Kt * 32};
        Shape shapeb = {1, 1, Kt * 32, Nt * 32};

        auto L1_OUTPUT_MEMORY_CONFIG =
            MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1};
        auto mem_config = (l1_output) ? (L1_OUTPUT_MEMORY_CONFIG) : (operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

        auto a = get_tensor(device, shapea, bfp8_format);
        Tensor b = (bfp8_format) ? (get_tensor(device, shapeb, bfp8_format))
                                 : (diagonal(shapeb, 1.0f).to(Layout::TILE).to(device));

        ////////////////////////////////////////////////////////////////////////////
        //                      Run
        ////////////////////////////////////////////////////////////////////////////
        constexpr int giga_byte = 1000000;
        constexpr long long tera_byte = 1000000000000LL;
        const tt::ARCH arch = device->arch();
        int tt_npu_clock = get_tt_npu_clock(device);
        auto grid_size = device->compute_with_storage_grid_size();
        double rpeak_tflops = get_tt_npu_rpeak_tflops(arch, grid_size, tt_npu_clock, bfp8_format);
        std::vector<double> rmax_tflops;
        uint64_t num_of_matmul_ops =
            (2 * static_cast<uint64_t>(Kt) * 32 - 1) * (static_cast<uint64_t>(Mt) * static_cast<uint64_t>(Nt) * 1024);

        // Create temp tensor
        Tensor out = diagonal(shapeb, 1.0f);
        for (int i = 0; i < num_count; ++i) {
            out = matmul(a, b, mem_config);
        }

        auto t0_to_any_riscfw_end = operation::get_cycles();
        for (int i = 0; i < num_count; ++i) {
            double cycle_time = 1 / static_cast<double>(tt_npu_clock) / giga_byte;
            auto execution_time = t0_to_any_riscfw_end[i] * cycle_time;
            rmax_tflops.push_back(static_cast<double>(num_of_matmul_ops) / execution_time / tera_byte);
            // log_info(LogTest, "cycle time {:.10f}s", cycle_time);
            // log_info(LogTest, "t0_to_any_riscfw_end {}", t0_to_any_riscfw_end);
        }

        log_info(LogTest, "rmax_tflops {}", rmax_tflops);
        rmax = calculateAverage(rmax_tflops, skip_first_run);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto out_cpu = out.cpu();
        const auto &out_shape = out_cpu.shape();
        log_info(
            LogTest,
            "out_shape {} - {}, {}, {}, {}",
            out_shape.rank(),
            out_shape[0],
            out_shape[1],
            out_shape[2],
            out_shape[3]);

        if (!bfp8_format) {
            pass &= compare(
                a.cpu().to(Layout::ROW_MAJOR),
                out_cpu.to(Layout::ROW_MAJOR),
                out_shape[0],
                out_shape[1],
                out_shape[2],
                out_shape[3],
                shapea[3],
                shapea[0],
                shapea[1],
                true);
        }

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    log_info(LogTest, "rmax: {}", rmax);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
