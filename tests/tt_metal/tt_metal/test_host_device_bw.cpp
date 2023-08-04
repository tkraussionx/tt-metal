#include <random>
#include <vector>
#include <chrono>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

#include "constants.hpp"

using namespace tt;
using namespace constants;

using namespace std::chrono;

int main(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }

    bool pass = true;
    bool disable_multi_bank = false;

    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    int pci_express_slot = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(arch, pci_express_slot);

    try {
        pass &= tt_metal::InitializeDevice(device);;

        // tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});

        uint32_t dram_buffer_size = 1024 * 1024;    // 1M elems
        uint64_t dram_buffer_bytes = dram_buffer_size * 2;  // 2 bytes for bf16
        uint64_t page_size = 32 * 32 * 2;   // one tile
        if (disable_multi_bank) {
            page_size = dram_buffer_bytes;
        }
        uint64_t dram_buffer_addr = 0;
        auto dram_buffer = tt_metal::Buffer(device, dram_buffer_bytes, dram_buffer_addr, page_size, tt_metal::BufferType::DRAM);
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 100, 0);
        std::vector<uint32_t> result_vec;

        int loop_count = 100;

        double total = 0.;
        for (int i = 0; i < loop_count; ++i) {
            auto start = high_resolution_clock::now();

            WriteToBuffer(dram_buffer, input_vec);

            ReadFromBuffer(dram_buffer, result_vec);

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            total += (double) duration.count();
        }
        double duration_per_loop = total / loop_count;
        std::cout << "Duration: " << duration_per_loop << " usecs" << std::endl;
        std::cout << "Write & read bandwidth: " << (dram_buffer_bytes / duration_per_loop) * 1e6 / (1024 * 1024) << " MB/sec" << std::endl;

        total = 0.;
        for (int i = 0; i < loop_count; ++i) {
            auto start = high_resolution_clock::now();

            WriteToBuffer(dram_buffer, input_vec);

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            total += (double) duration.count();
        }
        duration_per_loop = total / loop_count;
        std::cout << "Duration: " << duration_per_loop << " usecs" << std::endl;
        std::cout << "Write bandwidth: " << (dram_buffer_bytes / duration_per_loop) * 1e6 / (1024 * 1024) << " MB/sec" << std::endl;

        total = 0.;
        for (int i = 0; i < loop_count; ++i) {
            auto start = high_resolution_clock::now();

            ReadFromBuffer(dram_buffer, result_vec);

            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            total += (double) duration.count();
        }
        duration_per_loop = total / loop_count;
        std::cout << "Duration: " << duration_per_loop << " usecs" << std::endl;
        std::cout << "Read bandwidth: " << (dram_buffer_bytes / duration_per_loop) * 1e6 / (1024 * 1024) << " MB/sec" << std::endl;

        int argfail = -1;
        auto comparison_function = [](float a, float b) {
            const float rtol = 0.02f;
            const float atol = 1e-3f;
            float maxabs = fmaxf(fabsf(a), fabsf(b));
            float absdiff = fabsf(a - b);
            auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
            return result;
        };
        pass &= packed_uint32_t_vector_comparison(input_vec, result_vec, comparison_function, &argfail);
        if (!pass)
            log_error(LogTest, "Failure position={}", argfail);

        pass &= tt_metal::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
        tt_metal::CloseDevice(device);;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
