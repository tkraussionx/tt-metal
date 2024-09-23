// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

CoreRangeSet vector_to_core_range_set(const vector<CoreCoord>& vec)
{
    CoreRangeSet crs({CoreRange{vec[0], vec[0]}});
    for (uint32_t i = 1; i < vec.size(); i++)
    {
        crs = crs.merge(CoreRangeSet({CoreRange{vec[i], vec[i]}}));
    }
    return crs;
}

uint32_t run_atomic_storm(Device *device, vector<vector<uint32_t>> &output, vector<vector<uint32_t>> &output_debug, uint32_t semaphore_count, uint32_t inc_count, uint32_t receiver_count)
{
    /*
    * Setup program to execute along with its buffers and kernels to use
    */
    CommandQueue& cq = device->command_queue();
    Program program{};

    /*
    * Multi-Core prep
    */
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Output buffers
    tt_metal::InterleavedBufferConfig dram_config_out{
                    .device= device,
                    .size = 4 * 32 * 32,
                    .page_size = 4 * 32 * 32,
                    .buffer_type = tt_metal::BufferType::DRAM
        };
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> dst_dram_buffers;
    for (uint32_t i=0; i < receiver_count; i++)
    {
        dst_dram_buffers.push_back(CreateBuffer(dram_config_out));
    }

    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> debug_dram_buffers;
    for (uint32_t i=0; i < (num_cores_x * num_cores_y) - receiver_count; i++)
    {
        debug_dram_buffers.push_back(CreateBuffer(dram_config_out));
    }
    //uint32_t dst_addr = dst_dram_buffer->address();
    //uint32_t debug_addr = debug_dram_buffer->address();

    vector<CoreCoord> sender_cores = grid_to_cores( {0, 0}, {num_cores_x-1, num_cores_y-1} );
    vector<CoreCoord> receiver_cores;
    for (uint32_t i = 0; i < receiver_count; i++)
    {
        // Pick a random core from sender_cores, remove from sender_cores and put it into receiver_cores
        uint32_t random_index = rand() % sender_cores.size();
        receiver_cores.push_back(sender_cores[random_index]);
        sender_cores.erase(sender_cores.begin() + random_index);
    }

    CoreRangeSet receivers = vector_to_core_range_set(receiver_cores);
    vector<CoreCoord> receiver_cores_physical;
    for (uint32_t i = 0; i < receiver_count; i++)
    {
        receiver_cores_physical.push_back(device->worker_core_from_logical_core(receiver_cores[i]));
    }
    auto receiver_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/atomic_receiver.cpp",
        receivers,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {}});

    CoreRangeSet senders = vector_to_core_range_set(sender_cores);
    auto sender_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/atomic_inc.cpp",
        senders,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {}});

    vector<uint32_t> semaphore_ids;
    for (uint32_t i = 0; i < semaphore_count; i++)
    {
        semaphore_ids.push_back(tt_metal::CreateSemaphore(program, receivers, 0));
    }
    uint32_t final_expected_count = inc_count * senders.num_cores();
    std::cout << "senders: " << senders.num_cores() << ", inc_count: " << inc_count << ", final semaphore expected: " << final_expected_count << ", semaphore_count = " << semaphore_count << std::endl;
    vector<uint32_t> receiver_args = {num_cores_x * num_cores_y - 1, inc_count, semaphore_count};
    vector<uint32_t> sender_args = {inc_count, semaphore_count, receiver_count};
    for (uint32_t i = 0; i < semaphore_count; i++)
    {
        receiver_args.push_back(semaphore_ids[i]);
        sender_args.push_back(semaphore_ids[i]);
    }
    for (uint32_t i = 0; i < receiver_count; i++)
    {
        sender_args.push_back(receiver_cores_physical[i].x);
        sender_args.push_back(receiver_cores_physical[i].y);
    }

    for (uint32_t i = 0; i < receiver_count; i++)
    {
        vector<uint32_t> my_receiver_args = receiver_args;
        my_receiver_args.push_back(dst_dram_buffers[i]->address());
        tt_metal::SetRuntimeArgs(program, receiver_id, receiver_cores[i], my_receiver_args);
    }

    srand(time(NULL));
    for (uint32_t i = 0; i < sender_cores.size(); i++)
    {
        vector<uint32_t> my_sender_args = sender_args;
        my_sender_args.push_back(debug_dram_buffers[i]->address());
        my_sender_args.push_back(i);
        my_sender_args.push_back(rand());
        tt_metal::SetRuntimeArgs(program, sender_id, sender_cores[i], my_sender_args);
    }

    EnqueueProgram(cq, program, false);

    for (uint32_t i = 0; i < sender_cores.size(); i++)
    {
        EnqueueReadBuffer(cq, debug_dram_buffers[i], output_debug[i].data(), true);
    }
    for (uint32_t i = 0; i < receiver_count; i++)
    {
        EnqueueReadBuffer(cq, dst_dram_buffers[i], output[i].data(), true);
    }
    return final_expected_count;
}

int main(int argc, char **argv) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);

        uint32_t semaphore_count = 1;
        uint32_t inc_count = 1000;
        uint32_t receiver_count = 1;
        uint32_t iteration_count = 1;
        if (argc >= 2)
        {
            std::string inc_count_str = argv[1];
            inc_count = std::stoi(inc_count_str);
        }
        if (argc >= 3)
        {
            std::string semaphore_count_str = argv[2];
            semaphore_count = std::stoi(semaphore_count_str);
        }
        if (argc >= 4)
        {
            std::string receiver_count_str = argv[3];
            receiver_count = std::stoi(receiver_count_str);
        }
        if (argc >= 5)
        {
            std::string iteration_count_str = argv[4];
            iteration_count = std::stoi(iteration_count_str);
        }
        std::cout << "Test configuration" << std::endl;
        std::cout << "==================" << std::endl;
        std::cout << "inc_count: " << inc_count << std::endl;
        std::cout << "semaphore_count: " << semaphore_count << std::endl;
        std::cout << "receiver_count: " << receiver_count << std::endl;
        std::cout << "iterations: " << iteration_count << std::endl;
        std::cout << "==================" << std::endl;


        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t sender_count = num_cores_x * num_cores_y - receiver_count;

        vector<vector<uint32_t>> result_vec(receiver_count, vector<uint32_t>(32 * 32));
        vector<vector<uint32_t>> debug_vec(sender_count, vector<uint32_t>(32 * 32));

        for (uint32_t iteration=1; iteration <= iteration_count; iteration++)
        {
            std::cout << "Running iteration " << iteration << std::endl;
            uint32_t final_expected_count = run_atomic_storm(device, result_vec, debug_vec, semaphore_count, inc_count, receiver_count);
            //std::cout << "Output debug: " << debug_vec[0][0] << std::endl;
            std::cout << "Checking iteration " << iteration << std::endl;
            for (uint32_t rec=0; rec < receiver_count; rec++)
            {
                for (uint32_t i = 0; i < semaphore_count; i++)
                {
                    uint32_t index = (L1_ALIGNMENT/sizeof(uint32_t)) * i;
                    //std::cout << "Output semaphore: " << i << " (receiver " << rec << "): " << result_vec[rec][index] << std::endl;
                    TT_FATAL(final_expected_count == result_vec[rec][index], "Final sempahore count not equal to expected for receiver {}, semaphore {}. Result {}, Expected {}", rec, i, result_vec[rec][index], final_expected_count);
                }
            }
            std::cout << "Iteration " << iteration << " ok." << std::endl;
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
