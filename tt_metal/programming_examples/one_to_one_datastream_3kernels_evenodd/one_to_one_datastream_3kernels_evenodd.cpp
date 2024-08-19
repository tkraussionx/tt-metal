// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {

    // Get the directory name (e.g., SaWaRa) from the user or command-line arguments
    std::string case_name;
    if (argc > 1) {
        case_name = argv[1];
    } else {
        std::cout << "Please enter the case name (e.g., SaWaRa): ";
        std::cin >> case_name;
    }

    // Initialize Program and Device
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();

    constexpr CoreCoord core_odd = {5, 0};
    CoreCoord core_odd_physical = device->worker_core_from_logical_core(core_odd);
    uint32_t core_odd_x_cord = core_odd_physical.x;
    uint32_t core_odd_y_cord = core_odd_physical.y;

    constexpr CoreCoord core_even = {11, 8};
    CoreCoord core_even_physical = device->worker_core_from_logical_core(core_even);
    uint32_t core_even_x_cord = core_even_physical.x;
    uint32_t core_even_y_cord = core_even_physical.y;

    constexpr CoreCoord core_read = {8, 4};
    CoreCoord core_read_physical = device->worker_core_from_logical_core(core_read);
    uint32_t core_read_x_cord = core_read_physical.x;
    uint32_t core_read_y_cord = core_read_physical.y;

    int odd_loop_start = 1;
    int even_loop_start = 2;

    constexpr uint32_t single_tile_size = 4;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> collect_dram_buffer = CreateBuffer(dram_config);

    auto dram_noc_coord = collect_dram_buffer->noc_coordinates();
    uint32_t dram_noc_x = dram_noc_coord.x;
    uint32_t dram_noc_y = dram_noc_coord.y;

    // Semaphore setup
    std::vector<uint32_t> initial_int_value(1, 0);
    EnqueueWriteBuffer(cq, collect_dram_buffer, initial_int_value, false);
    uint32_t semaphore_initial_value = 0;
    uint32_t semaphore_odd_local_addr = CreateSemaphore(program, core_odd, semaphore_initial_value);
    uint32_t semaphore_even_local_addr = CreateSemaphore(program, core_even, semaphore_initial_value);
    uint32_t semaphore_read_local_addr = CreateSemaphore(program, core_read, semaphore_initial_value);

	constexpr uint32_t cb_write_index = CB::c_in0;
	CircularBufferConfig cb_write_config = CircularBufferConfig(single_tile_size, {{cb_write_index, tt::DataFormat::Float16_b}}).set_page_size(cb_write_index, single_tile_size);
	CBHandle cb_odd = tt_metal::CreateCircularBuffer(program, core_odd, cb_write_config);
    CBHandle cb_even = tt_metal::CreateCircularBuffer(program, core_even, cb_write_config);

    // This CB is on a remote core, so we can use its own cb_in0 here too
	constexpr uint32_t cb_read_index = CB::c_in0;
	CircularBufferConfig cb_read_config = CircularBufferConfig(single_tile_size, {{cb_read_index, tt::DataFormat::Float16_b}}).set_page_size(cb_read_index, single_tile_size);
	CBHandle cb_read = tt_metal::CreateCircularBuffer(program, core_read, cb_read_config);


    // Create Kernel Handles with dynamic directory name
    std::string gen_kernel_path = "tt_metal/programming_examples/one_to_one_datastream_3kernels_evenodd/kernels/" + case_name + "/int_num_gen_kernel.cpp";
    std::string gen_kernel_path2 = "tt_metal/programming_examples/one_to_one_datastream_3kernels_evenodd/kernels/" + case_name + "/int_num_gen_kernel.cpp";
    std::string reader_kernel_path = "tt_metal/programming_examples/one_to_one_datastream_3kernels_evenodd/kernels/" + case_name + "/int_num_reader_kernel.cpp";

    KernelHandle int_num_odd_kernel_id = CreateKernel(
        program,
        gen_kernel_path.c_str(),
        core_odd,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle int_num_even_kernel_id = CreateKernel(
        program,
        gen_kernel_path2.c_str(),
        core_even,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle int_num_reader_kernel_id = CreateKernel(
        program,
        reader_kernel_path.c_str(),
        core_read,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(program, int_num_odd_kernel_id, core_odd,
    {
        collect_dram_buffer->address(), semaphore_odd_local_addr, semaphore_read_local_addr, core_odd_x_cord, core_odd_y_cord, core_read_x_cord, core_read_y_cord, dram_noc_x, dram_noc_y, odd_loop_start
    }
    );
    SetRuntimeArgs(program, int_num_even_kernel_id, core_even,
    {
        collect_dram_buffer->address(), semaphore_even_local_addr, semaphore_read_local_addr, core_even_x_cord, core_even_y_cord, core_read_x_cord, core_read_y_cord, dram_noc_x, dram_noc_y, even_loop_start
    }
    );

    SetRuntimeArgs(program, int_num_reader_kernel_id, core_read,
    {
        collect_dram_buffer->address(), semaphore_odd_local_addr, semaphore_even_local_addr, semaphore_read_local_addr, core_odd_x_cord, core_odd_y_cord, core_even_x_cord, core_even_y_cord, core_read_x_cord, core_read_y_cord, dram_noc_x, dram_noc_y
    }
    );

    // Run the kernels
    EnqueueProgram(cq, program, false);
    Finish(cq);
    CloseDevice(device);
    return 0;
}
