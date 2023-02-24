#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "hostdevcommon/common_values.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        // set up the program

        // pass
        //uint32_t num_cores = 2;
        //uint32_t num_tiles = 2048;
        //uint32_t block_size_tiles = 8;
        //uint32_t double_buffer = true; // 2 cores pass w/ double buffer

        // pass
        //uint32_t num_cores = 12;
        //uint32_t num_tiles = 2048;
        //uint32_t block_size_tiles = 8;
        //uint32_t double_buffer = false; // 12 cores pass w/o double buffer

        // hang
        // uint32_t num_cores = 3; // 3 cores hang w/ double buffer
        // uint32_t num_tiles = 16;
        // uint32_t block_size_tiles = 8;
        // uint32_t double_buffer = true;

        // pass (same as above but num_tiles == block_size, so double buffers not exercised)
        uint32_t num_cores = 3; // 
        uint32_t num_tiles = 8; // 
        uint32_t block_size_tiles = 8;
        uint32_t double_buffer = true;

        TT_ASSERT(num_cores >= 2 && num_cores <= 12); // grayskull
        TT_ASSERT(num_tiles % block_size_tiles == 0);

        std::vector<tt_xy_pair> cores;
        for (uint32_t i = 0; i < num_cores; i++) {
            cores.push_back({i, 0});    
        }

        log_info(LogTest, "num_cores: {}", num_cores);
        log_info(LogTest, "num_tiles: {}", num_tiles);
        log_info(LogTest, "block_size_tiles: {}", block_size_tiles);
        log_info(LogTest, "double_buffer: {}", double_buffer);

        uint32_t single_tile_size = 2 * 1024;

        // source and destination buffers
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        auto src_dram_buffer = ll_buda::CreateDramBuffer(0, dram_buffer_size, 0);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(7, dram_buffer_size, 0);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates(device);
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates(device);

        // circular buffers
        uint32_t cb_index = 8;
        uint32_t cb_addr = 250 * 1024;
        uint32_t cb_size_tiles;
        if (double_buffer) {
            cb_size_tiles = 2 * block_size_tiles;
        } else {
            cb_size_tiles = block_size_tiles;
        }

        for (auto core : cores) {
            auto cb = ll_buda::CreateCircularBuffer(
                program,
                cb_index,
                core,
                cb_size_tiles,
                cb_size_tiles * single_tile_size,
                cb_addr,
                tt::DataFormat::Float16_b
            );
        }

        // semaphores in L1, 32B aligned for NOC transfers
        uint32_t sender_semaphore_addr = 109600;
        uint32_t receiver_semaphore_addr = 109632;
        TT_ASSERT(sender_semaphore_addr % 32 == 0);
        TT_ASSERT(receiver_semaphore_addr % 32 == 0);

        // create kernels 
        vector<ll_buda::DataMovementKernel*> receiver_kernels;
        vector<ll_buda::DataMovementKernel*> sender_kernels;
        for (int core_id = 0; core_id < num_cores; core_id++) {

            string receiver_kernel_name;
            if (core_id == 0) {
                receiver_kernel_name = "kernels/dataflow/reader_first_stage.cpp";
            } else {
                receiver_kernel_name = "kernels/dataflow/receiver_intermediate_stage.cpp";
            }

            ll_buda::DataMovementKernelArgs* receiver_kernel_compile_time_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
            receiver_kernels.push_back(ll_buda::CreateDataMovementKernel(
                program,
                receiver_kernel_name,
                cores[core_id],
                receiver_kernel_compile_time_args,
                ll_buda::DataMovementProcessor::RISCV_1,
                ll_buda::NOC::RISCV_1_default));

            string sender_kernel_name;
            if (core_id == num_cores - 1) {
                sender_kernel_name = "kernels/dataflow/writer_last_stage.cpp";
            } else {
                sender_kernel_name = "kernels/dataflow/sender_intermediate_stage.cpp";
            }
            ll_buda::DataMovementKernelArgs* sender_kernel_compile_time_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
            sender_kernels.push_back(ll_buda::CreateDataMovementKernel(
                program,
                sender_kernel_name,
                cores[core_id],
                sender_kernel_compile_time_args,
                ll_buda::DataMovementProcessor::RISCV_0,
                ll_buda::NOC::RISCV_0_default));
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        constexpr bool profile_kernel = true;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc, profile_kernel);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        pass &= ll_buda::WriteToDeviceDRAM(device, src_dram_buffer, src_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program, profile_kernel);

        // host initializes only the sender's semaphores, reciver's semaphores are initialized by the kernel
        std::vector<uint32_t> invalid = {INVALID};
        for (auto core : cores) {
            ll_buda::WriteToDeviceL1(device, core, invalid, sender_semaphore_addr);
        }

        // send run-time kernel arguments
        for (int core_id = 0; core_id < num_cores; core_id++) {
            if (core_id == 0) {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    receiver_kernels[core_id],
                    cores[core_id],
                    {src_dram_buffer->address(),
                    (uint32_t)dram_src_noc_xy.x,
                    (uint32_t)dram_src_noc_xy.y,
                    (uint32_t)num_tiles});
            } else {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    receiver_kernels[core_id],
                    cores[core_id],
                    {(uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).x,
                    (uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).y,
                    (uint32_t)num_tiles,
                    (uint32_t)sender_semaphore_addr,
                    (uint32_t)receiver_semaphore_addr});
            }

            if (core_id == num_cores - 1) {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    sender_kernels[core_id],
                    cores[core_id],
                    {dst_dram_buffer->address(),
                    (uint32_t)dram_dst_noc_xy.x,
                    (uint32_t)dram_dst_noc_xy.y,
                    (uint32_t)num_tiles});
            } else {
                ll_buda::WriteRuntimeArgsToDevice(
                    device,
                    sender_kernels[core_id],
                    cores[core_id],
                    {(uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).x,
                    (uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).y,
                    (uint32_t)num_tiles,
                    (uint32_t)sender_semaphore_addr,
                    (uint32_t)receiver_semaphore_addr});
            }
        }


        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAM(device, dst_dram_buffer, result_vec, dst_dram_buffer->size());
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src_vec == result_vec);

        pass &= ll_buda::CloseDevice(device);;

    } catch (const std::exception &e) {
        pass = false;
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
