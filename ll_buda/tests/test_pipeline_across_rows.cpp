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

        std::vector<tt_xy_pair> cores = {{0,0}, {1,0}, {2,0}};

        uint32_t single_tile_size = 2 * 1024;
        
        uint32_t num_tiles = 2048;
        //uint32_t num_tiles = 16;

        uint32_t block_size_tiles = 8;
        TT_ASSERT(num_tiles % block_size_tiles == 0);
        uint32_t double_buffer = false;

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

        // kernels 

        // core 0
        int core_id = 0;
        auto reader_first_stage_kernel_args  = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
        auto reader_first_stage_kernel       = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_first_stage.cpp",
            cores[core_id],
            reader_first_stage_kernel_args,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto sender_intermediate_stage_kernel_args_0 = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
        auto sender_intermediate_stage_kernel_0      = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/sender_intermediate_stage.cpp",
            cores[core_id],
            sender_intermediate_stage_kernel_args_0,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);
        
        // core 1
        core_id = 1;
        auto receiver_intermediate_stage_kernel_args_1 = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
        auto receiver_intermediate_stage_kernel_1      = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/receiver_intermediate_stage.cpp",
            cores[core_id],
            receiver_intermediate_stage_kernel_args_1,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto sender_intermediate_stage_kernel_args_1 = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
        auto sender_intermediate_stage_kernel_1     = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/sender_intermediate_stage.cpp",
            cores[core_id],
            sender_intermediate_stage_kernel_args_1,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        // core 2
        core_id = 2;
        auto receiver_intermediate_stage_kernel_args_2 = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
        auto receiver_intermediate_stage_kernel_2      = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/receiver_intermediate_stage.cpp",
            cores[core_id],
            receiver_intermediate_stage_kernel_args_2,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto writer_last_stage_kernel_args = ll_buda::InitializeCompileTimeDataMovementKernelArgs(cores[core_id], {cb_index, block_size_tiles});
        auto writer_last_stage_kernel      = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_last_stage.cpp",
            cores[core_id],
            writer_last_stage_kernel_args,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);


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

        // core 0
        core_id = 0;
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            reader_first_stage_kernel,
            cores[core_id],
            {src_dram_buffer->address(),
            (uint32_t)dram_src_noc_xy.x,
            (uint32_t)dram_src_noc_xy.y,
            (uint32_t)num_tiles});

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            sender_intermediate_stage_kernel_0,
            cores[core_id],
            {(uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).x,
             (uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).y,
             (uint32_t)num_tiles,
             (uint32_t)sender_semaphore_addr,
             (uint32_t)receiver_semaphore_addr});

        // core 1
        core_id = 1;
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            receiver_intermediate_stage_kernel_1,
            cores[core_id],
            {(uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).x,
             (uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).y,
             (uint32_t)num_tiles,
             (uint32_t)sender_semaphore_addr,
             (uint32_t)receiver_semaphore_addr});

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            sender_intermediate_stage_kernel_1,
            cores[core_id],
            {(uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).x,
             (uint32_t)device->worker_core_from_logical_core(cores[core_id+1]).y,
             (uint32_t)num_tiles,
             (uint32_t)sender_semaphore_addr,
             (uint32_t)receiver_semaphore_addr});


        // core 2
        core_id = 2;
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            receiver_intermediate_stage_kernel_2,
            cores[core_id],
            {(uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).x,
             (uint32_t)device->worker_core_from_logical_core(cores[core_id-1]).y,
             (uint32_t)num_tiles,
             (uint32_t)sender_semaphore_addr,
             (uint32_t)receiver_semaphore_addr});

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            writer_last_stage_kernel,
            cores[core_id],
            {dst_dram_buffer->address(),
            (uint32_t)dram_dst_noc_xy.x,
            (uint32_t)dram_dst_noc_xy.y,
            (uint32_t)num_tiles});

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
