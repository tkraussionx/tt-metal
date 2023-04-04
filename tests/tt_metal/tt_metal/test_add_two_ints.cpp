#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"

////////////////////////////////////////////////////////////////////////////
// Runs the add_two_ints kernel on BRISC to add two ints in L1
// Result is read from L1
////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program *program = new tt_metal::Program();
        tt_xy_pair core = {0, 0};
        std::vector<uint32_t> first_runtime_args = {101, 202};
        std::vector<uint32_t> second_runtime_args = {303, 606};

        tt_metal::DataMovementKernel *add_two_ints_kernel = tt_metal::CreateDataMovementKernel(
            program, "tt_metal/kernels/riscv_draft/add_two_ints.cpp", core, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        constexpr bool profile_device = true;
        pass &= tt_metal::CompileProgram(device, program, profile_device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(device, add_two_ints_kernel, core, first_runtime_args);

        pass &= tt_metal::LaunchKernels(device, program);
        tt_metal::DumpDeviceProfileResults(device, program);

        std::vector<uint32_t> second_kernel_result;
        tt_metal::ReadFromDeviceL1(device, core, BRISC_L1_RESULT_BASE, second_kernel_result, sizeof(int));

        //pass &= tt_metal::LaunchKernels(device, program);
        //tt_metal::DumpDeviceProfileResults(device, program);

        pass &= tt_metal::CloseDevice(device);;

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
