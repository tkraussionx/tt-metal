#include "tt_metal/host_api.hpp"

int main() {
    // Create device object and get command queue.
    constexpr int device_id = 0;
    tt::tt_metal::Device *device = tt::tt_metal::CreateDevice(device_id);
    tt::tt_metal::CommandQueue &cq = device->command_queue();

    /////////////////////////////////////////////////////////////////////////////////
    // Create program instance.
    /////////////////////////////////////////////////////////////////////////////////
    Program program = tt::tt_metal::CreateProgram();
    CoreCoord core = CoreCoord{0, 0};

    /////////////////////////////////////////////////////////////////////////////////
    // Create kernels
    /////////////////////////////////////////////////////////////////////////////////
    auto data_movement_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/debugging_and_profiling/kernels/dprint_exercise1.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

    //////////////////////////////////////////////////////////////////////////////////
    // EnqueueProgram and Copy output_dram_buffer to host buffer1
    //////////////////////////////////////////////////////////////////////////////////
    EnqueueProgram(cq, program, true /*blocking*/);

    bool pass = true;
    pass = tt::tt_metal::CloseDevice(device);

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }
    return 0;
}
