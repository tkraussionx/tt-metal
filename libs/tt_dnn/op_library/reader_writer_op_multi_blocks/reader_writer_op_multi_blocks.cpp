#include <math.h>

#include "tt_dnn/op_library/reader_writer_op_multi_blocks/reader_writer_op_multi_blocks.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "llrt/tt_debug_print_server.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor reader_writer_op_multi_blocks(const Tensor &a, vector<uint32_t> address_map, uint32_t num_blocks, uint32_t block_size) {
    TT_ASSERT(a.layout() == Layout::ROW_MAJOR, "activation should be in row major layout");
    tt_metal::Program *program = new tt_metal::Program();
    //tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});

    tt_xy_pair core = {0, 0};

    TT_ASSERT(not a.on_host(), "Operand needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    uint32_t block_size_bytes = block_size * 2;
    assert(block_size_bytes % single_tile_size == 0);
    uint32_t block_size_tiles = block_size_bytes / single_tile_size;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Tensor output = tt_metal::Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device, MemoryConfig({false,1}));

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    // Only one CB
    // Reader kernel reads from DRAM writes to this CB and writer kernel reads from this CB and writes to DRAM
    uint32_t cb_index = 0;
    uint32_t cb_addr = 220 * 1024;
    uint32_t num_tiles_cb = block_size_tiles * 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        cb_index,
        core,
        num_tiles_cb,
        num_tiles_cb * single_tile_size,
        cb_addr,
        DataFormat::Float16_b
    );

    auto l1_b0 = tt_metal::CreateL1Buffer(program, device, core, address_map.size() * sizeof(uint32_t));
    uint32_t address_map_l1_addr =l1_b0->address();
    vector<uint32_t> reader_kernel_args = {(uint32_t) num_blocks,
                                            src0_dram_buffer->address(),
                                            (uint32_t)dram_src0_noc_xy.x,
                                            (uint32_t)dram_src0_noc_xy.y,
                                            block_size_tiles,
                                            address_map_l1_addr,
                                            block_size_bytes
                                            };

    // Blocked reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_with_address_map_blocked.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Blocked writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_blocked.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> blank_compute_args = {};
    tt_metal::ComputeKernelArgs *blank_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, blank_compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        core,
        blank_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::CompileProgram(device, program, false);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::WriteToDeviceL1(device, core, address_map, address_map_l1_addr);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );
    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (uint32_t) dram_dst_noc_xy.x,
        (uint32_t) dram_dst_noc_xy.y,
        num_blocks,
        block_size_tiles,
        block_size_bytes
        }
    );
    tt_metal::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
