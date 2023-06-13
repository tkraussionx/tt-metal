#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

#include "llrt/tt_debug_print_server.hpp"
// #include "hostdevcommon/debug_print_common.h"

// #include "tools/tt_gdb/tt_gdb.hpp"

namespace tt {
namespace tt_metal {

void create_cb_bmm_single_core_tilize_untilize(Program &program,
                                                Device* device,
                                                CoreCoord core,
                                                DataFormat in0_df,
                                                DataFormat in1_df,
                                                DataFormat out_df,
                                                uint32_t in0_tile_nbytes,
                                                uint32_t in1_tile_nbytes,
                                                uint32_t out_tile_nbytes) {
    // buffer indices
    uint32_t in0_cb = CB::c_in0;
    uint32_t in1_cb = CB::c_in1;
    uint32_t out_cb = CB::c_out0;

    // inputs

    // in0
    auto cb_in0 = CreateCircularBuffer(
        program,
        device,
        in0_cb,
        core,
        1,
        in0_tile_nbytes,
        in0_df
    );
    // in1
    auto cb_in1 = CreateCircularBuffer(
        program,
        device,
        in1_cb,
        core,
        1,
        in1_tile_nbytes,
        in1_df
    );

    // out
    auto cb_output = CreateCircularBuffer(
        program,
        device,
        out_cb,
        core,
        1,
        out_tile_nbytes,
        out_df
    );
}

// NOTE (AS): This is a temporary utility function for dtype -> dformat mapping.
// TODO: Replace with a common utility.
DataFormat get_tensor_df_from_dt_test(const DataType& dt) {
    switch (dt) {
        case DataType::BFLOAT16:
            return DataFormat::Float16_b;
        case DataType::BFLOAT8_B:
            return DataFormat::Bfp8_b;
        default:
            TT_ASSERT(false, "Unsupported data type encountered!!");
            return DataFormat::Invalid;
    } // switch
}



Tensor bmm_test(const Tensor &in0,       // activations
                const Tensor &in1,       // weights
                DataType out_dt) {
    const auto [in0_batch, in0_channel, in0_height, in0_width] = in0.shape();
    const auto [in1_batch, in1_channel, in1_height, in1_width] = in1.shape();

    // input matrix shape checks
    TT_ASSERT(in0_batch == 1, "Supports only batch = 1");
    TT_ASSERT(in1_batch == in0_batch, "Batch dimension needs to match for two inputs");
    TT_ASSERT(in0_channel == in1_channel, "Channel dimension needs to match for two inputs");
    TT_ASSERT(in0_width == in1_height, "Input matrices should be compatible for multiplication");

    // tile size checks
    TT_ASSERT(in0_height % constants::TILE_HEIGHT == 0, "Input tensor in0 height needs to be divisible by TILE_HEIGHT");
    TT_ASSERT(in1_height % constants::TILE_HEIGHT == 0, "Input tensor in1 height needs to be divisible by TILE_HEIGHT");
    TT_ASSERT(in0_width % constants::TILE_WIDTH == 0, "Input tensor in0 width needs to be divisible by TILE_WIDTH");
    TT_ASSERT(in1_width % constants::TILE_WIDTH == 0, "Input tensor in1 width needs to be divisible by TILE_WIDTH");

    // device compatibility checks
    TT_ASSERT(!in0.on_host() && !in1.on_host(), "Operands need to be on the device!");
    TT_ASSERT(in0.device() == in1.device(), "Operands need to be on the same device!");
    TT_ASSERT(in0.buffer() != nullptr && in1.buffer() != nullptr, "Operands need to have buffers allocated on the device!");

    // input data type and formats
    const auto in0_dt = in0.dtype();
    const auto in1_dt = in1.dtype();
    const auto in0_df = get_tensor_df_from_dt_test(in0_dt);
    const auto in1_df = get_tensor_df_from_dt_test(in1_dt);

    // input data format checks
    TT_ASSERT(in1_dt == DataType::BFLOAT16 || in1_dt == DataType::BFLOAT8_B, "in1 only supports BFLOAT16 and BFLOAT8_B formats for now!");

    // output data format
    const auto out_df = get_tensor_df_from_dt_test(out_dt);

    const auto in0_tile_nbytes = tile_size(in0_df);
    const auto in1_tile_nbytes = tile_size(in1_df);
    const auto out_tile_nbytes = tile_size(out_df);

    Buffer *src0_dram_buffer = in0.buffer();
    Buffer *src1_dram_buffer = in1.buffer();

    TT_ASSERT(src0_dram_buffer->size() % in0_tile_nbytes == 0, "Buffer size of tensor in0 must be multiple of tile size");
    TT_ASSERT(src1_dram_buffer->size() % in1_tile_nbytes == 0, "Buffer size of tensor in1 must be multiple of tile size");

    CoreCoord core = {0, 0};
    CoreCoord debug_core = {1, 1};
    Program program = Program();
    Device *device = in0.device();

    tt_start_debug_print_server(device->cluster(), {0}, {debug_core});

    // start debug server for kernel dprint
    // int hart_mask = DPRINT_HART_NC | DPRINT_HART_BR;
    // tt_start_debug_print_server(device->cluster(), {0}, {debug_core});

    const std::array<uint32_t, 4> out_shape{in0_batch, in0_channel, in0_height, in1_width};
    Tensor output = Tensor(out_shape,
                           out_dt,
                           Layout::TILE,
                           device);
    Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // in0
    uint32_t in0_dram_addr = src0_dram_buffer->address();
    // in1
    uint32_t in1_dram_addr = src1_dram_buffer->address();
    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    auto out_dram_noc_xy = dst_dram_buffer->noc_coordinates();
    uint32_t out_dram_noc_x = out_dram_noc_xy.x;
    uint32_t out_dram_noc_y = out_dram_noc_xy.y;

    {   // debug
        // in0
        log_debug("in0_dram_addr: {}", in0_dram_addr);
        log_debug("in0_df: {}", in0_df);
        // in1
        log_debug("in1_dram_addr: {}", in1_dram_addr);
        log_debug("in1_df: {}", in1_df);
        // out
        log_debug("out_dram_addr: {}", out_dram_addr);
        log_debug("out_df: {}", out_df);
    }

    create_cb_bmm_single_core_tilize_untilize(
        program,
        in0.device(),
        core,
        in0_df,
        in1_df,
        out_df,
        in0_tile_nbytes,
        in1_tile_nbytes,
        out_tile_nbytes);

    // Reader kernel
    std::string reader_kernel;
    std::vector<uint32_t> reader_rt_args;
    reader_kernel = "tt_metal/kernels/dataflow/reader_bmm_single_core.cpp";
    reader_rt_args = {
        in0_dram_addr,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        in1_dram_addr,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        static_cast<uint32_t>(in0_df),
        static_cast<uint32_t>(in1_df)
    };
    auto reader = CreateDataMovementKernel(
        program,                            // program
        reader_kernel,                      // file name
        core,                               // core
        DataMovementProcessor::RISCV_1,     // processor type
        NOC::RISCV_1_default                // noc
    );

    // Writer kernel
    std::string writer_kernel;
    vector<uint32_t> writer_rt_args;
    writer_kernel = "tt_metal/kernels/dataflow/writer_bmm_single_core_tiled.cpp";
    writer_rt_args = {
        out_dram_addr,
        0,                                              // UNUSED
        1,                                              // out_stride_w
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        static_cast<uint32_t>(out_df)
    };
    auto writer = CreateDataMovementKernel(
        program,                        // program
        writer_kernel,                  // file name
        core,                           // core
        DataMovementProcessor::RISCV_0, // processor type
        NOC::RISCV_0_default);          // noc

    // Compute kernel
    std::string compute_kernel = "tt_metal/kernels/compute/bmm_test.cpp";
    std::vector<uint32_t> compute_comptime_args = {};
    auto bmm_compute = CreateComputeKernel(
        program,
        compute_kernel,
        core,
        compute_comptime_args,
        MathFidelity::HiFi4,
        false,  // fp32_dest_acc_en
        false   // math_approx_mode
    );

    // Reader rt args
    WriteRuntimeArgsToDevice(device, reader, core, reader_rt_args);
    // Writer rt args
    WriteRuntimeArgsToDevice(device, writer, core, writer_rt_args);

    // Compile and launch
    bool pass = CompileProgram(device, program, false);
    pass &= ConfigureDeviceWithProgram(device, program);
    pass &= LaunchKernels(device, program);

    TT_ASSERT(pass);

    return output;
}

}  // namespace tt_metal
}  // namespace tt
