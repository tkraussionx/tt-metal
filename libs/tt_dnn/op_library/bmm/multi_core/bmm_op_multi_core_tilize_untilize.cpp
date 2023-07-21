#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

namespace tt {
namespace tt_metal {

void create_cb_bmm_multi_core_tilize_untilize(Program &program,
                                              CoreRange cores,
                                              DataFormat in0_df,
                                              DataFormat in1_df,
                                              DataFormat out_df,
                                              uint32_t in0_block_w,
                                              uint32_t in0_block_h,
                                              uint32_t in1_block_w,
                                              uint32_t in0_tile_nbytes,
                                              uint32_t in1_tile_nbytes,
                                              uint32_t out_tile_nbytes,
                                              bool tilize_in0 = true,
                                              bool untilize_out = true) {
    // buffer indices
    uint32_t in0_cb                                 = CB::c_in0;
    uint32_t in1_cb                                 = CB::c_in1;
    uint32_t matmul_partials_cb                     = CB::c_intermed0;
    uint32_t tilize_mode_tilized_in0_cb             = CB::c_intermed1;
    uint32_t untilize_mode_final_matmul_partials_cb = CB::c_intermed2;
    uint32_t untilize_mode_reblock_cb               = CB::c_intermed3;
    uint32_t out_cb                                 = CB::c_out0;

    const uint32_t cb0_ntiles = in0_block_h * in0_block_w * 2;  // double buffer
    const uint32_t cb1_ntiles = in0_block_w * in1_block_w * 2;  // double buffer
    const uint32_t out_ntiles = in0_block_h * in1_block_w;

    // in0 (RM/TM)
    auto cb_in0 = CreateCircularBuffers(
        program,
        in0_cb,
        cores,
        cb0_ntiles,
        cb0_ntiles * in0_tile_nbytes,
        in0_df
    );
    // in1 (TM)
    auto cb_in1 = CreateCircularBuffers(
        program,
        in1_cb,
        cores,
        cb1_ntiles,
        cb1_ntiles * in1_tile_nbytes,
        in1_df
    );

    if (tilize_in0) {
        // in0 (RM -> TM)
        auto cb_src0_tilized = CreateCircularBuffers(
            program,
            tilize_mode_tilized_in0_cb,
            cores,
            cb0_ntiles,
            cb0_ntiles * in0_tile_nbytes,
            in0_df
        );
    }

    if (untilize_out) {
        // partial sums
        auto cb_matmul_partials = CreateCircularBuffers(
            program,
            matmul_partials_cb,
            cores,
            out_ntiles,
            out_ntiles * out_tile_nbytes,
            out_df
        );
        // final partial sums
        auto cb_final_matmul_partials = CreateCircularBuffers(
            program,
            untilize_mode_final_matmul_partials_cb,
            cores,
            out_ntiles,
            out_ntiles * out_tile_nbytes,
            out_df
        );
        // to reorganize output blocks to fill the whole "per core output block width"
        auto cb_reblock = CreateCircularBuffers(
            program,
            untilize_mode_reblock_cb,
            cores,
            in1_block_w,                    // a single row of tiles
            in1_block_w * out_tile_nbytes,
            out_df
        );
        // output
        auto cb_output = CreateCircularBuffers(
            program,
            out_cb,
            cores,
            out_ntiles,
            out_ntiles * out_tile_nbytes,
            out_df
        );
    } else {
        // partials and output share same memory
        CoreRangeSet cores_set(std::set<CoreRange>{ cores });
        auto cb_matmul_partials = CreateCircularBuffers(
            program,
            { matmul_partials_cb, out_cb },
            cores_set,
            out_ntiles,
            out_ntiles * out_tile_nbytes,
            out_df
        );
    }
}

operation::ProgramWithCallbacks bmm_multi_core_tilize_untilize(
                                    const Tensor &in0,       // activations
                                    const Tensor &in1,       // weights
                                    DataType out_dt,
                                    uint32_t in0_nblocks_h,
                                    uint32_t in0_nblocks_w,
                                    uint32_t in1_nblocks_w,
                                    uint32_t in0_block_ntiles_h,
                                    uint32_t in0_block_ntiles_w,
                                    uint32_t in1_block_ntiles_w,
                                    uint32_t out_subblock_ntiles_h,
                                    uint32_t out_subblock_ntiles_w,
                                    bool tilize_in0,
                                    bool untilize_out,
                                    const CoreCoord& grid_size,
                                    Tensor &out) {
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
    TT_ASSERT(in0.storage_type() == StorageType::DEVICE and in1.storage_type() == StorageType::DEVICE, "Operands need to be on the device!");
    TT_ASSERT(in0.device() == in1.device(), "Operands need to be on the same device!");

    // data type and formats
    const auto in0_dt = in0.dtype();
    const auto in1_dt = in1.dtype();
    const auto in0_df = datatype_to_dataformat_converter(in0_dt);
    const auto in1_df = datatype_to_dataformat_converter(in1_dt);
    const auto out_df = datatype_to_dataformat_converter(out_dt);
    const auto in0_tile_nbytes = tile_size(in0_df);
    const auto in1_tile_nbytes = tile_size(in1_df);
    const auto out_tile_nbytes = tile_size(out_df);

    // input/output data type checks
    TT_ASSERT(in0_dt == DataType::BFLOAT16 || (in0_dt == DataType::BFLOAT8_B && !tilize_in0),
              "in0 only supports BFLOAT16 and BFLOAT8_B data types for now");
    TT_ASSERT(in1_dt == DataType::BFLOAT16 || in1_dt == DataType::BFLOAT8_B, "in1 only supports BFLOAT16 and BFLOAT8_B formats for now!");
    TT_ASSERT(!untilize_out || (untilize_out && out_dt == DataType::BFLOAT16), "out only supports BFLOAT16 when untilizing.");

    Buffer *src0_dram_buffer = in0.buffer();
    Buffer *src1_dram_buffer = in1.buffer();
    Buffer *dst_dram_buffer = out.buffer();

    TT_ASSERT(src0_dram_buffer != nullptr && src1_dram_buffer != nullptr, "Operands need to have buffers allocated on the device!");
    TT_ASSERT(src0_dram_buffer->size() % in0_tile_nbytes == 0, "Buffer size of tensor in0 must be multiple of tile size");
    TT_ASSERT(src1_dram_buffer->size() % in1_tile_nbytes == 0, "Buffer size of tensor in1 must be multiple of tile size");
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    TT_ASSERT(dst_dram_buffer->size() % out_tile_nbytes == 0, "Buffer size of tensor out must be multiple of tile size");
    TT_ASSERT(src0_dram_buffer->buffer_type() == BufferType::DRAM &&
              src1_dram_buffer->buffer_type() == BufferType::DRAM &&
              dst_dram_buffer->buffer_type() == BufferType::DRAM, "All buffers are expected to be in DRAM!");

    Program program{};
    Device *device = in0.device();
    uint32_t ncores_w = grid_size.x, ncores_h = grid_size.y;

    // TODO [AS]: Idea is to take in just the input tensor sizes (as in number of tiles in the full output tensor,)
    // and calculate the block decomposition based on how to distribute the tiles equally across all given cores in the grid.
    // For now just take in the given decomposition and core grid size assuming they have been decided intelligently until the
    // full automatic work decomposition is implemented.

    // TODO: Get rid of the assumption that the blocks are uniformly distributed across all cores.
    auto [per_core_nblocks_h, per_core_nblocks_w] = split_matmul_blocks_to_cores(in0_nblocks_h, in0_nblocks_w, in1_nblocks_w,
                                                                                 in0_block_ntiles_h, in0_block_ntiles_w, in1_block_ntiles_w,
                                                                                 out_subblock_ntiles_h, out_subblock_ntiles_w,
                                                                                 ncores_h, ncores_w);
    CoreCoord core_start = {0, 0}, core_end = {ncores_w - 1, ncores_h - 1};
    CoreRange core_range { .start = core_start, .end = core_end };

    // start debug server for kernel dprint
    // int hart_mask = DPRINT_HART_NC | DPRINT_HART_BR;
    // CoreCoord debug_core = {1, 1};  // corresponds to core {0, 0}
    // tt_start_debug_print_server(device->cluster(), {0}, {debug_core});

    // Convert tensor dims to tile dims
    uint32_t in0_ntiles_h = in0_height / constants::TILE_HEIGHT;    // == in0_nblocks_h * in0_block_ntiles_h
    uint32_t in0_ntiles_w = in0_width / constants::TILE_WIDTH;      // == in0_nblocks_w * in0_block_ntiles_w
    uint32_t in1_ntiles_w = in1_width / constants::TILE_WIDTH;      // == in1_nblocks_w * in1_block_ntiles_w
    // Ensure the size arguments match the input tensors
    TT_ASSERT(in0_ntiles_h == in0_nblocks_h * in0_block_ntiles_h, "Mismatch in tensor in0 height!");
    TT_ASSERT(in0_ntiles_w == in0_nblocks_w * in0_block_ntiles_w, "Mismatch in tensor in0 width!");
    TT_ASSERT(in1_ntiles_w == in1_nblocks_w * in1_block_ntiles_w, "Mismatch in tensor in1 width!");

    // in0
    uint32_t in0_dram_addr = src0_dram_buffer->address();
    uint32_t in0_subblock_h = out_subblock_ntiles_h;
    uint32_t in0_nblocks_w = in0_nblocks_w;
    uint32_t in0_nblocks_h = in0_nblocks_h;
    uint32_t in0_block_w = in0_ntiles_w / in0_nblocks_w;
    uint32_t in0_block_h = in0_ntiles_h / in0_nblocks_h;
    uint32_t in0_block_ntiles = in0_block_h * in0_block_w;
    uint32_t in0_nsubblocks = in0_block_h / in0_subblock_h;
    uint32_t in0_subblock_ntiles = in0_subblock_h * in0_block_w;
    TT_ASSERT(in0_block_h % out_subblock_ntiles_h == 0);

    // in1
    uint32_t in1_dram_addr = src1_dram_buffer->address();
    uint32_t in1_subblock_w = out_subblock_ntiles_w;
    uint32_t in1_nblocks_w = in1_nblocks_w;
    uint32_t in1_nblocks_h = in0_nblocks_w;
    uint32_t in1_block_w = in1_block_ntiles_w;
    uint32_t in1_nsubblocks = in1_block_w / in1_subblock_w;
    uint32_t in1_block_h = in0_block_w;
    uint32_t in1_block_ntiles = in1_block_w * in1_block_h;
    TT_ASSERT(in1_block_w % out_subblock_ntiles_w == 0);

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_subblock_ntiles = out_subblock_ntiles_h * out_subblock_ntiles_w;
    TT_ASSERT(out_subblock_ntiles <= 8, "Subblock can have at most 8 tiles to fit computed intermediates in dst[half]");

    {   // debug
        // in0
        log_debug("in0_dram_addr: {}", in0_dram_addr);
        log_debug("in0_ntiles_h: {}", in0_ntiles_h);
        log_debug("in0_ntiles_w: {}", in0_ntiles_w);
        log_debug("in0_subblock_h: {}", in0_subblock_h);
        log_debug("in0_nblocks_w: {}", in0_nblocks_w);
        log_debug("in0_nblocks_h: {}", in0_nblocks_h);
        log_debug("in0_block_w: {}", in0_block_w);
        log_debug("in0_block_h: {}", in0_block_h);
        log_debug("in0_block_ntiles: {}", in0_block_ntiles);
        log_debug("in0_nsubblocks: {}", in0_nsubblocks);
        log_debug("in0_subblock_ntiles: {}", in0_subblock_ntiles);
        // in1
        log_debug("in1_dram_addr: {}", in1_dram_addr);
        log_debug("in1_ntiles_w: {}", in1_ntiles_w);
        log_debug("in1_subblock_w: {}", in1_subblock_w);
        log_debug("in1_nsubblocks: {}", in1_nsubblocks);
        log_debug("in1_block_ntiles: {}", in1_block_ntiles);
        log_debug("in1_block_w: {}", in1_block_w);
        log_debug("in1_block_h: {}", in1_block_h);
        log_debug("in1_nblocks_w: {}", in1_nblocks_w);
        log_debug("in1_nblocks_h: {}", in1_nblocks_h);
        // out
        log_debug("out_dram_addr: {}", out_dram_addr);
        log_debug("out_subblock_ntiles_h: {}", out_subblock_ntiles_h);
        log_debug("out_subblock_ntiles_w: {}", out_subblock_ntiles_w);
        log_debug("out_subblock_ntiles: {}", out_subblock_ntiles);
        // extra
        log_debug("out size: {}", dst_dram_buffer->size());
        log_debug("out pagesize: {}", dst_dram_buffer->page_size());
        // data formats
        log_debug("in0_df: {}", in0_df);
        log_debug("in1_df: {}", in1_df);
        log_debug("out_df: {}", out_df);
    }

    // create CBs for L1
    create_cb_bmm_multi_core_tilize_untilize(program, core_range,
                                             in0_df, in1_df, out_df,
                                             in0_block_w, in0_block_h, in1_block_w,
                                             in0_tile_nbytes, in1_tile_nbytes, out_tile_nbytes,
                                             tilize_in0, untilize_out);

    // Reader kernel
    std::string reader_kernel;
    std::vector<uint32_t> reader_rt_args;
    if (tilize_in0) {
        // in0 is row major, in1 is tiled
        // NOTE: this only makes sense for non-tile-shared datatypes for in0
        reader_kernel = "tt_metal/kernels/dataflow/reader_bmm_single_core_tilize_untilize.cpp";
        reader_rt_args = {
            // in0
            in0_dram_addr,
            in0_block_h,
            in0_nblocks_h,
            in0_nblocks_w,
            in0_block_ntiles,
            in0_block_h * constants::TILE_HEIGHT,                       // in0_block_nrows,
            in0.element_size(),                                         // UNUSED
            in0_width * in0.element_size(),                             // page size (size of an in0 row)
            in0_block_w * constants::TILE_WIDTH * in0.element_size(),   // size of partial row to fit within a block width
            // in1
            in1_dram_addr,
            in1_block_h,
            in1_block_w,
            in1_nblocks_w,
            in1_block_ntiles,
            in1_ntiles_w,
            in1_ntiles_w * in1_block_h,
            in1_block_w,
            static_cast<uint32_t>(in0_df),
            static_cast<uint32_t>(in1_df)
        };
    } else {
        // in0 is tiled, in1 is tiled
        reader_kernel = "tt_metal/kernels/dataflow/reader_bmm_single_core.cpp";
        reader_rt_args = {
            // in0
            in0_dram_addr,                  // in0_addr
            in0_nblocks_h,
            in0_nblocks_w,
            1,                              // in0_stride_w
            in0_ntiles_w,                   // in0_stride_h
            in0_block_w,                    // in0_next_block_stride
            in0_block_w,                    // in0_block_w
            in0_block_h,                    // in0_block_h
            in0_block_ntiles,               // in0_block_ntiles
            // in1
            in1_dram_addr,                  // in1_addr
            in1_nblocks_w,
            0,                              // in1_start_tile_id
            1,                              // in1_stride_w
            in1_ntiles_w,                   // in1_stride_h
            in0_block_w * in1_ntiles_w,     // in1_next_block_stride UNUSED
            in1_block_w,                    // in1_block_w
            in1_block_h,                    // in1_block_h
            in1_block_ntiles,               // in1_block_ntiles
            in0_ntiles_w * in0_block_h,     // in0_next_block_stride_h,
            in0_block_w,                    // in0_next_block_stride_w,
            in1_ntiles_w * in1_block_h,     // in1_next_block_stride_h,
            in1_block_w,                    // in1_next_block_stride_w
            static_cast<uint32_t>(in0_df),
            static_cast<uint32_t>(in1_df)
        };
    }
    auto reader = CreateDataMovementKernel(
        program,                            // program
        reader_kernel,                      // file name
        core_range,                         // core
        DataMovementProcessor::RISCV_1,     // processor type
        NOC::RISCV_1_default                // noc
    );


    // Writer kernel
    std::string writer_kernel;
    vector<uint32_t> writer_rt_args;

    if (untilize_out) {
        // out is row major
        writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank_blocks.cpp";
        writer_rt_args = {
            out_dram_addr,
            in0_height / in0_nblocks_h;                                 // data elements along height of an in0 block
            in1_block_w * constants::TILE_WIDTH * out.element_size(),   // block_row_size
            1,                                                          // batch
            in0_nblocks_h,
            in1_nblocks_w,
            in1_width * out.element_size(),                             // output_row_size
            static_cast<uint32_t>(out_df)
        };
    } else {
        // out is tiled
        writer_kernel = "tt_metal/kernels/dataflow/writer_bmm_single_core_tiled.cpp";
        writer_rt_args = {
            out_dram_addr,
            0,                                              // UNUSED
            1,                                              // out_stride_w
            in1_ntiles_w,                               // out_stride_h
            out_subblock_ntiles_w,                      // out_next_subblock_stride_w
            out_subblock_ntiles_h * in1_ntiles_w,  // out_next_subblock_stride_h
            out_subblock_ntiles_w,                      // out_subblock_w
            out_subblock_ntiles_h,                     // out_subblock_h
            out_subblock_ntiles,                            // out_subblock_tile_count
            in1_ntiles_w / out_subblock_ntiles_w,   // out_nsubblocks_w
            in0_ntiles_h / out_subblock_ntiles_h, // out_nsubblocks_h
            static_cast<uint32_t>(out_df)
        };
    }
    auto writer = CreateDataMovementKernel(
        program,                        // program
        writer_kernel,                  // file name
        core_range,                     // core
        DataMovementProcessor::RISCV_0, // processor type
        NOC::RISCV_0_default);          // noc

    // Compute kernel
    std::string compute_kernel = "tt_metal/kernels/compute/bmm_tilize_untilize.cpp";
    std::vector<uint32_t> compute_comptime_args = {
        in0_block_w,
        in0_nsubblocks,
        in0_block_ntiles,
        in0_subblock_ntiles,
        in0_subblock_h,
        in1_nsubblocks,
        in1_block_ntiles,
        in1_block_w,
        in0_nblocks_h,
        in0_nblocks_w,
        in1_nblocks_w,
        out_subblock_ntiles_h,
        out_subblock_ntiles_w,
        out_subblock_ntiles,
        tilize_in0,
        untilize_out
    };
    auto bmm_compute = CreateComputeKernel(
        program,
        compute_kernel,
        core_range,
        compute_comptime_args,
        MathFidelity::HiFi4,
        false,  // fp32_dest_acc_en
        false   // math_approx_mode
    );

    // Reader rt args
    SetRuntimeArgs(reader, core_range, reader_rt_args);
    // Writer rt args
    SetRuntimeArgs(writer, core_range, writer_rt_args);

    // Compile and launch
    bool pass = CompileProgram(device, program, false);
    pass &= ConfigureDeviceWithProgram(device, program);
    WriteRuntimeArgsToDevice(device, program);
    pass &= LaunchKernels(device, program);

    TT_ASSERT(pass);

    auto override_runtime_args_callback = [kernel_reader = reader, kernel_writer = writer](
                                            const std::vector<Buffer*>& input_buffers,
                                            const std::vector<Buffer*>& output_buffers) {
        auto in0_dram_buffer = input_buffers.at(0);
        auto in1_dram_buffer = input_buffers.at(1);
        auto out_dram_buffer = output_buffers.at(0);
        CoreCoord core = {0, 0};
        {
            auto runtime_args = GetRuntimeArgs(kernel_reader, core);
            runtime_args[0] = in0_dram_buffer->address();
            runtime_args[9] = in1_dram_buffer->address();
            SetRuntimeArgs(kernel_reader, core, runtime_args);
        }
        {
            auto runtime_args = GetRuntimeArgs(kernel_writer, core);
            runtime_args[0] = out_dram_buffer->address();
            SetRuntimeArgs(kernel_writer, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
} // bmm_multi_core_tilize_untilize()


}  // namespace tt_metal
}  // namespace tt
