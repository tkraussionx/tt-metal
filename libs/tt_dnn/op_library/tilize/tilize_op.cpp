#include <math.h>

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "tt_dnn/op_library/conv/conv_op.hpp"
#include "llrt/tt_debug_print_server.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor tilize(const Tensor &a) {
    if (a.layout() == Layout::TILE) {
        std::cout << "Perf warning: tilize called on already tilized tensor." << std::endl;
        return a;
    } else {
        TT_ASSERT(a.layout() == Layout::ROW_MAJOR or a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");
    }
    tt_metal::Program *program = new tt_metal::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;
    uint32_t stick_s =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_sticks = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] * a.shape()[2] : a.shape()[0] * a.shape()[2] * a.shape()[3];
    uint32_t stick_size = stick_s * 2; // Assuming bfloat16 dataformat
    TT_ASSERT((stick_size % 2) == 0, "Stick size must be divisible by 2");

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = a.shape();
    if(a.layout() == Layout::CHANNELS_LAST) {
        // Set channels last in the innermost dim in the shape
        output_shape = {a.shape()[0], a.shape()[2], a.shape()[3], a.shape()[1]};
    }
    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = stick_s / 32;

    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = stick_s / 32;

    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    // Reader compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(stick_size)) == floor(log2(stick_size)));
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(), num_sticks, stick_size};
    DataMovementKernelArgs *compile_time_args;
    if (stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {1});
    } else {
        compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {0});
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
        core,
        compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_sticks / 32), // per_core_block_cnt
        uint32_t(stick_s / 32) // per_core_block_tile_cnt
    };
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        eltwise_unary_args,
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
        (uint32_t) (a.shape()[0] * a.shape()[1] * a.shape()[2] * a.shape()[3] / TILE_HW)}
    );
    tt_metal::LaunchKernels(device, program);

    delete program;

    return output;
}

Tensor tilize_with_zero_padding(const Tensor &a) {
    if (a.layout() == Layout::TILE) {
        std::cout << "Perf warning: tilize called on already tilized tensor." << std::endl;
        return a;
    } else {
        TT_ASSERT(a.layout() == Layout::ROW_MAJOR or a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");
    }
    tt_metal::Program *program = new tt_metal::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = a.shape();
    if(a.layout() == Layout::CHANNELS_LAST) {
        // Set channels last in the innermost dim in the shape
        output_shape = {a.shape()[0], a.shape()[2], a.shape()[3], a.shape()[1]};
    }
    // pad height
    output_shape[2] = (uint32_t) (ceil((double) output_shape[2] / (double) TILE_HEIGHT ) * TILE_HEIGHT);
    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::TILE, device);


    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(output.volume() % TILE_HW == 0);
    int32_t num_tiles = output.volume() / TILE_HW;
    uint32_t row_size_datum =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_rows =  a.layout() == Layout::ROW_MAJOR ? a.shape()[2] : a.shape()[3];
    uint32_t num_rows_padded = ceil((double) num_rows / (double) TILE_HEIGHT) * TILE_HEIGHT;
    assert(row_size_datum % TILE_WIDTH == 0);
    uint32_t num_2d_faces = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] : a.shape()[0] * a.shape()[2];
    uint32_t row_size_bytes = row_size_datum * 2; // Assuming bfloat16 dataformat

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = row_size_datum / 32;
    assert(num_input_tiles > 0);
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = row_size_datum / 32;

    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t zero_buffer_l1_addr = 600 * 1024;
    auto zero_buffer_l1 = tt_metal::CreateL1Buffer(program, device, core, row_size_bytes, zero_buffer_l1_addr);

    // Reader compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(row_size_bytes)) == floor(log2(row_size_bytes)));
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(),
                                            num_2d_faces,
                                            num_rows,
                                            num_rows_padded,
                                            row_size_bytes,
                                            zero_buffer_l1_addr};
    DataMovementKernelArgs *compile_time_args;
    if (stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(row_size_bytes));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {1});
    } else {
        compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {0});
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_pad_rows.cpp",
        core,
        compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint32_t((num_rows_padded/TILE_HEIGHT) * num_2d_faces),
        uint32_t(row_size_datum / TILE_WIDTH)
    };

    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        eltwise_unary_args,
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
        (uint32_t) (output.shape()[0] * output.shape()[1] * output.shape()[2] * output.shape()[3] / TILE_HW)}
    );
    std::vector<uint32_t> zero_buffer_stick(row_size_datum, 0);
    tt_metal::WriteToDeviceL1(device, core, zero_buffer_stick, zero_buffer_l1_addr);
    tt_metal::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}
vector<uint32_t> compute_conv_as_mm_shape_(vector<int> shape, vector<int> conv_params) {
    int conv_input_x = shape[2];
    int conv_input_y = shape[1];
    int conv_input_z = shape[0];
    int R = conv_params[0];
    int S = conv_params[1];
    int U = conv_params[2];
    int V = conv_params[3];
    int Pad_H = conv_params[4];
    int Pad_W = conv_params[5];
    int conv_output_h = ((conv_input_x - R + (2 * Pad_H)) / U) + 1;
    int conv_output_w = ((conv_input_y - S + (2 * Pad_W)) / V) + 1;
    std::cout << "conv_input_x=" << conv_input_x << std::endl;
    std::cout << "conv_input_y=" << conv_input_y << std::endl;
    std::cout << "conv_input_z=" << conv_input_z << std::endl;
    std::cout << "conv_output_h=" << conv_output_h << std::endl;
    std::cout << "conv_output_w=" << conv_output_w << std::endl;
    // pad height
    uint32_t num_rows = (uint32_t) conv_output_h*conv_output_w;
    uint32_t num_rows_padded = (uint32_t) (ceil((double) num_rows / (double) TILE_HEIGHT ) * TILE_HEIGHT);
    uint32_t num_cols = conv_input_z*R*S;
    uint32_t num_cols_padded = (uint32_t) (ceil((double) num_cols / (double) TILE_WIDTH ) * TILE_HEIGHT);
    return {1,num_rows_padded, num_cols_padded};
}
Tensor tilize_conv_activation(const Tensor &a, vector<int> conv_params, int conv_output_channels) {
    TT_ASSERT(a.layout() == Layout::CHANNELS_LAST, "Conv activation should be in channels last layout");

    // Compute the 2d matrix shape
    vector<int> shape = {(int)a.shape()[1], (int)a.shape()[2], (int)a.shape()[3]};

    auto matrix_shape = compute_conv_as_mm_shape_(shape , conv_params);
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];
    assert(num_rows > 0);
    assert(num_cols > 0);
    auto Ha = num_rows;
    auto Wa = num_cols;
    auto Wb = conv_output_channels;
    // Convert tensor dims to tile dims
    uint32_t Hat = Ha / TILE_HEIGHT;
    uint32_t Wat = Wa / TILE_WIDTH;
    uint32_t Wbt = Wb / TILE_WIDTH;
    std::cout << "Hat(M in tiles)=" << Hat << std::endl;
    std::cout << "Wat(K in tiles)=" << Wat << std::endl;
    std::cout << "Wbt(N in tiles)=" << Wbt << std::endl;
    // compute block info
    auto [num_blocks, out_subblock_h, out_subblock_w, report_string] = compute_conv_op_block_info(Hat, Wat, Wbt);
    assert(report_string == "pass");
    std::cout << "num block=" << num_blocks << std::endl;
    // in0 block info
    uint32_t in0_block_w = Wat / num_blocks; // Two blocks in the W dimension
    uint32_t in0_block_w_datums = Wa / num_blocks;
    std::pair<vector<int>,vector<int>> block_info;
    block_info.first = {0,1,2};
    block_info.second = {(int)num_rows, (int)in0_block_w_datums};

    DataTransformations * dtx = conv_transform(shape, conv_params, block_info);
    // copy transfer addresses into a vector
    std::vector<uint32_t> address_map;
    uint32_t t_bytes = 0;

    // Generate address map for reader kernel
    assert(dtx->transformations.size() == 2);
    assert(dtx->transformations.back()->groups[0]->transfers.size() > 0);
    uint32_t block_size_bytes = num_rows * in0_block_w_datums * 2;
    uint32_t b_bytes = 0;
    uint32_t n_blocks = 0;
    uint32_t block_start_address = 0;
    uint32_t next_block_start_index = 0;
    uint32_t num_reads_current_block = 0;
    address_map.push_back(0); // Update this value with number of reads for first block
    for(auto transfer : dtx->transformations.back()->groups[0]->transfers){
        assert(n_blocks < num_blocks);
        bool dst_address_in_block = (uint32_t) transfer->dst_address*2 >= block_start_address;
        bool dst_read_in_block = (uint32_t) transfer->dst_address*2 - block_start_address + transfer->size*2 <= block_size_bytes;
        if(!dst_read_in_block || !dst_address_in_block) {
            std::cout << "dst_address_in_block=" << dst_address_in_block << std::endl;
            std::cout << "dst_read_in_block=" << dst_read_in_block << std::endl;
            std::cout << "n_blocks=" << n_blocks << std::endl;
            std::cout << "block_start_address=" << block_start_address << std::endl;
            std::cout << "dst_address=" << transfer->dst_address*2 << std::endl;
            std::cout << "dst_size=" << transfer->size*2 << std::endl;
            std::cout << "block_size_bytes=" << block_size_bytes << std::endl;
        }
        assert(dst_read_in_block);
        assert(dst_address_in_block);
        if(address_map.size()==0) {
            std::cout << "in conv op" << std::endl;
            std::cout << "src=" << transfer->src_address*2 << std::endl;
            std::cout << "dst=" << transfer->dst_address*2 << std::endl;
            std::cout << "rs=" << transfer->size*2 << std::endl;
            std::cout << "pad=" << transfer->pad << std::endl;
        }
        assert(transfer->size*2 % 32 == 0);
        assert(transfer->src_address*2 % 32 == 0);
        assert(transfer->dst_address*2 % 32 == 0);
        address_map.push_back(transfer->src_address*2);
        address_map.push_back(transfer->dst_address*2);
        address_map.push_back(transfer->size*2);
        address_map.push_back(transfer->pad);
        num_reads_current_block++;
        t_bytes += transfer->size*2;
        b_bytes += transfer->size*2;
        if(b_bytes == block_size_bytes) {
            address_map[next_block_start_index] = num_reads_current_block;
            next_block_start_index = address_map.size();
            block_start_address = t_bytes;
            b_bytes = 0;
            n_blocks++;
            if (n_blocks != num_blocks) {
                address_map.push_back(0); // This value will be updated once we have pushed all entries for the next block bytes
            }
            num_reads_current_block = 0;
        }
    }
    uint32_t total_bytes = num_rows * num_cols * 2; // 2 for bfloat16
    assert(b_bytes == 0);
    assert(n_blocks == num_blocks);
    assert(total_bytes == t_bytes);
    assert(total_bytes % num_blocks == 0);
    uint32_t in0_block_size_bytes = total_bytes / num_blocks;
    assert(in0_block_size_bytes == block_size_bytes);

    tt_metal::Program *program = new tt_metal::Program();
    tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    assert(num_cols % TILE_WIDTH == 0);
    assert(num_rows % TILE_HEIGHT == 0);
    uint32_t num_tiles_c = num_cols / TILE_WIDTH;
    uint32_t num_tiles_r = num_rows / TILE_HEIGHT;
    uint32_t num_tiles = num_tiles_r * num_tiles_c;
    uint32_t block_size_tiles = num_tiles / num_blocks;
    assert(block_size_tiles * single_tile_size == block_size_bytes);

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> output_shape = {1, 1, num_rows, num_cols};
    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device, MemoryConfig({false,1}));

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();


    uint32_t ouput_cb_index = 0; // output operands start at index 16
    uint32_t output_cb_addr = 220 * 1024;
    uint32_t num_tiles_cb = num_tiles_r * in0_block_w * 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_tiles_cb,
        num_tiles_cb * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );


    auto l1_b0 = tt_metal::CreateL1Buffer(program, device, core, address_map.size() * sizeof(uint32_t));
    uint32_t address_map_l1_addr =l1_b0->address();
    vector<uint32_t> reader_kernel_args = {num_blocks,
                                            src0_dram_buffer->address(),
                                            (uint32_t)dram_src0_noc_xy.x,
                                            (uint32_t)dram_src0_noc_xy.y,
                                            block_size_tiles,
                                            address_map_l1_addr,
                                            block_size_bytes
                                            };

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_dtx.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_blocked.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles_r), // per_core_block_cnt
        uint32_t(num_tiles_c) // per_core_block_tile_cnt
    };
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        core,
        eltwise_unary_args,
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
    // std::vector<uint32_t> zero_buffer_stick(num_cols, 0);
    // tt_metal::WriteToDeviceL1(device, core, zero_buffer_stick, zero_buffer_l1_addr);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );
    assert(block_size_bytes == num_tiles_r * in0_block_w * 2048);
    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (uint32_t) dram_dst_noc_xy.x,
        (uint32_t) dram_dst_noc_xy.y,
        num_blocks,
        num_tiles_r * in0_block_w,
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
