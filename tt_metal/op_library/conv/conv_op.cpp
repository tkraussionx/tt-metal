#include "tt_metal/op_library/conv/conv_op.hpp"
#include "tt_metal/op_library/bmm/bmm_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"
#include "tt_metal/impl/dtx/dtx.hpp"
#include "tt_metal/impl/dtx/dtx_passes.hpp"
//#include "test/tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "llrt/tt_debug_print_server.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {
// Allows support for tilizing A with DTX address map, untilize B, untilize output

// WORK IN PROGRESS
Tensor conv_as_large_bmm_single_block_single_core(const Tensor& a, const Tensor& b, bool untilize_out) {
    bool tilize_a = true;
    std::cout << "Untilize output? - " << untilize_out << std::endl;

    TT_ASSERT(a.layout() == Layout::CHANNELS_LAST, "Conv activation should be in channels last layout");

    //vector<int> shape = {5, 4,4};
    vector<int> shape = {(int) a.shape()[1], (int) a.shape()[2], (int) a.shape()[3]};

    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    bool pass = true;
    pass &= convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1(dtx_right);
    // Get the 2d matrix shape
    auto matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];
    pass &= row_major_memory_store(dtx_right);

    //cout << "\n\nDTX_RIGHT" << endl;
    //dtx_right->print();


    // Left side: AbstractTensor --> producer memory buffer
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx_left);

    //cout << "\n\nDTX_LEFT" << endl;
    //dtx_left->print();

    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();

    pass &= optimize_away_transpose(combined);
    //cout << "\n\nDTX_OPTIMIZED" << endl;
    //combined->print();

    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);
    //combined->print();
    // copy transfer addresses into a vector
    std::vector<uint32_t> address_map;
    uint32_t t_bytes = 0;
    for(auto transfer : combined->transformations.back()->groups[0]->transfers){
        address_map.push_back(transfer->src_address*2); // 2 for bfloat16
        address_map.push_back(transfer->dst_address*2);
        address_map.push_back(transfer->size*2);
        t_bytes += transfer->size*2;
    }
    uint32_t total_bytes = num_rows * num_cols * 2; // 2 for bfloat16
    assert(total_bytes == t_bytes);

    uint32_t Ba = 1;
    uint32_t Ca = 1;
    auto Ha = num_rows;
    auto Wa = num_cols;

    const auto [Bb, Cb, Hb, Wb] = b.shape();
    TT_ASSERT(Ha == 8 * TILE_HEIGHT, "For now, assuming practically hard-coded dimensions so that blocking makes sense");
    TT_ASSERT(Wa == 9 * TILE_WIDTH, "For now, assuming practically hard-coded dimensions so that blocking makes sense");
    TT_ASSERT(Wb == 4 * TILE_WIDTH, "For now, assuming practically hard-coded dimensions so that blocking makes sense");
    //TT_ASSERT(Hb == Wb, "For now, assuming practically hard-coded dimensions so that blocking makes sense");

    // Normal matrix shape checks
    TT_ASSERT(Ba == 1, "So far, large matmul op has only been tested for batch one.");
    TT_ASSERT(Ba == Bb, "Batch dimension needs to match");
    TT_ASSERT(Ca == Cb, "Channel dimension needs to match");
    TT_ASSERT(Wa == Hb, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(Ha % TILE_HEIGHT == 0, "Height of tensor a needs to be divisible by 32");
    TT_ASSERT(Wa % TILE_WIDTH == 0, "Width of tensor a needs to be divisible by 32");
    TT_ASSERT(Hb % TILE_HEIGHT == 0, "Height of tensor b needs to be divisible by 32");
    TT_ASSERT(Wb % TILE_WIDTH == 0, "Width of tensor b needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to large matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to large matmul need to be allocated in buffers on device!");

    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};
    tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});

    uint32_t single_tile_size = 2 * 1024; // TODO(agrebenisan): Refactor on df
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    // same condition as above, different message
    //TT_ASSERT(src0_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor a must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> cshape{Ba, Ca, Ha, Wb};
    // pad height
    cshape[2] = (uint32_t) (ceil((double) cshape[2] / (double) TILE_HEIGHT ) * TILE_HEIGHT);

    // For height padding in reader kernel
    uint32_t total_zeroes_bytes = (cshape[2] - Ha) * Wa * 2; // 2 for bfloat16
    uint32_t zero_buffer_size = l1_mem::address_map::ZEROS_SIZE;
    uint32_t num_bytes_of_zeroes_per_transfer = 0;
    uint32_t num_transfers_of_zeroes = 0;

    if(total_zeroes_bytes > zero_buffer_size) {
        num_bytes_of_zeroes_per_transfer = zero_buffer_size;
        assert(total_zeroes_bytes % zero_buffer_size == 0);
        num_transfers_of_zeroes = total_zeroes_bytes / zero_buffer_size;
    }
    else if(total_zeroes_bytes > 0) {
        num_bytes_of_zeroes_per_transfer = total_zeroes_bytes;
        num_transfers_of_zeroes = 1;
    }

    tt::tt_metal::Layout out_layout;
    if (untilize_out) {
        out_layout = tt::tt_metal::Layout::ROW_MAJOR;
    } else {
        out_layout = tt::tt_metal::Layout::TILE;
    }
    tt_metal::Tensor output = tt_metal::Tensor(
        cshape,
        a.dtype(),
        out_layout,
        device,
        {.interleaved = false, .dram_channel = 0});

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    uint32_t address_map_l1_addr = 800 * 1024;
    auto l1_b0 = tt_metal::CreateL1Buffer(program, device, core, address_map.size() * sizeof(uint32_t), address_map_l1_addr);

    {
        // Convert tensor dims to tile dims
        uint32_t B   = Ba;
        uint32_t Hat = ceil((double) Ha / (double) TILE_HEIGHT );
        uint32_t Wat = Wa / TILE_WIDTH;
        uint32_t Wbt = Wb / TILE_WIDTH;

        uint32_t out_subblock_h = 4;
        uint32_t out_subblock_w = 2;
        uint32_t in0_block_w = Wat;

        uint32_t in0_dram_addr = src0_dram_buffer->address();
        uint32_t in1_dram_addr = src1_dram_buffer->address();
        uint32_t out_dram_addr = dst_dram_buffer->address();

        auto in0_dram_noc_xy = src0_dram_buffer->noc_coordinates();
        auto in1_dram_noc_xy = src1_dram_buffer->noc_coordinates();
        auto out_dram_noc_xy = dst_dram_buffer->noc_coordinates();

        // NOC coordinates
        uint32_t in0_dram_noc_x = uint32_t(in0_dram_noc_xy.x);
        uint32_t in0_dram_noc_y = uint32_t(in0_dram_noc_xy.y);
        uint32_t in1_dram_noc_x = uint32_t(in1_dram_noc_xy.x);
        uint32_t in1_dram_noc_y = uint32_t(in1_dram_noc_xy.y);
        uint32_t out_dram_noc_x = uint32_t(out_dram_noc_xy.x);
        uint32_t out_dram_noc_y = uint32_t(out_dram_noc_xy.y);

        {
            create_CBs_for_fused_matmul(
                program,
                a.device(),
                core,
                tilize_a,
                untilize_out,
                Hat,
                Wbt,
                in0_block_w,
                out_subblock_h,
                2); // TODO(agrebenisan): fix df num bytes

std::cout << "Reader kernel args - " << std::endl;
std::cout << "in1 tiles " << Wbt * in0_block_w << std::endl;
std::cout << "in0 tiles " << Hat * in0_block_w << std::endl;
std::cout << "Num bytes of zeroes " << num_bytes_of_zeroes_per_transfer << std::endl;
std::cout << "Num transfers of zeroes " << num_transfers_of_zeroes << std::endl;

            vector<uint32_t> reader_rt_args = {
                // arguments for in1
                in1_dram_addr,
                in1_dram_noc_x,
                in1_dram_noc_y,
                Wbt * in0_block_w, // input 1 block num tiles
                Wbt * in0_block_w * single_tile_size, // input 1 block bytes
                // arguments for in0
                in0_dram_addr,
                in0_dram_noc_x,
                in0_dram_noc_y,
                Hat * in0_block_w, // input 0 block num tiles
                num_bytes_of_zeroes_per_transfer,
                num_transfers_of_zeroes,
                address_map_l1_addr,
                (uint32_t)address_map.size()
            };

            string writer_kernel;
            vector<uint32_t> writer_rt_args;
            if (untilize_out) {
                writer_kernel = "tt_metal/kernels/dataflow/writer_unary.cpp";
                writer_rt_args = {
                    out_dram_addr,
                    out_dram_noc_x,
                    out_dram_noc_y,
                    Hat * Wbt
                };
            } else {
                writer_kernel = "tt_metal/kernels/dataflow/writer_unswizzle.cpp";
                writer_rt_args = {
                    out_dram_addr,
                    out_dram_noc_x,
                    out_dram_noc_y,
                    out_subblock_h, // num tiles per sub block m
                    out_subblock_w, // num tiles per sub block n
                    Hat / out_subblock_h, // num sub blocks m
                    Wbt / out_subblock_w, // num sub blocks n
                    out_subblock_w * single_tile_size * (Wbt / out_subblock_w), // bytes offset to next row within sub-block
                    out_subblock_h * out_subblock_w * single_tile_size * (Wbt / out_subblock_w), // bytes offset to next row of sub-blocks
                    out_subblock_w * single_tile_size
                    };
            }

            auto reader = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_large_single_matmul_block_with_address_map.cpp",
                core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

            auto writer = tt_metal::CreateDataMovementKernel(
                program,
                writer_kernel,
                core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

            uint32_t num_blocks = (Wat / in0_block_w);
            uint32_t in0_num_subblocks = (Hat / out_subblock_h);
            uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
            uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

            uint32_t in1_num_subblocks = (Wbt / out_subblock_w);
            uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w*in1_num_subblocks;
            uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

            uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
            uint32_t in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;

            vector<uint32_t> compute_kernel_args = {
                in0_block_w,
                in0_num_subblocks,
                in0_block_num_tiles,
                in0_subblock_num_tiles,
                in0_subblock_h,

                in1_num_subblocks,
                in1_block_num_tiles,
                in1_per_core_w,

                num_blocks,

                out_subblock_h,
                out_subblock_w,
                out_subblock_num_tiles,

                tilize_a,
                untilize_out
            };

            tt_metal::ComputeKernelArgs *bmm_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);
            bool fp32_dest_acc_en = false;
            bool math_approx_mode = false;
            auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
                program,
                "kernels/compute/matmul_large_block_3m.cpp",
                core,
                bmm_args,
                MathFidelity::HiFi4,
                fp32_dest_acc_en,
                math_approx_mode
            );

            tt_metal::WriteRuntimeArgsToDevice(
                device, reader, core,
                reader_rt_args
            );

            tt_metal::WriteRuntimeArgsToDevice(
                device, writer, core,
                writer_rt_args
            );

            pass &= tt_metal::CompileProgram(device, program, false);
            pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
            tt_metal::WriteToDeviceL1(device, core, address_map, address_map_l1_addr);
        }


        // read_trisc_debug_mailbox(device->cluster(), 0, {1, 1}, 0);

        pass &= tt_metal::LaunchKernels(device, program);
    }

    TT_ASSERT(pass);

    return output;

    }
}

}
