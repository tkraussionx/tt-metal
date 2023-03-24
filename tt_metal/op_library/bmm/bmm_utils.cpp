#include "tt_metal/op_library/bmm/bmm_utils.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void create_CBs_for_fused_matmul(tt_metal::Program* program, tt_metal::Device* device, tt_xy_pair core, bool activations_rm, bool output_rm, uint32_t M, uint32_t N, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t num_bytes_for_df) {


    uint32_t single_tile_size = num_bytes_for_df * 1024;

    // Invariants
    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 150 * 1024;
    uint32_t cb0_tiles = M * in0_block_w * 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        src0_cb_addr,
        tt::DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    uint32_t src1_cb_addr = 300 * 1024;
    uint32_t cb1_tiles = N * in0_block_w * 2;
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        src1_cb_addr,
        tt::DataFormat::Float16_b
    );

    if (not activations_rm and not output_rm) { // no tilize, no untilize
        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = M * N;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );
    } else if (not activations_rm and output_rm) { // no tilize, just untilize

        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 500 * 1024;
        uint32_t num_output_tiles = M * N;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_index = 26;
        uint32_t reblock_cb_addr = 600 * 1024;
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            device,
            reblock_cb_index,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            reblock_cb_addr,
            tt::DataFormat::Float16_b
        );

    } else if (activations_rm and not output_rm) { // just tilize, no untilize

        uint32_t src0_tilized_index = 24;
        uint32_t src_0_tilized_cb_addr = 500 * 1024;
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_tilized_index,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            src_0_tilized_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = M * N;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t interm0_cb_index = 25;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );
    } else { // tilize activations and untilize output

        // Used for placing tilized activations
        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 500 * 1024;
        uint32_t num_output_tiles = M * N;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        // Used
        uint32_t interm1_cb_index = 25;
        uint32_t interm1_cb_addr = 600 * 1024;
        uint32_t interm1_cb_tiles = M * N;
        auto cb_interm1 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm1_cb_index,
            core,
            interm1_cb_tiles,
            interm1_cb_tiles * single_tile_size,
            interm1_cb_addr,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_index = 26;
        uint32_t reblock_cb_addr = 700 * 1024;
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            device,
            reblock_cb_index,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            reblock_cb_addr,
            tt::DataFormat::Float16_b
        );
    }
}

} // namespace tt_metal

} // namespace tt
