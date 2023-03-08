#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int num_reductions;

    // per-batch params
    int num_input_blocks;
    int input_block_size;
    int input_block_shape_r;
    int input_block_shape_c;

    float coefficient;
};

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{
    hlk_reduce_tile_init(core_ptr);
    for (int reduction_index = 0; reduction_index < args->num_reductions; reduction_index++) {

        hlk_acquire_dst(core_ptr, DstMode::Full);

        // reduce across blocks in the col dim (they all reduce onto input_block_shape_r)
        for(int in_block_idx=0; in_block_idx < args->num_input_blocks; ++in_block_idx)
        {
            hlk_wait_tiles(core_ptr, HlkOperand::in0, args->input_block_size);

            int input_tile_index = 0;
            for(int r=0;r < args->input_block_shape_r; ++r)
            {
                // reduce a row within a block
                for(int c = 0;c < args->input_block_shape_c; ++c)
                {
                    int dst_tile_index = c;
                    hlk_reduce_tile(core_ptr, (int)ReduceFunc::Sum, (int)Dim::R, HlkOperand::in0, input_tile_index, dst_tile_index, args->coefficient);
                    input_tile_index++;
                }
            }
            hlk_pop_tiles(core_ptr, HlkOperand::in0, args->input_block_size);
        }

        // Pack out
        hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed1, args->input_block_shape_c);
        for (int dst_tile_index = 0; dst_tile_index < args->input_block_shape_c; ++dst_tile_index)
        {
            hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed1);
        }
        hlk_push_tiles(core_ptr, HlkOperand::intermed1, args->input_block_shape_c);

        hlk_release_dst(core_ptr, DstMode::Full);
    }

    hlk_add_tile_init(core_ptr);
    for (int reduction_index = 0; reduction_index < args->num_reductions; reduction_index++) {

        hlk_acquire_dst(core_ptr, DstMode::Full);

        hlk_wait_tiles(core_ptr, HlkOperand::intermed0, args->input_block_shape_c);
        hlk_wait_tiles(core_ptr, HlkOperand::intermed1, args->input_block_shape_c);

        for (int dst_tile_index = 0; dst_tile_index < args->input_block_shape_c; ++dst_tile_index)
        {
            hlk_add_tile(core_ptr, HlkOperand::intermed0, HlkOperand::intermed1, dst_tile_index, dst_tile_index, dst_tile_index);
        }

        hlk_pop_tiles(core_ptr, HlkOperand::intermed0, args->input_block_shape_c);
        hlk_pop_tiles(core_ptr, HlkOperand::intermed1, args->input_block_shape_c);

        // Pack out
        hlk_wait_for_free_tiles(core_ptr, HlkOperand::intermed0, args->input_block_shape_c);
        for (int dst_tile_index = 0; dst_tile_index < args->input_block_shape_c; ++dst_tile_index)
        {
            hlk_pack_tile_to_stream(core_ptr, dst_tile_index, HlkOperand::intermed0);
        }
        hlk_push_tiles(core_ptr, HlkOperand::intermed0, args->input_block_shape_c);

        hlk_release_dst(core_ptr, DstMode::Full);
    }
}
