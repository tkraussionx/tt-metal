#include <cstdint>

#include "compute_hlk_api.h"

struct hlk_args_t {
    int batch_size;

    // per-batch params
    int input_block_inner_dim;
    int per_block_r_tiles;
    int per_iteration_r_tiles;
    int per_block_c_tiles;
    int input_block_cnt;
    int in0_block_tile_cnt;
    int out_block_tile_cnt;
    int total_out_tile_cnt;
    int num_iterations;

    // shifting
    int col_tile_count = 0; // number of tiles in (post-transposed, flattened) X dimension
    int row_tile_count = 0; // number of tiles in (post-transposed, flattened) Y dimension - not needed any more
    int original_x = 0; // number of rows in original X dimension before flattening
    int original_y = 0; // number of rows in original Y dimension before flattening
    int stride = 1;
    int stride_offset = 0; // where to start striding from

    constexpr static std::uint32_t MAX_WEIGHT_Z = 8;  // Max of 8 inputs
    constexpr static std::uint32_t kTileHeight = 32;
    int pack_row_shift_x[MAX_WEIGHT_Z] = {0}; // shift packing by this many rows. Can be negative.
    int initial_rd_ptr[MAX_WEIGHT_Z] = {0}; // initial read state for the shifted packing
    int initial_x[MAX_WEIGHT_Z] = {0};
    int initial_y[MAX_WEIGHT_Z] = {0};
    int initial_padding[MAX_WEIGHT_Z] = {0};
};

//
// In order to make shifted packing easier, we're going to reorder destination writing to be
// column major instead of row major. The packing will still be row-major to keep things consistent... but column
// major order in dest allows us to traverse each column in order without having to skip tiles.
//

void hlk_main(tt_core *core_ptr, const hlk_args_t *args)
{
    // Pack state per column of data
    hlk_pack_shifted_state_t pack_state[2 * args->batch_size];

    // Global pack params
    hlk_pack_shifted_params_t pack_params;
    pack_params.original_x = args->original_x;
    pack_params.original_y = args->original_y;
    pack_params.stride = args->stride;
    pack_params.stride_offset = args->stride_offset;
    pack_params.valid_row_count = args->per_iteration_r_tiles * hlk_args_t::kTileHeight;
    pack_params.relu = false;

    hlk_pack_shifted_init(core_ptr, &pack_params);

    int total_packed_tiles[hlk_args_t::MAX_WEIGHT_Z] = {0};

    int initial_padding[hlk_args_t::MAX_WEIGHT_Z];
    __builtin_memcpy(initial_padding, args->initial_padding, sizeof(initial_padding));

    int dbl_pack_indices[hlk_args_t::MAX_WEIGHT_Z] = {0};

    for (int iteration = 0; iteration < args->num_iterations; iteration++) {
        int current_column_count = 0;  // accumulate columns over batch

        hlk_acquire_dst(core_ptr, DstMode::Full);

        hlk_wait_tiles(core_ptr, HlkOperand::in0, args->in0_block_tile_cnt);

        int in0_block_tile_index = 0;
        for (int r = 0; r < args->per_iteration_r_tiles; ++r) {
            int dst_tile_index = r;
            for (int c = 0; c < args->per_block_c_tiles; ++c) {
                hlk_load_mm_partial_to_dst(core_ptr, HlkOperand::in0, in0_block_tile_index, dst_tile_index);
                dst_tile_index += args->per_iteration_r_tiles;
                in0_block_tile_index++;
            }
        }

        for (int b = 0; b < args->batch_size; b++) {
            int &dbl_pack_index = dbl_pack_indices[b];
            int pack_index = b * 2 + dbl_pack_index;

            // Pack out
            if (iteration == 0) {
                hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0 + b, args->col_tile_count);

                // Set initial state
                pack_state[pack_index].current_rd_ptr = args->initial_rd_ptr[b];
                pack_state[pack_index].current_wr_ptr = 0;
                pack_state[pack_index].current_x = args->initial_x[b];
                pack_state[pack_index].current_y = args->initial_y[b];
                pack_state[pack_index].partial_tile = false;

                pack_state[pack_index + 1].current_rd_ptr = args->initial_rd_ptr[b];
                pack_state[pack_index + 1].current_wr_ptr = 0;
                pack_state[pack_index + 1].current_x = args->initial_x[b];
                pack_state[pack_index + 1].current_y = args->initial_y[b];
                pack_state[pack_index + 1].partial_tile = false;
            }
            else
            {
                int valid_row_count = args->per_iteration_r_tiles * hlk_args_t::kTileHeight;
                pack_state[pack_index].current_rd_ptr =
                    pack_state[pack_index].current_rd_ptr - valid_row_count;  // keep the roll-over
            }

            pack_params.row_shift_x = args->pack_row_shift_x[b];
            pack_params.final_iteration = (iteration == args->num_iterations - 1);

            int column = 0;
            int baseline_row = args->per_iteration_r_tiles * hlk_args_t::kTileHeight;

            // Pack until we reach end of dest, or if this is the final iteration, until we've generated enough tiles.
            while ((!pack_params.final_iteration && (pack_state[pack_index].current_rd_ptr < baseline_row)) ||
                   (pack_params.final_iteration && (total_packed_tiles[b] < args->total_out_tile_cnt))) {
                pack_params.valid_row_count = baseline_row;
                pack_params.column_number = current_column_count + column;
                pack_params.initial_padding = initial_padding[b];

                hlk_pack_shifted_state_t tmp_pack_state = pack_state[pack_index];

                hlk_pack_shifted_tile_to_stream(core_ptr, &pack_params, &tmp_pack_state, HlkOperand::out0 + b, column);

                int next_dbl_pack_index = ((dbl_pack_index + 1) & 1);
                if (column == 0) {
                    pack_state[b * 2 + next_dbl_pack_index] = tmp_pack_state;
                }

                baseline_row += args->per_iteration_r_tiles * hlk_args_t::kTileHeight;
                pack_state[pack_index].current_rd_ptr += args->per_iteration_r_tiles * hlk_args_t::kTileHeight;

                column++;
                if (column >= args->col_tile_count) {
                    column = 0;
                    if (!tmp_pack_state.partial_tile) {
                        hlk_push_tiles(core_ptr, HlkOperand::out0 + b, args->col_tile_count);  // FIXME: check if args->args->col_tile_count == args->out_block_tile_cnt
                        total_packed_tiles[b] += args->col_tile_count;
                        if (total_packed_tiles[b] == args->total_out_tile_cnt) {
                            break;  // all done
                        }
                        hlk_wait_for_free_tiles(core_ptr, HlkOperand::out0 + b, args->col_tile_count);  // args->args->col_tile_count == args->out_block_tile_cnt???
                    }
                    baseline_row = args->per_iteration_r_tiles * hlk_args_t::kTileHeight;
                    initial_padding[b] -= hlk_args_t::kTileHeight;
                    dbl_pack_index = next_dbl_pack_index;
                    pack_index = dbl_pack_index + b * 2;
                }
            }

            current_column_count += args->col_tile_count;
        }

        hlk_pop_tiles(core_ptr, HlkOperand::in0, args->in0_block_tile_cnt);

        hlk_release_dst(core_ptr, DstMode::Full);
    }
}
