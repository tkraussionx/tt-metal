// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "debug/dprint.h"

// inline void print_cb_details(uint32_t cb_id) {
//     DPRINT << "cb_id " << cb_id << ": { "
//             << "size: " << cb_interface[cb_id].fifo_size << ", "
//             << "limit: " << cb_interface[cb_id].fifo_limit << ", "
//             << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
//             << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
//             << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
//             << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
//             << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL();
// }

void kernel_main() {

    uint32_t i = 0;

    uint32_t local_untilized_data_start_offset_in_reshard_buffer = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_untilized_data_num_blocks_of_tile_height = get_arg_val<uint32_t>(i); i+=1;
    uint32_t local_untilized_data_num_rows_in_last_block = get_arg_val<uint32_t>(i); i+=1;

    uint32_t num_output_tiles = get_arg_val<uint32_t>(i); i+=1;

    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    constexpr uint32_t untilized_resharded_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t final_tilize_output_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t df_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t input_shard_width_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t log_base_2_of_input_shard_width_bytes = get_compile_time_arg_val(4);

    // print_cb_details(untilize_cb_index);
    // print_cb_details(untilize_downsampled_cb_index);
    // print_cb_details(final_tilize_output_cb_index);

    // // checking halo address
    // uint32_t input_cb_address = get_write_ptr(0);
    // DPRINT << "shard_input_address" << "=" << input_cb_address << ENDL();
    // DPRINT << "halo_prev_start_addr" << "=" << halo_prev_start_addr << ENDL();
    // DPRINT << "halo_prev_addr_offset" << "=" << halo_prev_addr_offset << ENDL();
    // DPRINT << "halo_prev_read_offset" << "=" << halo_prev_read_pattern_offset << ENDL();
    // DPRINT << "local_read_offset" << "=" << local_read_pattern_offset << ENDL();
    // DPRINT << "halo_prev_size_bytes" << "=" << halo_prev_size_bytes << ENDL();

    uint32_t img_flat_h_idx = 0;
    uint32_t reader_pattern_l1_addr = get_write_ptr(reader_pattern_cb_index);
    volatile tt_l1_ptr std::uint32_t* reader_pattern = (volatile tt_l1_ptr uint32_t*)(reader_pattern_l1_addr);
    uint32_t reader_pattern_index = 0;

    if (halo_prev_read_enabled) {
        // Read # of row of tiles from previous core into local cb
        // noc_async_read_barrier
        // Push cb to compute for untilizing
        cb_reserve_back(halo_prev_input_cb_index, halo_prev_num_tiles);
        uint32_t halo_prev_cb_write_addr = get_write_ptr(halo_prev_input_cb_index);
        uint32_t halo_prev_addr = halo_prev_start_addr + halo_prev_addr_offset;
        noc_async_read(get_noc_addr(halo_prev_core_noc_x, halo_prev_core_noc_y, halo_prev_addr), halo_prev_cb_write_addr, halo_prev_size_bytes);
        noc_async_read_barrier();
        cb_push_back(halo_prev_input_cb_index, halo_prev_num_tiles);
    }

    if (halo_next_read_enabled) {
        // Read # of row of tiles from next core into local cb
        // noc_async_read_barrier
        // Push cb to compute for untilizing
        cb_reserve_back(halo_next_input_cb_index, halo_next_num_tiles);
        uint32_t halo_next_cb_write_addr = get_write_ptr(halo_next_input_cb_index);
        uint32_t halo_next_addr = halo_next_start_addr + halo_next_addr_offset;
        noc_async_read(get_noc_addr(halo_next_core_noc_x, halo_next_core_noc_y, halo_next_addr), halo_next_cb_write_addr, halo_next_size_bytes);
        noc_async_read_barrier();
        cb_push_back(halo_next_input_cb_index, halo_next_num_tiles);
    }

    if (halo_prev_read_enabled) {
        img_flat_h_idx = halo_prev_read_pattern_offset;
        // generate reader pattern - halo from prev core region
        reader_pattern_index = generate_reader_pattern_indices(image_height,
                                                                image_width,
                                                                stride_h,
                                                                stride_w,
                                                                img_flat_h_idx,
                                                                stride_h_x_image_width,
                                                                reader_pattern,
                                                                reader_pattern_index,
                                                                halo_prev_top_partial_middle_aligned_row_width,
                                                                halo_prev_skip_top_partial_middle_right_aligned_row,
                                                                halo_prev_top_partial_right_aligned_row_width,
                                                                halo_prev_skip_top_partial_right_aligned_row,
                                                                halo_prev_num_rows_top_partial_image,
                                                                halo_prev_num_skip_rows_top_partial_image,
                                                                halo_prev_num_full_images,
                                                                halo_prev_num_rows_bottom_partial_image,
                                                                halo_prev_num_skip_rows_bottom_partial_image,
                                                                halo_prev_bottom_partial_left_aligned_row_width,
                                                                halo_prev_skip_bottom_partial_left_aligned_row);
    }
    // halo and local data will be pushed to the same untilize cb
    // img flat h idx is index within the concatenated untilized halo and local data
    // compute reader pattern - local region
    img_flat_h_idx = local_read_pattern_offset;
    reader_pattern_index = generate_reader_pattern_indices(image_height,
                                                            image_width,
                                                            stride_h,
                                                            stride_w,
                                                            img_flat_h_idx,
                                                            stride_h_x_image_width,
                                                            reader_pattern,
                                                            reader_pattern_index,
                                                            local_top_partial_middle_aligned_row_width,
                                                            local_skip_top_partial_middle_right_aligned_row,
                                                            local_top_partial_right_aligned_row_width,
                                                            local_skip_top_partial_right_aligned_row,
                                                            local_num_rows_top_partial_image,
                                                            local_num_skip_rows_top_partial_image,
                                                            local_num_full_images,
                                                            local_num_rows_bottom_partial_image,
                                                            local_num_skip_rows_bottom_partial_image,
                                                            local_bottom_partial_left_aligned_row_width,
                                                            local_skip_bottom_partial_left_aligned_row);

    if (halo_next_read_enabled) {
        img_flat_h_idx = halo_next_read_pattern_offset;
        // generate reader pattern - halo from prev core region
        reader_pattern_index = generate_reader_pattern_indices(image_height,
                                                                image_width,
                                                                stride_h,
                                                                stride_w,
                                                                img_flat_h_idx,
                                                                stride_h_x_image_width,
                                                                reader_pattern,
                                                                reader_pattern_index,
                                                                halo_next_top_partial_middle_aligned_row_width,
                                                                halo_next_skip_top_partial_middle_right_aligned_row,
                                                                halo_next_top_partial_right_aligned_row_width,
                                                                halo_next_skip_top_partial_right_aligned_row,
                                                                halo_next_num_rows_top_partial_image,
                                                                halo_next_num_skip_rows_top_partial_image,
                                                                halo_next_num_full_images,
                                                                halo_next_num_rows_bottom_partial_image,
                                                                halo_next_num_skip_rows_bottom_partial_image,
                                                                halo_next_bottom_partial_left_aligned_row_width,
                                                                halo_next_skip_bottom_partial_left_aligned_row);
    }

    // wait for untilized blocks from compute, drop rows based on reader_pattern and push to untilize_downsampled_cb_index
    reader_pattern_index = 0;
    uint32_t untilize_block_offset = 0;
    cb_reserve_back(untilize_downsampled_cb_index, num_output_tiles);
    //DPRINT << "reserved tiles in untilize downsampled cb" << ENDL();
    uint32_t untilize_downsampled_cb_l1_write_addr = get_write_ptr(untilize_downsampled_cb_index);
    // num_untilized_input_blocks contains halo as well as local data
    for (uint32_t untilized_input_block_i = 0; untilized_input_block_i < num_untilized_input_blocks; untilized_input_block_i++) {
        cb_wait_front(untilize_cb_index, num_tiles_untilized_input_block); // 1 row of tiles
        uint32_t untilize_block_l1_read_addr = get_read_ptr(untilize_cb_index);
        // untilize block is only 1 row of tiles
        while(reader_pattern[reader_pattern_index] < 32 + untilize_block_offset) {
            // local read into untilized downsampled l1, read_address = l1_read_addr + (reader_pattern[reader_pattern_index] * conv_act_c_bytes), read_size = conv_act_c_bytes, write_address = untilize_downsampled_cb_l1_write_addr
            // reader pattern indices are for the whole sharded input. Need to subtract by block offset to read from the block which is 1 row of tiles.
            uint32_t read_address = untilize_block_l1_read_addr + ((uint32_t) ((int32_t) reader_pattern[reader_pattern_index] - (int32_t) untilize_block_offset) << log_base_2_of_conv_act_size_c_bytes);
            noc_async_read(get_noc_addr(read_address), untilize_downsampled_cb_l1_write_addr, conv_act_size_c_bytes);
            untilize_downsampled_cb_l1_write_addr += conv_act_size_c_bytes;
            reader_pattern_index += 1;
        }
        //DPRINT << "waiting noc barrier" << ENDL();
        noc_async_read_barrier();
        //DPRINT << "downsampled 1 row of untilized cb buffer" << ENDL();
        cb_pop_front(untilize_cb_index, num_tiles_untilized_input_block);
        untilize_block_offset += 32;
    }
    cb_push_back(untilize_downsampled_cb_index, num_output_tiles); // to be tilized by compute kernel
    // wait for tilized cb
    cb_wait_front(final_tilize_output_cb_index, num_output_tiles); // wait we dont need this.. sharded output.
}
