// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"

inline void pad_l1_buffer_with_zeroes(uint32_t l1_addr, uint32_t pad_size_bytes) {
    volatile std::uint32_t* dst = reinterpret_cast<volatile std::uint32_t*>(l1_addr);
    volatile std::uint32_t* end_dst = dst + (pad_size_bytes >> 2);  // Divide by 4 using right shift

    while (dst < end_dst) {
        *dst++ = 0;
    }

    uint32_t remainder = pad_size_bytes & 0x3;  // Get the remainder using bitwise AND
    if (remainder != 0) {
        volatile std::uint8_t* byte_dst = reinterpret_cast<volatile std::uint8_t*>(dst);
        for (uint32_t i = 0; i < remainder; ++i) {
            *byte_dst++ = 0;
        }
    }
}

void kernel_main() {
    uint32_t i = 0;
    uint32_t act_addr_dram_base  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_dram_noc_x = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_dram_noc_y = get_arg_val<uint32_t>(i); i+=1;

    uint32_t conv_act_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_act_size_c_ = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weight_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_h_ = get_arg_val<uint32_t>(i); i+=1;
    uint32_t stride_w_ = get_arg_val<uint32_t>(i); i+=1;
    uint32_t pad_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t pad_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t conv_output_size_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_act_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_blocks_weight_w = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_groups = get_arg_val<uint32_t>(i); i+=1;

    uint32_t act_matrix_height_unpadded = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width_unpadded = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_height = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_height_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_matrix_width_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_h_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_w_datums = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_h_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_w_ntiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t act_block_num_tiles = get_arg_val<uint32_t>(i); i+=1;
    uint32_t src_dram_act_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t dst_l1_act_buffer_size_bytes = get_arg_val<uint32_t>(i); i+=1;
    uint32_t n_start = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_h_start = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_w_start = get_arg_val<uint32_t>(i); i+=1;
    uint32_t total_h_start = get_arg_val<uint32_t>(i); i+=1;

    uint32_t first_partial_right_aligned_row_width = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_partial_right_aligned_row  = get_arg_val<uint32_t>(i); i+=1;
    uint32_t first_partial_image_num_rows          = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_first_partial_image_row    = get_arg_val<uint32_t>(i); i+=1;
    uint32_t num_full_images                       = get_arg_val<uint32_t>(i); i+=1;
    uint32_t skip_after_full_image                 = get_arg_val<uint32_t>(i); i+=1;
    uint32_t last_partial_image_num_rows           = get_arg_val<uint32_t>(i); i+=1;
    uint32_t last_partial_left_aligned_row_width   = get_arg_val<uint32_t>(i); i+=1;

    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_size_w_ = get_compile_time_arg_val(3);
    constexpr uint32_t conv_output_w_last_index = get_compile_time_arg_val(4) - 1;
    constexpr uint32_t conv_act_size_c_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t log_base_2_of_conv_act_size_c_bytes = get_compile_time_arg_val(6);

    constexpr uint32_t cb_id_act = 0;
    constexpr uint32_t tile_size_pow2_exponent = 11;
    const DataFormat data_format = get_dataformat(cb_id_act);
    const InterleavedPow2AddrGenFast<act_in_dram> s_act = {
        .bank_base_address = act_addr_dram_base,
        .log_base_2_of_page_size = log_base_2_of_conv_act_size_c_bytes
    };

    // Assumptions. Must be true. Validate on host.
    // assert(act_block_w_datums == C * weight_size_w)
    // assert(num_blocks_act_w == weight_size_h)
    // assert(act_block_w_datums % C == 0)
    // assert(act_block_w_datums % 32 == 0)
    // assert(act_block_h_datums % 32 == 0)
    // assert(act_block_h_ntiles == act_block_h_datums/32)
    // assert(act_block_w_ntiles == act_block_w_datums/32)
    // assert(act_block_num_tiles == (act_block_h_datums * act_block_w_datums)/1024)


    //DPRINT << "partial right aligned width: " << first_partial_right_aligned_row_width << ENDL();
    //DPRINT << "--- skip after: " << skip_after_partial_right_aligned_row << ENDL();
    //DPRINT << "first partial image rows: " << first_partial_image_num_rows << ENDL();
    //DPRINT << "--- skip after: " << skip_after_first_partial_image_row << ENDL();
    //DPRINT << "full images: " << num_full_images << ENDL();
    //DPRINT << "--- skip after: " << skip_after_full_image << ENDL();
    //DPRINT << "last partial image rows: " << last_partial_image_num_rows << ENDL();
    //DPRINT << "partial left aligned width: " << last_partial_left_aligned_row_width << ENDL();


    // DUMMY LOOP TO FILL READER INDICES
    constexpr uint32_t cb_reader_indices = tt::CB::c_intermed5;
    volatile tt_l1_ptr uint32_t* reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    uint32_t weights_top_left_corner_idx = 0;
    uint32_t reader_idx = 0;

    // First partial right-aligned row
    for (uint32_t k = 0; k < first_partial_right_aligned_row_width; k++) {
        reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
    }
    weights_top_left_corner_idx += skip_after_partial_right_aligned_row; // Skip padded width

    // First partial image
    for (uint32_t j = 0; j < first_partial_image_num_rows; j++) {
        for (uint32_t k = 0; k < conv_act_size_w_; k++) {
            reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
        }
        weights_top_left_corner_idx += weight_size_w - 1;
    }
    weights_top_left_corner_idx += skip_after_first_partial_image_row; // Skip padded rows

    // Full images
    for (uint32_t i = 0; i < num_full_images; i++) {
        for (uint32_t j = 0; j < conv_act_size_h; j++) {
            for (uint32_t k = 0; k < conv_act_size_w; k++) {
                reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
            }
            weights_top_left_corner_idx += weight_size_w - 1;
        }
        weights_top_left_corner_idx += skip_after_full_image; // Skip padded rows
    }

    // Last partial image
    for (uint32_t j = 0; j < last_partial_image_num_rows; j++) {
        for (uint32_t k = 0; k < conv_act_size_w; k++) {
            reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
        }
        weights_top_left_corner_idx += weight_size_w - 1;
    }

    // Last partial left-alighted row
    for (uint32_t k = 0; k < last_partial_left_aligned_row_width; k++) {
        reader_indices_ptr[reader_idx++] = weights_top_left_corner_idx++;
    }

    //DPRINT << "num indices: " << reader_idx << ENDL();
    //for (uint32_t i = 0; i < reader_idx; i++) {
        //DPRINT << reader_indices_ptr[i] << ENDL();
    //}

    uint32_t reader_offset = 0;
    uint32_t dummy_read_addr = get_read_ptr(tt::CB::c_in3); // TODO: Delete me
    for (uint32_t channel_stick_h = 0; channel_stick_h < weight_size_h; channel_stick_h++) {
        for (uint32_t channel_stick_w = 0; channel_stick_w < weight_size_w; channel_stick_w++) {
            // Reset reader_idx to finish act_block_h_datums
            reader_idx = 0;
            cb_reserve_back(cb_id_act, act_block_num_tiles);
            uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
            for (uint32_t bh = 0; bh < act_block_h_datums; bh++) {
                // local read from reader_indices_ptr[reader_idx + reader_offset];
                noc_async_read(get_noc_addr(dummy_read_addr), l1_write_addr_act, conv_act_size_c_bytes);
                l1_write_addr_act += conv_act_size_c_bytes;
                reader_idx++;
            }
            cb_push_back(cb_id_act, act_block_num_tiles);
            reader_offset++;
        }
        reader_offset += conv_act_size_w; // Assuming (weight_size_w - 1) / 2 == pad_w
    }


    /* OLD READER LOGIC
    for(uint32_t nbr = 0; nbr < num_blocks_weight_w; nbr++) {
        uint32_t out_h = out_h_start;
        uint32_t out_w = out_w_start;
        uint32_t out_h_reset = out_h_start;
        uint32_t out_w_reset = out_w_start;
        uint32_t total_h = total_h_start;
        uint32_t total_h_reset = total_h_start;
        uint32_t n = n_start;
        uint32_t n_reset = n_start;
        for(uint32_t nbh = 0; nbh < num_blocks_act_h; nbh++) {
            uint32_t in_h_offset_within_kernel_window = 0;
            for (uint32_t channel_stick_h = 0; channel_stick_h < weight_size_h; channel_stick_h++) {
                uint32_t in_w_offset_within_kernel_window = 0;
                for (uint32_t channel_stick_w = 0; channel_stick_w < weight_size_w; channel_stick_w++) {
                    out_h = out_h_reset;
                    out_w = out_w_reset;
                    total_h = total_h_reset;
                    n = n_reset;
                    cb_reserve_back(cb_id_act, act_block_num_tiles);
                    uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
                    uint32_t l1_addr_offset = 0;
                    for(uint32_t bh = 0; bh < act_block_h_datums; bh++) {
                        uint32_t in_h_offset = out_h * stride_h;
                        uint32_t in_w_offset = out_w * stride_w; // expect stride 1 or 2.. make this compile time args - also conv input width
                        uint32_t read_size_bytes = conv_act_size_c_bytes;

                        if (total_h < act_matrix_height_unpadded) {
                            uint32_t in_h = in_h_offset + in_h_offset_within_kernel_window;
                            uint32_t in_w = in_w_offset + in_w_offset_within_kernel_window;

                            if(in_h < pad_h || in_w < pad_w || in_h >= (conv_act_size_h + pad_h) || in_w >= (conv_act_size_w_ + pad_w)) {
                                // pad 0s in l1
                                uint32_t dst_addr = l1_write_addr_act + l1_addr_offset;
                                uint32_t pad_size_bytes = read_size_bytes;
                                pad_l1_buffer_with_zeroes(dst_addr, pad_size_bytes);
                            } else {
                                // read one channel from dram multi bank - row_id = channel_id
                                uint32_t in_h_raw = in_h - pad_h;
                                uint32_t in_w_raw = in_w - pad_w;
                                uint32_t channel_id = (n * conv_act_size_h * conv_act_size_w_) + (in_h_raw * conv_act_size_w_) + in_w_raw;
                                //DPRINT << "n=" << n << " h=" << in_h_raw << " w=" << in_w_raw << " conv_act_size_h=" << conv_act_size_h << " conv_act_size_w_=" << conv_act_size_w_ << ENDL();
                                uint32_t dst_addr = l1_write_addr_act + l1_addr_offset;
                                s_act.noc_async_read_page(channel_id, dst_addr);
                            }
                        } //else { DPRINT << "total_h here =" << total_h << ENDL();  } //do nothing. let garbage rows be in l1
                        l1_addr_offset += read_size_bytes;
                        if(out_w < conv_output_size_w - 1) {
                            out_w += 1;
                        } else {
                            out_w = 0;
                            if (out_h < conv_output_size_h - 1) {
                                out_h += 1;
                            } else if (total_h < act_matrix_height_unpadded){
                                // next image in batch
                                out_h = 0;
                                n += 1;
                            }
                        }
                        total_h += 1;
                    } // for block height
                    in_w_offset_within_kernel_window += 1;
                    //DPRINT << "waiting on read barrier" << ENDL();
                    noc_async_read_barrier();
                    //DPRINT << "done on read barrier" << ENDL();
                    cb_push_back(cb_id_act, act_block_num_tiles);
                } // for filter window width
                in_h_offset_within_kernel_window += 1;
            } // for filter window height
            out_h_reset = out_h;
            out_w_reset = out_w;
            total_h_reset = total_h;
            n_reset = n;
        } // for num of act blocks in height dim
    } // for num of weight blocks in width dim
    */
}
