// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"


#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"


inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++ page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}
#endif


void kernel_main() {
    // This writer is for output tensor in tile format
    constexpr bool out_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_weight = get_compile_time_arg_val(2);

    constexpr uint32_t num_blocks_weight_h = get_compile_time_arg_val(5);
    constexpr uint32_t weight_block_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t weight_block_height_num_outer = get_compile_time_arg_val(7);
    constexpr uint32_t weight_block_height_ntiles = get_compile_time_arg_val(8);
    constexpr uint32_t weight_block_width_ntiles = get_compile_time_arg_val(9);
    constexpr uint32_t weight_stride_h = get_compile_time_arg_val(10);
    constexpr uint32_t weight_next_block_stride_h = get_compile_time_arg_val(11);
    constexpr uint32_t weight_next_block_stride_w = get_compile_time_arg_val(12);

    // Bias arg. Unused if bias fusion is not enabled.
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(13);

    constexpr uint32_t out_next_tile_stride_h = get_compile_time_arg_val(14);
    constexpr uint32_t out_next_tile_stride_w = get_compile_time_arg_val(15);
    constexpr uint32_t out_next_subblock_stride_h = get_compile_time_arg_val(16);
    constexpr uint32_t out_next_subblock_stride_w = get_compile_time_arg_val(17);
    constexpr uint32_t out_next_block_stride_h = get_compile_time_arg_val(18);
    constexpr uint32_t out_next_block_stride_w = get_compile_time_arg_val(12); // == weight_next_block_stride_w
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(19);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(20);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(21);
    constexpr uint32_t out_num_subblocks_h = get_compile_time_arg_val(22);
    constexpr uint32_t out_num_subblocks_w = get_compile_time_arg_val(23);
    constexpr uint32_t out_num_blocks_h = get_compile_time_arg_val(24);
    constexpr uint32_t out_num_blocks_w = get_compile_time_arg_val(25);
    constexpr uint32_t out_block_height_num_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t out_height_num_tiles = get_compile_time_arg_val(27);
    constexpr uint32_t out_width_num_tiles = get_compile_time_arg_val(28);

    constexpr uint32_t out_addr = get_compile_time_arg_val(29);

    #ifdef UNPAD_UNTILIZE_OUT
    constexpr uint32_t out_block_width_ntiles = get_compile_time_arg_val(32);
    constexpr uint32_t out_block_width_padded_bytes = get_compile_time_arg_val(33);
    constexpr uint32_t out_block_width_bytes = get_compile_time_arg_val(34);
    constexpr uint32_t untilized_padded_out_cb = get_compile_time_arg_val(35);
    #endif


    uint32_t i = 0;
    i+=19;
    uint32_t out_start_tile_id = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_start_tile_id_h = get_arg_val<uint32_t>(i); i+=1;
    uint32_t out_start_tile_id_w = get_arg_val<uint32_t>(i); i+=1;
    i+=10;
    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;
    if(noop) {
        return;
    }

    // mcast args
    uint32_t weights_mcast_sender_noc_x           = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weights_mcast_sender_noc_y           = get_arg_val<uint32_t>(i); i+=1;
    uint32_t weights_mcast_sender_semaphore_addr    = get_semaphore(get_arg_val<uint32_t>(i)); i+=1;
    uint32_t weights_mcast_receiver_semaphore_addr  = get_semaphore(get_arg_val<uint32_t>(i)); i+=1;
    uint32_t out_aligned_page_size                  = get_arg_val<uint32_t>(i); i+=1;

    volatile tt_l1_ptr uint32_t* weights_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights_mcast_receiver_semaphore_addr);
    uint64_t weights_mcast_sender_semaphore_noc_addr = get_noc_addr(weights_mcast_sender_noc_x, weights_mcast_sender_noc_y, weights_mcast_sender_semaphore_addr);

    const uint32_t tile_nbytes = get_tile_size(cb_id_out0);

    constexpr uint32_t tile_size_pow2_exponent = 11;    // == 2^11 = 2048 = 2 * 32 * 32 (assuming dtype = 2 bytes)
    constexpr InterleavedPow2AddrGen<out_in_dram> s = {
        .bank_base_address = out_addr,
        .log_base_2_of_page_size = tile_size_pow2_exponent
    };

    // read in bias if enabled (done only once for all batches)
    #ifdef FUSE_BIAS
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(3);
    bool load_bias = true;
    #endif

    // OUTER most loop is looping over out blocks in width dim because blocks from compute are in col major order.
    // Write out col major blocks in row major layout to output
    uint32_t out_block_w_start_tile_id = out_start_tile_id;
    uint32_t out_block_w_start_tile_id_w = out_start_tile_id_w;
    uint32_t weight_start_tile_id = out_start_tile_id_w;
    for (uint32_t bw = 0; bw < out_num_blocks_w; bw++) {
        uint32_t out_block_h_start_tile_id = out_block_w_start_tile_id;
        uint32_t out_block_h_start_tile_id_h = out_start_tile_id_h;
        for(uint32_t bh = 0; bh < out_num_blocks_h; bh++) {
            // MCAST RECEIVE WEIGHTS
            // read weight blocks inner dim
            // read weight slice - 1 block of weights in width dim and full weight matrix height
            // read slice only once for all activation blocks
            for(uint32_t weight_tile_h_outer_i = 0; weight_tile_h_outer_i < weight_block_height_num_outer; weight_tile_h_outer_i++) {
                cb_reserve_back(cb_id_weight, weight_block_num_tiles);
                // Set weights semaphore value to INVALID
                noc_semaphore_set(weights_mcast_receiver_semaphore_addr_ptr, INVALID);

                // Atomic increment source core counter
                noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1);

                // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                noc_semaphore_wait(weights_mcast_receiver_semaphore_addr_ptr, VALID);

                cb_push_back(cb_id_weight, weight_block_num_tiles);
            } // for weight_block_height_num_outer

            #ifdef FUSE_BIAS
            if (load_bias) {
                cb_reserve_back(bias_cb_id, bias_ntiles);

                // Set weights semaphore value to INVALID
                noc_semaphore_set(weights_mcast_receiver_semaphore_addr_ptr, INVALID);

                // Atomic increment source core counter
                noc_semaphore_inc(weights_mcast_sender_semaphore_noc_addr, 1);

                // wait on weights semaphore value to become VALID (set by mcast sender after it multicasts data)
                noc_semaphore_wait(weights_mcast_receiver_semaphore_addr_ptr, VALID);

                cb_push_back(bias_cb_id, bias_ntiles);
                load_bias = false;
            }
            #endif

            #ifndef SHARDED_OUT
            uint32_t out_sbh_start_tile_id = out_block_h_start_tile_id;
            uint32_t out_sbh_start_tile_id_h = out_block_h_start_tile_id_h; //
            for(uint32_t sbh = 0; sbh < out_num_subblocks_h; sbh++) {
                uint32_t out_sbw_start_tile_id = out_sbh_start_tile_id;
                uint32_t out_sbw_start_tile_id_w = out_block_w_start_tile_id_w;
                for(uint32_t sbw = 0; sbw < out_num_subblocks_w; sbw++) {
                    uint32_t out_sb_row_start_tile_id = out_sbw_start_tile_id;
                    // wait for one subblock worth tiles
                    cb_wait_front(cb_id_out0, out_subblock_tile_count);
                    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
                    for(uint32_t h = 0; h < out_subblock_h; h++) {
                        uint32_t out_tile_id = out_sb_row_start_tile_id;
                        uint32_t out_tile_id_h = out_sbh_start_tile_id_h + h;
                        if (out_tile_id_h >= out_height_num_tiles) { // block shape height padding
                            break;
                        }
                        for(uint32_t w = 0; w < out_subblock_w; w++) {
                            uint32_t out_tile_id_w = out_sbw_start_tile_id_w + w;
                            if (out_tile_id_w >= out_width_num_tiles) { // block shape width padding
                                l1_read_addr += tile_nbytes;
                            } else {
                                uint64_t out_tile_noc_addr = get_noc_addr(out_tile_id, s);
                                noc_async_write(l1_read_addr, out_tile_noc_addr, tile_nbytes);
                                l1_read_addr += tile_nbytes;
                                out_tile_id += out_next_tile_stride_w;
                            }
                        } // out_subblock_w (ntiles)
                        out_sb_row_start_tile_id += out_next_tile_stride_h;
                    } // out_subblock_h (ntiles)
                    noc_async_write_barrier();
                    cb_pop_front(cb_id_out0, out_subblock_tile_count);
                    out_sbw_start_tile_id += out_next_subblock_stride_w;
                    out_sbw_start_tile_id_w += out_subblock_w;
                } // out_num_subblocks_w
                out_sbh_start_tile_id += out_next_subblock_stride_h;
                out_sbh_start_tile_id_h += out_subblock_h;
            } // out_num_subblocks_h
            out_block_h_start_tile_id += out_next_block_stride_h;
            out_block_h_start_tile_id_h += out_block_height_num_tiles;
            #endif
        } // out_num_blocks_h
        out_block_w_start_tile_id += out_next_block_stride_w;
        out_block_w_start_tile_id_w += weight_block_width_ntiles;

        // Increment weight start tile id for next block in width dim
        weight_start_tile_id += weight_next_block_stride_w;
    } // out_num_blocks_w

    #ifdef SHARDED_OUT
    #ifdef UNPAD_UNTILIZE_OUT
    uint32_t dst_cb_addr = get_write_ptr(cb_id_out0);
    /*DPRINT << "dst_cb_addr: " << dst_cb_addr << ENDL();*/
    /*DPRINT << "out_num_blocks_w: " << out_num_blocks_w << ENDL();*/
    /*DPRINT << "out_num_blocks_h: " << out_num_blocks_h << ENDL();*/
    /*DPRINT << "out_block_width_ntiles: " << out_block_width_ntiles << ENDL();*/
    /*DPRINT << "out_block_height_ntiles: " << out_block_height_num_tiles << ENDL();*/
    /*DPRINT << "out_blwidth_padded_bytes: " << out_block_width_padded_bytes << ENDL();*/
    /*DPRINT << "out_block_width_bytes: " << out_block_width_bytes << ENDL();*/
    /*DPRINT << "out_blt_num_tiles = " << out_block_height_num_tiles << ENDL();*/
    /*print_pages(get_read_ptr(untilized_padded_out_cb), out_block_width_padded_bytes/2, 32*4);*/

    uint32_t src_cb_addr = get_read_ptr(untilized_padded_out_cb);
    for (uint32_t nbw = 0; nbw < out_num_blocks_w; nbw++) {
        for(uint32_t nbh = 0; nbh < out_num_blocks_h; nbh++) {
            for (uint32_t bh = 0; bh < out_block_height_num_tiles; bh++) {
                /*DPRINT << "Waiting for out_block_width_ntiles: " << out_block_width_ntiles << ENDL();*/
                cb_wait_front(untilized_padded_out_cb, out_block_width_ntiles);
                uint32_t src_cb_addr = get_read_ptr(untilized_padded_out_cb);
                /*DPRINT << "src_cb_addr: " << src_cb_addr << ENDL();*/
                /*DPRINT << "Done waiting for out_block_width_ntiles: " << out_block_width_ntiles << ENDL();*/
                for (uint32_t r = 0; r < 32; r++) {
                    noc_async_read(get_noc_addr(src_cb_addr), dst_cb_addr, out_block_width_bytes);
                    noc_async_read_barrier();
                    /*print_pages(get_noc_addr(src_cb_addr), out_block_width_bytes/2, 1);*/
                    src_cb_addr += out_block_width_padded_bytes;
                    /*dst_cb_addr += out_block_width_bytes;*/

                    dst_cb_addr += out_aligned_page_size;
                }
                cb_pop_front(untilized_padded_out_cb, out_block_width_ntiles);
            }
        }
    }
    #else
    cb_wait_front(cb_id_out0, out_subblock_tile_count * out_num_subblocks_h * out_num_subblocks_w * out_num_blocks_w * out_num_blocks_h);
    #endif
    #endif
}
