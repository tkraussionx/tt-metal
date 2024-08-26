// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "firmware_common.h"
#define ENABLE_DEBUG 1


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

// FORCE_INLINE
// void read_channels(uint32_t& l1_write_addr_act, const uint32_t act_l1_read_addr, const uint32_t reader_channel_idx,
//         const uint32_t conv_act_c_read_bytes, const uint32_t coalesced_read_bytes, const uint32_t stride_h_bytes) {

//     constexpr uint32_t unroll_factor = WINDOW_INNER;
//     uint32_t act_l1_read_addr_plus_offset = act_l1_read_addr + (reader_channel_idx * conv_act_c_read_bytes);
//     #pragma GCC unroll unroll_factor
//     for (uint32_t inner = 0; inner < WINDOW_INNER; inner++) {
//         noc_async_read_one_packet_with_state<true>(act_l1_read_addr_plus_offset, l1_write_addr_act);
//         l1_write_addr_act += coalesced_read_bytes;
//         // +2 is hard-coded, TODO: generalize
//         act_l1_read_addr_plus_offset += stride_h_bytes;
//     }
// }

#define ALWI inline __attribute__((always_inline))

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n/4; ++ i) {
        ptr[i] = 0;
    }
    return true;
}


void kernel_main() {
    constexpr uint32_t LOCAL_PACKED_READER_INDICES_MAX_SIZE = 128;
    uint32_t local_packed_reader_indices[LOCAL_PACKED_READER_INDICES_MAX_SIZE];

    constexpr bool act_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t stride_w = get_compile_time_arg_val(2);
    constexpr uint32_t conv_act_size_w_ = get_compile_time_arg_val(3);
    constexpr uint32_t conv_output_w_last_index = get_compile_time_arg_val(4) - 1;
    constexpr uint32_t conv_act_c_read_bytes = get_compile_time_arg_val(5);
    // need to have these as compile-time, they are inner loop bouds / unroll loops / constexpr conditionals based on them
    constexpr uint32_t window_outer = get_compile_time_arg_val(6);
    constexpr uint32_t window_inner = get_compile_time_arg_val(7);
    constexpr uint32_t act_block_h_datums = get_compile_time_arg_val(8);
    constexpr uint32_t act_block_num_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t weight_size_w = get_compile_time_arg_val(10);
    constexpr uint32_t conv_act_size_w_padded = get_compile_time_arg_val(11);
    constexpr uint32_t act_block_w_extra_align_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t weight_size_h= get_compile_time_arg_val(13);
    constexpr uint32_t act_num_blocks_h = get_compile_time_arg_val(14);
    constexpr uint32_t act_block_h_datums_last_block = get_compile_time_arg_val(23);

    DPRINT << "conv_act_size_w " << conv_act_size_w_ << " act_block_h_datums " << act_block_h_datums << " act_block_num_tiles "
     << act_block_num_tiles << " act_num_blocks_h " << act_num_blocks_h << " act_block_h_datums_last_block " << act_block_h_datums_last_block << ENDL();
    DPRINT << "windoes_OUTER" << window_outer << " window_inner " << window_inner << " weight_size_w " << weight_size_w << " weight_size_h " << weight_size_h << ENDL();
    uint32_t act_block_h_datums_read_last_block;
    if (act_block_h_datums_last_block > act_block_h_datums) {
        act_block_h_datums_read_last_block = act_block_h_datums / 2;
    } else {
        act_block_h_datums_read_last_block = act_block_h_datums_last_block / 2;
    }

    // if( my_x[0] != 0 || (uint) my_y[0] >4){
    //     return;
    // }

    uint32_t i = 0;
    uint32_t noop = get_arg_val<uint32_t>(i); i+=1;

    if(noop) {
        return;
    }

    constexpr uint32_t cb_id_act = 0;
    constexpr uint32_t cb_id_sharded_act = 3;

    // LOOP TO FILL READER OFFSETS
    /* We can add another loop to read chunks of a stick as well.
     * - Duplicate reader_offset for same stick X times (window_inner must be 1)
     * - New loop between outer and inner that loops X times reading from same stick
     * - Read conv_act_c_read_bytes / X each time
     * - Update l1_write_addr_act by conv_act_c_read_bytes
     */
    uint32_t reader_offsets[weight_size_w*weight_size_h];
    uint32_t reader_offset = 0; // Constant offset for each pixel within filter window
    uint32_t reader_offset_idx = 0;
    for (uint32_t channel_stick_h = 0; channel_stick_h < weight_size_h; channel_stick_h++) {
        uint32_t reader_offset_row = reader_offset;
        for (uint32_t channel_stick_w = 0; channel_stick_w < weight_size_w; channel_stick_w++) {
            reader_offsets[reader_offset_idx++] = reader_offset_row++;
        }
        // -1 to go back to previous reader_offset
        reader_offset += conv_act_size_w_padded;
    }

    #ifdef SPLIT_READER
    constexpr uint32_t act_block_h_datums_read = act_block_h_datums / 2; // Extra /2 because of packed uint16 reads
    constexpr uint32_t act_block_num_tiles_read = act_block_num_tiles;
    #else
    constexpr uint32_t act_block_h_datums_read = act_block_h_datums / 2; // packed uint16 reads
    constexpr uint32_t act_block_num_tiles_read = act_block_num_tiles;
    #endif


    // LOOP TO FILL READER INDICES
    constexpr uint32_t cb_reader_indices = tt::CB::c_in4;
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_reader_indices));

    uint32_t reader_idx = 0;

    // Copy packed reader indices to local memory for faster access
    constexpr bool cache_packed_reader_indices = act_block_h_datums_read <= LOCAL_PACKED_READER_INDICES_MAX_SIZE;

    // TODO: need to make the read coalescing optimization cleaner
    // pass coalesce_window_inner_reads as a compile time arg and num_coalesced_reads so we can constexpr the if
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both src/dst side
    // we check if window_inner == weight_size_w to make sure coalescing is legal along full window_inner so the loop can be removed
    constexpr bool coalesce_window_inner_reads = true;
    constexpr uint32_t num_coalesced_reads = weight_size_w;
    constexpr uint32_t coalesced_read_bytes = num_coalesced_reads * conv_act_c_read_bytes;
    // the conditional selecting between coalescing and no-colescing must be constexpr to that compiler can optimized the other path away
    // this has shown to be a big perf win
    static_assert(act_block_h_datums % 2 == 0); // need to be even to read 2 in the body, due to packing of 2 indices in 1 uint32_t word
    // coalesce reads along weight_size_w
    reader_offset_idx = 0;
    uint32_t act_l1_offset = 0;
    uint32_t act_l1_read_addr = get_read_ptr(cb_id_sharded_act);


    static_assert(coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
    // set_state uses just x/y from the get_noc_addr, addr is ignored
    noc_async_read_one_packet_set_state(get_noc_addr(act_l1_read_addr), coalesced_read_bytes);
    uint32_t start_reader_idx = 0;
    for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
        #ifdef SPLIT_READER
        DPIRNT << "SPLIT_READER" << ENDL();
        if constexpr (cache_packed_reader_indices) {
            for (uint32_t i = 0; i < act_block_h_datums_read; i++) {
                local_packed_reader_indices[i] = packed_reader_indices_ptr[start_reader_idx+i];
            }
        }
        #endif
        //window_outer = 1;
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            // Reset reader_idx to finish act_block_h_datums
            reader_idx = start_reader_idx;

            cb_reserve_back(cb_id_act, act_block_num_tiles_read);
            uint32_t l1_write_addr_act = get_write_ptr(cb_id_act);
            uint32_t reader_offset = act_l1_read_addr + (reader_offsets[reader_offset_idx] * conv_act_c_read_bytes);
            if(outer == 0) {
                //print_pages(reader_offset, 64, 128, 0);
            }
            //print_pages(reader_offset, 64, 49, 0);
            // #pragma GCC unroll 4 // unroll didn't help, but act_block_h_datums (loop bound) being const does help
            uint32_t act_block_h_datums_read_curr = bh == act_num_blocks_h - 1 ? act_block_h_datums_read_last_block : act_block_h_datums_read;
            DPRINT << " act_block_h_datums_read_curr " << act_block_h_datums_read_curr << " act_block_num_tiles_read " << act_block_num_tiles_read << ENDL();
            for (uint32_t bhd = 0; bhd < 25; bhd++) {
                // local read from reader_index + reader_offset;
                #ifdef SPLIT_READER
                uint32_t two_reader_indices = cache_packed_reader_indices ? local_packed_reader_indices[bhd] : packed_reader_indices_ptr[reader_idx];
                #else // no split reader
                uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];
                #endif
                uint32_t reader_idx_1 = two_reader_indices & 0xffff;
                uint32_t reader_idx_2 = two_reader_indices >> 16;

                act_l1_offset = reader_offset + (reader_idx_1 * conv_act_c_read_bytes);
                noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);

                if(bhd != 24){
                    act_l1_offset = reader_offset + (reader_idx_2 * conv_act_c_read_bytes);
                    noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                    l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
                }


                //print_pages(reader_offset, 64, 1, 0);

                reader_idx++;
            }
            for (uint32_t bhd=0; bhd < 15; bhd++){
                fill_with_val(l1_write_addr_act, coalesced_read_bytes, 0);
                l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
            }
            noc_async_read_barrier();

            cb_push_back(cb_id_act, act_block_num_tiles_read);

            reader_offset_idx += window_inner;
        }
        reader_offset_idx = 0;

        start_reader_idx = reader_idx;
        #ifdef SPLIT_READER
        start_reader_idx += act_block_h_datums_read;
        #endif
    }
    DPRINT << "reader completed " << ENDL();
}
