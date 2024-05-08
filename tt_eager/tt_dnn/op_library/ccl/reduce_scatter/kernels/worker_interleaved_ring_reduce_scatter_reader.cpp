// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/assert.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "READER: Start\n";
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_transfers = get_arg_val<uint32_t>(2);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(3);
    const uint32_t page_size = get_arg_val<uint32_t>(4);
    const uint32_t output_page_size = get_arg_val<uint32_t>(5);
    const uint32_t num_pages = get_arg_val<uint32_t>(6);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(7);
    const uint32_t input_start_idx = get_arg_val<uint32_t>(8);
    const uint32_t output_start_idx = get_arg_val<uint32_t>(9);
    const uint32_t output_start_addr_offset = get_arg_val<uint32_t>(10);
    const uint32_t row_start_idx = get_arg_val<uint32_t>(11);
    const uint32_t col_start_idx = get_arg_val<uint32_t>(12);
    const uint32_t row_offset = get_arg_val<uint32_t>(13);
    const uint32_t col_offset = get_arg_val<uint32_t>(14);
    const uint32_t num_rows = get_arg_val<uint32_t>(15);
    const uint32_t num_cols = get_arg_val<uint32_t>(16);
    const uint32_t last_output_page_offset = get_arg_val<uint32_t>(17);
    const uint32_t output_page_offset = get_arg_val<uint32_t>(18);
    const uint32_t last_output_addr_offset = get_arg_val<uint32_t>(19);
    const uint32_t output_addr_offset = get_arg_val<uint32_t>(20);
    const uint32_t input_start_ring_idx = get_arg_val<uint32_t>(21);
    const uint32_t sem_addr = get_arg_val<uint32_t>(22);
    const bool is_clockwise_direction = get_arg_val<uint32_t>(23) == 1;
    const uint32_t half_cb_n_pages = get_arg_val<uint32_t>(24);
    const uint32_t ring_size = get_arg_val<uint32_t>(25);

    const uint32_t edm_core_noc0_core_x = get_arg_val<uint32_t>(26);
    const uint32_t edm_core_noc0_core_y = get_arg_val<uint32_t>(27);
    const uint32_t edm_core_semaphore_address = get_arg_val<uint32_t>(28);
    const uint32_t edm_core_buffer_address = get_arg_val<uint32_t>(29);
    ASSERT(half_cb_n_pages > rem_num_pages);

    constexpr uint32_t to_dm_sender_short_circuit_cb = tt::CB::c_out0;
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;

    #ifdef RM_INTERLEAVED
    InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = page_size};
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr, .page_size = page_size};
    #elif defined TILE_INTERLEAVED
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = page_size,
        .data_format = in0_df
    };

    InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = output_page_size,
        .data_format = in0_df
    };
    #endif
    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    const uint64_t eth_receiver_l1_base_noc_addr = get_noc_addr(edm_core_noc0_core_x, edm_core_noc0_core_y, edm_core_buffer_address);
    const uint64_t eth_receiver_l1_semaphore_noc_addr = get_noc_addr(edm_core_noc0_core_x, edm_core_noc0_core_y, edm_core_semaphore_address);


    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t input_page_idx = input_start_idx;
    uint32_t output_base_page_idx = output_start_idx;
    // uint32_t output_page_idx = output_base_page_idx;
    // uint32_t col_idx = col_start_idx;
    // uint32_t row_idx = row_start_idx;

    // For the first timestep, there is no other input to reduce with, so we just send it straight to the input CB
    // of the output data movement kernel - short-circuiting past the (reducer) math kernel
    {

        // Fetch from input tensor chunk to cb_in1
        uint32_t output_page_idx = output_base_page_idx;
        uint32_t col_idx = col_start_idx;
        uint32_t row_idx = row_start_idx;
        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                // read_chunk_from_input_tensor(input_page_idx, to_dm_sender_short_circuit_cb, s, num_pages, page_size);
                read_chunk_from_output_tensor(
                    output_page_idx, col_idx, row_idx, cb_id_in1, s, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);
            }
        }
        if (rem_num_pages > 0) {
            // read_chunk_from_input_tensor(input_page_idx, to_dm_sender_short_circuit_cb, s, rem_num_pages, page_size);
            read_chunk_from_output_tensor(
                output_page_idx, col_idx, row_idx, cb_id_in1, s, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            push_filler_pages_to_cb(to_dm_sender_short_circuit_cb, half_cb_n_pages - rem_num_pages);
        }
    }

    uint32_t total_num_pages_to_send = num_pages * num_full_chunks + rem_num_pages;

    // num_transfers = num_devices - 1
    for (uint32_t i = 1; i < num_transfers; ++i) {
        DPRINT << "READER: TRANSFER " << i << "\n";

        // Fetch from EDM to cb_in0
        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                uint64_t eth_receiver_l1_curr_noc_addr = eth_receiver_l1_base_noc_addr;
                DPRINT << "READER: FCSW\n";
                noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
                noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
                // Read page by page so that writer can be kicked off instead of being blocked waiting for full chunk to be read
                // Look into perf/optimizations for this
                DPRINT << "READER: FETCH\n";
                fetch_chunk(cb_id_in0, num_pages, page_size, eth_receiver_l1_base_noc_addr);
                DPRINT << "READER: SI\n";
                noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
                // transfers_completed++;
            }
        }
        if (rem_num_pages > 0) {
            uint64_t eth_receiver_l1_curr_noc_addr = eth_receiver_l1_base_noc_addr;
            DPRINT << "READER: RCSW\n";
            noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
            noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
            DPRINT << "READER: FETCH\n";
            fetch_chunk(cb_id_in0, rem_num_pages, page_size, eth_receiver_l1_base_noc_addr);
            DPRINT << "READER: SI\n";
            noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
            // transfers_completed++;
        }

        // Fetch from input tensor chunk to cb_in1
        uint32_t output_page_idx = output_base_page_idx;
        uint32_t col_idx = col_start_idx;
        uint32_t row_idx = row_start_idx;

        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                // read_chunk_from_input_tensor(7output_page_idx,
                DPRINT << "READER: FC FETCH TENSOR\n";
                read_chunk_from_output_tensor(output_page_idx,
                    col_idx, row_idx, cb_id_in1, s, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);
                DPRINT << "READER: COMPLETE FC FETCH TENSOR\n";
            }
        }
        if (rem_num_pages > 0) {
            DPRINT << "READER: RC FETCH TENSOR\n";
            read_chunk_from_output_tensor(output_page_idx, col_idx, row_idx, cb_id_in1, s, num_cols, num_rows, col_offset, row_offset, rem_num_pages, page_size);
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            DPRINT << "READER: COMPLETE RC FETCH TENSOR\n";
            push_filler_pages_to_cb(cb_id_in1, half_cb_n_pages - rem_num_pages);
            DPRINT << "READER: FILLER PAGES\n";
        }

        // //// Optimized version
        // while (total_num_pages_to_send > 0) {
        //     uint32_t num_pages_to_send = std::min<uint32_t>(total_num_pages_to_send, num_pages);
        //     total_num_pages_to_send -= num_pages_to_send;
        //     // TODO: fuse these two so we don't require a hard read barrier before reading the input tensor
        //     noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
        //     noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
        //     fetch_chunk(cb_id_in0, num_pages_to_send, page_size, eth_receiver_l1_base_noc_addr);
        //     noc_semaphore_inc(eth_receiver_l1_semaphore_noc_addr, 1);

        //     // read_chunk_from_input_tensor(
        //     read_chunk_from_output_tensor(
        //         output_page_idx, col_idx, row_idx, cb_id_in1, s, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);

        //     if (half_cb_n_pages - num_pages_to_send > 0) {
        //         push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - num_pages_to_send);
        //         push_filler_pages_to_cb(cb_id_in1, half_cb_n_pages - num_pages_to_send);
        //     }
        // }
        // //////



        // TODO: move to interleaved ring worker utils - it's used in a lot of places
        if (is_clockwise_direction) {
            if (input_ring_idx == 0) {
                input_ring_idx = ring_size - 1;
                d.bank_base_address += last_output_addr_offset;
                output_base_page_idx += last_output_page_offset;
            } else {
                input_ring_idx--;
                d.bank_base_address -= output_addr_offset;
                output_base_page_idx -= output_page_offset;
            }
        } else {
            if (input_ring_idx == ring_size - 1) {
                input_ring_idx = 0;
                d.bank_base_address -= last_output_addr_offset;
                output_base_page_idx -= last_output_page_offset;
            } else {
                input_ring_idx++;
                d.bank_base_address += output_addr_offset;
                output_base_page_idx += output_page_offset;
            }
        }
    }

    DPRINT << "READER: END\n";
}
