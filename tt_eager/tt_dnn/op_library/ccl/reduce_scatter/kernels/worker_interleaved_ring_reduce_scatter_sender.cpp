// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "tt_eager/tt_dnn/op_library/ccl/kernel_common/worker_edm_utils.hpp"

#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "SENDER START\n";
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t eth_sender_l1_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t eth_sender_l1_sem_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_transfers = get_arg_val<uint32_t>(3);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(4);
    const uint32_t page_size = get_arg_val<uint32_t>(5);
    const uint32_t output_page_size = get_arg_val<uint32_t>(6);
    const uint32_t full_chunk_num_pages = get_arg_val<uint32_t>(7);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(8);
    const uint32_t input_start_idx = get_arg_val<uint32_t>(9);
    const uint32_t output_start_idx = get_arg_val<uint32_t>(10);
    const uint32_t output_start_addr_offset = get_arg_val<uint32_t>(11);
    const uint32_t row_start_idx = get_arg_val<uint32_t>(12);
    const uint32_t col_start_idx = get_arg_val<uint32_t>(13);
    const uint32_t row_offset = get_arg_val<uint32_t>(14);
    const uint32_t col_offset = get_arg_val<uint32_t>(15);
    const uint32_t num_rows = get_arg_val<uint32_t>(16);
    const uint32_t num_cols = get_arg_val<uint32_t>(17);
    const uint32_t input_start_ring_idx = get_arg_val<uint32_t>(18);
    const uint32_t writer_send_sem_addr = get_arg_val<uint32_t>(19);
    const uint32_t eth_sender_noc_x = get_arg_val<uint32_t>(20);
    const uint32_t eth_sender_noc_y = get_arg_val<uint32_t>(21);
    const uint32_t half_cb_n_pages = get_arg_val<uint32_t>(22);
    const uint32_t ring_size = get_arg_val<uint32_t>(23);
    const uint32_t output_start_page_idx = get_arg_val<uint32_t>(24);
    const uint32_t last_output_addr_offset = get_arg_val<uint32_t>(25);
    const uint32_t last_output_page_offset = get_arg_val<uint32_t>(26);
    const uint32_t output_addr_offset = get_arg_val<uint32_t>(27);
    const uint32_t output_page_offset = get_arg_val<uint32_t>(28);

    const bool is_clockwise_direction = get_arg_val<uint32_t>(29) == 1;
    ASSERT(half_cb_n_pages > rem_num_pages);//, "half_cb_n_pages must be greater than or equal to rem_num_pages");

    constexpr uint32_t cb_id_in0 = tt::CB::c_out0;
    // constexpr uint32_t cb_id_in1 = tt::CB::c_in1;
    // constexpr uint32_t to_dataflow_out_short_circuit_cb = tt::CB::c_out0;
    #ifdef RM_INTERLEAVED
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = output_page_size};
    #elif defined TILE_INTERLEAVED
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr,
        .page_size = output_page_size,
        .data_format = in0_df
    };
    #endif

    // Used to wait until eth sender has space available
    volatile tt_l1_ptr uint32_t* writer_send_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_send_sem_addr);

    uint32_t output_base_page_idx = output_start_page_idx;
    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t output_page_idx = output_start_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;
    // This is different per writer core
    const uint64_t eth_l1_sender_base_noc_addr = get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_base_addr);
    // Used to signal eth sender that data is available. This is different per writer core
    const uint64_t eth_l1_sender_semaphore_addr = get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_sem_addr);


    for (uint32_t i = 1; i < num_transfers; ++i) {
        DPRINT << "SENDER TRANSFER " << i << "\n";
        // TODO: Move this to a flat count (pages per full transfer, count-down from there)
        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
                noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
                send_chunk(cb_id_in0, full_chunk_num_pages, page_size, eth_l1_sender_base_noc_addr);
                noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
            }
        }
        if (rem_num_pages > 0) {
            noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
            noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
            send_chunk(cb_id_in0, rem_num_pages, page_size, eth_l1_sender_base_noc_addr);
            noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
            ASSERT(full_chunk_num_pages == 0 || full_chunk_num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }

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
            if (input_ring_idx == ring_size - 1) {//0) {
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

    // write the final reduced chunk for this chip out to the output tensor
    if (num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
            noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
            write_chunk(
                output_page_idx,
                col_idx,
                row_idx,
                cb_id_in0,
                d,
                num_cols,
                num_rows,
                col_offset,
                row_offset,
                full_chunk_num_pages,
                page_size);
            noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
        }
    }
    if (rem_num_pages > 0) {
        noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
        noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
        write_chunk(
            output_page_idx,
            col_idx,
            row_idx,
            cb_id_in0,
            d,
            num_cols,
            num_rows,
            col_offset,
            row_offset,
            rem_num_pages,
            page_size);
        noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
        ASSERT(full_chunk_num_pages == 0 || full_chunk_num_pages > rem_num_pages);
        ASSERT(half_cb_n_pages > rem_num_pages);
        pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
    }
    DPRINT << "SENDER END\n";
}
