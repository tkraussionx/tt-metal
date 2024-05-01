// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "dprint.h"

void kernel_main() {
    const uint32_t cache_addr  = get_arg_val<uint32_t>(0);
    const uint32_t input_addr  = get_arg_val<uint32_t>(1);
    const uint32_t Wt          = get_arg_val<uint32_t>(2);
    const uint32_t B           = get_arg_val<uint32_t>(3);
    const uint32_t num_batched_heads      = get_arg_val<uint32_t>(4);
    const uint32_t cache_total_num_tiles  = get_arg_val<uint32_t>(5);
    const uint32_t cache_batch_num_tiles  = get_arg_val<uint32_t>(6);
    const uint32_t cache_head_num_tiles   = get_arg_val<uint32_t>(7);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(8);
    const uint32_t batch_start_id = get_arg_val<uint32_t>(9);
    const uint32_t input_start_id = get_arg_val<uint32_t>(10);
    const uint32_t Wbytes      = get_arg_val<uint32_t>(11);
    const uint32_t offset      = get_arg_val<uint32_t>(12);
    const uint32_t batch_read_offset = get_arg_val<uint32_t>(13);

    // mcast args
    const uint32_t mcast_dest_noc_start_x         = get_arg_val<uint32_t>(14);
    const uint32_t mcast_dest_noc_start_y         = get_arg_val<uint32_t>(15);
    const uint32_t mcast_dest_noc_end_x           = get_arg_val<uint32_t>(16);
    const uint32_t mcast_dest_noc_end_y           = get_arg_val<uint32_t>(17);

    constexpr bool cache_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_cache_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t untilized_cache2_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t untilized_input_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t granularity = get_compile_time_arg_val(7);
    constexpr uint32_t u_count = get_compile_time_arg_val(8);
    constexpr uint32_t start_u = get_compile_time_arg_val(9);
    constexpr uint32_t end_u = get_compile_time_arg_val(10);

    // mcast args
    constexpr uint32_t mcast_sender_semaphore_addr    = get_compile_time_arg_val(9);
    constexpr uint32_t mcast_receiver_semaphore_addr  = get_compile_time_arg_val(10);
    constexpr uint32_t mcast_num_dests                = get_compile_time_arg_val(11);
    constexpr uint32_t mcast_num_cores                = get_compile_time_arg_val(12);

    #ifndef SKIP_MCAST
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    *(mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);

    const uint64_t mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        mcast_receiver_semaphore_addr);

    const uint64_t multicast_data_noc = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        0);
    #endif

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const DataFormat input_data_format = get_dataformat(input_cb_id);

    const InterleavedAddrGenFast<cache_is_dram> s0 = {
        .bank_base_address = cache_addr,
        .page_size = cache_tile_bytes,
        .data_format = cache_data_format
    };

    #ifdef INPUT_SHARDED
    cb_reserve_back(input_cb_id, Wt * num_batched_heads);
    cb_push_back(input_cb_id, Wt * num_batched_heads);
    #else
    const InterleavedAddrGenFast<input_is_dram> s1 = {
        .bank_base_address = input_addr,
        .page_size = input_tile_bytes,
        .data_format = input_data_format
    };
     uint32_t input_id = input_start_id;
    #endif

    uint32_t cache_id = cache_start_id;
    uint32_t b = batch_start_id;

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
        #ifndef INPUT_SHARDED
        cb_reserve_back(input_cb_id, Wt);
        uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
        for (uint32_t i = 0; i < Wt; ++i) {
            noc_async_read_tile(input_id, s1, input_l1_write_addr);
            input_l1_write_addr += input_tile_bytes;
            input_id++;
        }
        noc_async_read_barrier();
        cb_push_back(input_cb_id, Wt);
        #endif

        // wait on itself to pushback original cache new
        cb_wait_front(input_cb_id, Wt);
        uint64_t input_l1_read_addr = get_read_ptr(input_cb_id) + batch_read_offset;

        uint64_t mcast_start_address = input_l1_read_addr; //copy start address of block, to be used for mcasting

        #ifndef SKIP_MCAST
        // wait until all in1 mcast destinations have atomically incremented the in1 semaphore_addr (i.e. its value should be in0_mcast_num_dests), then reset
        // the semaphore_addr value back to zero for the next block
        noc_semaphore_wait(mcast_sender_semaphore_addr_ptr, mcast_num_dests);
        noc_semaphore_set(mcast_sender_semaphore_addr_ptr, 0);

        // Now we have the block in the CB address, we can mcast to dests!
        uint64_t multicast_data_addr = multicast_data_noc | mcast_start_address;

        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_async_write_multicast(mcast_start_address, multicast_data_addr, Wbytes, mcast_num_cores, false, false);

        // Note: no need for write barrier, since these two multicasts are done on the same noc id, same vc, same cmd_buf
        // Also, this only works because we are setting VCs statically (using NOC_CMD_STATIC_VC).

        // We should also multicast the flag to destinations
        // num_dests must not include source, since we are NOT really doing a local copy!
        noc_semaphore_set_multicast(mcast_receiver_semaphore_addr, mcast_receiver_semaphore_noc_addr, mcast_num_cores, false, false);

        #endif

        for (uint32_t u = start_u; u <= end_u; ++u) {
            // Operating on a granularity > 1 led to performance improvements.
            // It introduces a double-buffered pipeline between compute and writer.
            for (uint32_t g = 0; g < granularity; ++g) {
                // Wait on compute to untilize a block. Update that block in L1.
                cb_wait_front(untilized_cache_cb_id, Wt);
                cb_reserve_back(untilized_cache2_cb_id, Wt);
                uint32_t cache_l1_write_addr = get_read_ptr(untilized_cache_cb_id) + offset;
                noc_async_read(input_l1_read_addr, cache_l1_write_addr, Wbytes);
                input_l1_read_addr += Wbytes;
                noc_async_read_barrier();
                cb_push_back(untilized_cache2_cb_id, Wt);
                cb_pop_front(untilized_cache_cb_id, Wt); // NEW
            }

            for (uint32_t g = 0; g < granularity; ++g) {
                // Wait on compute to tilize an updated block. Write that block to DRAM
                cb_wait_front(cache_cb_id, Wt);
                uint32_t out_l1_read_addr = get_read_ptr(cache_cb_id);
                for(uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                    noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
                    out_l1_read_addr += cache_tile_bytes;
                }
                cache_id += cache_batch_num_tiles; // Input is read in by batch, then heads so skip to next batch
                b++;
                if (b == B) {
                    b = 0;
                    cache_id = cache_id - cache_total_num_tiles + cache_head_num_tiles; // Start of next head
                }
                noc_async_writes_flushed();
                cb_pop_front(cache_cb_id, Wt);
            }
        }
        cb_pop_front(untilized_input_cb_id, Wt);
    }
    // Delay syncing the writes to maximize perf.
    noc_async_write_barrier();
}
