// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
// #include "dprint.h"

void kernel_main() {
    const uint32_t cache_addr  = get_arg_val<uint32_t>(0);
    const uint32_t Wt          = get_arg_val<uint32_t>(1);
    const uint32_t B           = get_arg_val<uint32_t>(2);
    const uint32_t num_batched_heads      = get_arg_val<uint32_t>(3);
    const uint32_t cache_total_num_tiles  = get_arg_val<uint32_t>(4);
    const uint32_t cache_batch_num_tiles  = get_arg_val<uint32_t>(5);
    const uint32_t cache_head_num_tiles   = get_arg_val<uint32_t>(6);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(7);
    const uint32_t batch_start_id = get_arg_val<uint32_t>(8);
    const uint32_t Wbytes      = get_arg_val<uint32_t>(9);
    const uint32_t offset      = get_arg_val<uint32_t>(10);

    constexpr bool cache_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_input_cb_id = get_compile_time_arg_val(4);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);

    constexpr uint32_t sizeof_bf16 = 2;
    const uint32_t bytes_per_face_row = 16 * sizeof_bf16;
    const uint32_t bytes_per_face = 16 * 16 * sizeof_bf16;

    const uint32_t bytes_per_row = Wbytes / Wt;

    // DPRINT << "Wbytes: " << Wbytes << ENDL();
    // DPRINT << "Wt: " << Wt << ENDL();
    // DPRINT << "bytes_per_row: " << bytes_per_row << ENDL();
    // DPRINT << "offset: " << offset << ENDL();
    // DPRINT << "cache_tile_bytes: " << cache_tile_bytes << ENDL();


    const InterleavedAddrGenFast<cache_is_dram> s0 = {
        .bank_base_address = cache_addr,
        .page_size = cache_tile_bytes,
        .data_format = cache_data_format
    };

    uint32_t cache_id = cache_start_id;
    uint32_t b = batch_start_id;

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
        cb_wait_front(untilized_input_cb_id, Wt);
        uint64_t input_l1_read_addr = get_noc_addr(get_read_ptr(untilized_input_cb_id));

        for (uint32_t u = 0; u < 32; ++u) {

            // Wait for reader to push Wt tiles into cache_cb_id
            cb_wait_front(cache_cb_id, Wt);
            // Get pointer to first tilized row of cache to write
            uint32_t cache_l1_write_addr = get_read_ptr(cache_cb_id) + offset;
            uint64_t input_untilized_row_addr = input_l1_read_addr;

            // Scatter untilized input row into tilized cache
            for (uint32_t tile_num = 0; tile_num < Wt; ++tile_num) {
                // DPRINT << "Writing " << bytes_per_row << " bytes from " << input_untilized_row_addr << " to " << cache_l1_write_addr << ENDL();
                for (uint32_t face_col = 0; face_col < 2; ++face_col) {
                    noc_async_read(input_untilized_row_addr, cache_l1_write_addr, bytes_per_face_row);
                    input_untilized_row_addr += bytes_per_face_row;
                    cache_l1_write_addr += bytes_per_face;
                }
                cache_l1_write_addr += cache_tile_bytes - 2*bytes_per_face;
            }
            // Block on writes to tilized output buffer
            noc_async_read_barrier();

            input_l1_read_addr += Wbytes;
            uint32_t out_l1_read_addr = get_read_ptr(cache_cb_id);
            for(uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                // DPRINT << "Writing tile ID" << curr_cache_id << " from L1 addr " << out_l1_read_addr << ENDL();
                noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
                out_l1_read_addr += cache_tile_bytes;
            }
            cache_id += cache_batch_num_tiles; // Input is read in by batch, then heads so skip to next batch
            b++;
            if (b == B) {
                b = 0;
                cache_id = cache_id - cache_total_num_tiles + cache_head_num_tiles; // Start of next head
            }
            noc_async_write_barrier();
            cb_pop_front(cache_cb_id, Wt);
        }
        cb_pop_front(untilized_input_cb_id, Wt);
    }
}
