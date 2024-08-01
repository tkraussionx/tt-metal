// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include <vector>

#include "debug/dprint.h"

template<uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
     return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
 }

void kernel_main() {
    /*
    In DRAM, Q is (B, PNHt, DHt), K is (B, St, DHt), V is (B, St, DHt), mask is (B, PNHt, PSt)
    We want to read for a particular batch cur_batch, and sequence length up to padded layer length.
    We read Q: (cur_batch, PNHt, DHt), K: (cur_batch, PSt, DHt), V: (cur_batch, PSt, DHt), mask: (cur_batch, PNHt, PSt)
    */
    constexpr uint32_t B = get_compile_time_arg_val(0);  // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);  // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);  // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);  // head dim
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t num_cores = get_compile_time_arg_val(5);
    constexpr uint32_t num_groups = get_compile_time_arg_val(6);
    constexpr uint32_t org_batch = get_compile_time_arg_val(7);

    const uint32_t q_addr  = get_arg_val<uint32_t>(0);
    const uint32_t k_addr  = get_arg_val<uint32_t>(1);
    const uint32_t v_addr  = get_arg_val<uint32_t>(2);
    const uint32_t PSt = get_arg_val<uint32_t>(3);  // padded layer length in tiles
    const uint32_t k_num_chunks = get_arg_val<uint32_t>(4);  // number of chunks in K, where k_num_chunks*Sk_chunk_t = PSt
    const uint32_t k_chunk_start = get_arg_val<uint32_t>(5);
    const uint32_t k_chunk_end = get_arg_val<uint32_t>(6);
    const uint32_t cur_batch =  get_arg_val<uint32_t>(7);
    const bool is_worker = get_arg_val<uint32_t>(8) == 1;

    tt_l1_ptr uint32_t * all_reducer_noc_x          = (tt_l1_ptr uint32_t*)(get_arg_addr(9));
    tt_l1_ptr uint32_t * all_reducer_noc_y          = (tt_l1_ptr uint32_t*)(get_arg_addr(9 + B));

    uint32_t reduce_core_noc_x = all_reducer_noc_x[cur_batch];
    uint32_t reduce_core_noc_y = all_reducer_noc_y[cur_batch];

    if (k_chunk_start == k_chunk_end) {
        return; // early exit because no computes needs to be done
    }

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t mask_chunk_tiles = PNHt * Sk_chunk_t;

    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;


    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    // First, read Q entirely, it is interleaved
    uint32_t q_batch_offset = cur_batch * q_chunk_tiles;
    uint32_t q_chunk_tiles_bytes = q_chunk_tiles * q_tile_bytes;

    uint32_t q_write_ptr = get_write_ptr(cb_q_in);
    uint64_t q_read_addr = get_noc_addr(q_addr);
    uint32_t q_element_bytes = q_tile_bytes / (32 * 32);
    DPRINT << "[Reader] reading Q" << ENDL();
    cb_reserve_back(cb_q_in, DHt);
    for (uint32_t q_head=(cur_batch % org_batch) * num_groups; q_head< ((cur_batch % org_batch) + 1) * num_groups; q_head++){
        for (uint32_t dim_chunk=0; dim_chunk< DHt; dim_chunk++){

            uint32_t read_head_offset = q_head*DHt*q_tile_bytes;
            uint32_t read_col_offset = dim_chunk*q_tile_bytes;
            uint32_t read_row_offset = cur_batch/org_batch * 32 * q_element_bytes;

            uint32_t write_col_offset = dim_chunk*q_tile_bytes;
            uint32_t write_row_offset = (q_head%num_groups) * 32 * q_element_bytes;

            uint32_t num_bytes_written = 32 * q_element_bytes;

            noc_async_read(q_read_addr + read_head_offset + read_col_offset + read_row_offset,
                        q_write_ptr + write_col_offset + write_row_offset,
                        num_bytes_written);
            noc_async_read_barrier();
        }

    }
    cb_push_back(cb_q_in, DHt);

    // Read the rest
    const InterleavedAddrGenFast<is_dram> k_reader = {
        .bank_base_address = k_addr,
        .page_size = k_tile_bytes,
        .data_format = k_data_format
    };

    const InterleavedAddrGenFast<is_dram> v_reader = {
        .bank_base_address = v_addr,
        .page_size = v_tile_bytes,
        .data_format = v_data_format
    };

    // Offset for current batch
    const uint32_t k_batch_offset = cur_batch * St * DHt;
    const uint32_t v_batch_offset = cur_batch * St * DHt;

    DPRINT << "[Reader] read Q" << ENDL();

    // Then, read K, V, Mask k_chunk_tiles at a time
    const uint32_t k_chunk_offset = k_chunk_start * Sk_chunk_t * DHt;
    const uint32_t v_chunk_offset = k_chunk_start * Sk_chunk_t * DHt;
    const uint32_t mask_chunk_offset = k_chunk_start * Sk_chunk_t;
    uint32_t k_start_tile_id = k_batch_offset + k_chunk_offset;
    uint32_t v_start_tile_id = v_batch_offset + v_chunk_offset;

    DPRINT << "[Reader] push kvm " << k_chunk_start << " to " << k_chunk_end << ENDL();

    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        // Read K chunk transposed
        cb_reserve_back(cb_k_in, k_chunk_tiles);
        uint32_t k_write_ptr = get_write_ptr(cb_k_in);
        barrier_count = 0;
        for (uint32_t col = 0; col < DHt; ++col) {
            uint32_t k_tile_id = k_start_tile_id + col;
            for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);
                k_tile_id += DHt;
                k_write_ptr += k_tile_bytes;

                if (++barrier_count == barrier_threshold) {
                    noc_async_read_barrier();
                    barrier_count = 0;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, k_chunk_tiles);
        k_start_tile_id += k_chunk_tiles;

        DPRINT << "[Reader] Finished Reading K at " << k_chunk << ENDL();

        // Read V chunk
        cb_reserve_back(cb_v_in, k_chunk_tiles);
        uint32_t v_write_ptr = get_write_ptr(cb_v_in);
        barrier_count = 0;
        uint32_t v_tile_id = v_start_tile_id;
        for (uint32_t tile = 0; tile < k_chunk_tiles; ++tile) {
            noc_async_read_tile(v_tile_id, v_reader, v_write_ptr);
            v_tile_id++;
            v_write_ptr += v_tile_bytes;

            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_v_in, k_chunk_tiles);
        v_start_tile_id += k_chunk_tiles;
        DPRINT << "[Reader] Finished Reading V at " << k_chunk << ENDL();
    }

    DPRINT << "[Reader] Done" << ENDL();
}
