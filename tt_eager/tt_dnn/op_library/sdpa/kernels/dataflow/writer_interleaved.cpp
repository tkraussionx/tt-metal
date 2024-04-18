// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// #include "debug/dprint.h"

void kernel_main() {
    uint32_t out_addr  = get_arg_val<uint32_t>(0);
    uint32_t B         = get_arg_val<uint32_t>(1);
    uint32_t NQH         = get_arg_val<uint32_t>(2);
    uint32_t St       = get_arg_val<uint32_t>(3);
    uint32_t DHt      = get_arg_val<uint32_t>(4);
    uint32_t S_chunk_t    = get_arg_val<uint32_t>(5);
    uint32_t num_chunks    = get_arg_val<uint32_t>(6);
    uint32_t core_id    = get_arg_val<uint32_t>(7);
    uint32_t num_cores    = get_arg_val<uint32_t>(8);

    const uint32_t out_chunk_tiles = S_chunk_t * DHt;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CB::c_out0;

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    uint32_t out_tile_id = 0;

    for (uint32_t nb = 0; nb < B; ++nb) {
        // DPRINT << "WRITER: "  << "nb=" << nb << ENDL();
        for (uint32_t nq = 0; nq < NQH; ++nq) {
            if (nq != core_id) {
                continue;
            }
            out_tile_id = nq * St * DHt;
            // DPRINT << "WRITER: "  << "nq=" << nq << ENDL();
            for (uint32_t q_chunk = 0; q_chunk < num_chunks; ++q_chunk) {
                // DPRINT << "WRITER: "  << "q_chunk=" << q_chunk << ENDL();
                // Wait for compute to deliver output chunk
                cb_wait_front(cb_out, out_chunk_tiles);
                uint32_t l1_read_addr = get_read_ptr(cb_out);
                for (uint32_t tile = 0; tile < out_chunk_tiles; ++tile) {
                    // DPRINT << "WRITER: "  << "out_tile_id=" << out_tile_id << ENDL();
                    noc_async_write_tile(out_tile_id, out_writer, l1_read_addr);
                    ++out_tile_id;
                    l1_read_addr += tile_bytes;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, out_chunk_tiles);
            }
        }
    }
}
