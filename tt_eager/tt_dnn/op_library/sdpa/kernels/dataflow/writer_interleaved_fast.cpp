// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t out_addr  = get_arg_val<uint32_t>(0);
    uint32_t B         = get_arg_val<uint32_t>(1);
    uint32_t NQH         = get_arg_val<uint32_t>(2);
    uint32_t St       = get_arg_val<uint32_t>(3);
    uint32_t DHt      = get_arg_val<uint32_t>(4);
    uint32_t Sq_chunk_t    = get_arg_val<uint32_t>(5);
    uint32_t q_num_chunks    = get_arg_val<uint32_t>(6);
    uint32_t Sk_chunk_t    = get_arg_val<uint32_t>(7);
    uint32_t k_num_chunks    = get_arg_val<uint32_t>(8);
    uint32_t core_id    = get_arg_val<uint32_t>(9);
    uint32_t num_cores    = get_arg_val<uint32_t>(10);
    uint32_t q_parallel_factor    = get_arg_val<uint32_t>(11);

    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(0);
    constexpr uint32_t scale_val = get_compile_time_arg_val(1);

    const uint32_t num_local_q_chunks = q_num_chunks / q_parallel_factor;
    const uint32_t local_batch = core_id / (NQH * q_parallel_factor);
    const uint32_t local_q_head = (core_id / q_parallel_factor) % NQH;
    const uint32_t local_q_chunk_start = num_local_q_chunks * (core_id % q_parallel_factor);
    const uint32_t local_q_chunk_end = local_q_chunk_start + num_local_q_chunks;

    // const uint32_t my_q_head = core_id / num_chunks;
    // const uint32_t my_q_chunk = core_id % num_chunks;

    const uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CB::c_out0;

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    uint32_t out_tile_id = 0;

    for (uint32_t nb = 0; nb < B; ++nb) {
        // DPRINT << "WRITER: "  << "nb=" << nb << ENDL();
        if (nb != local_batch) {
            continue;
        }
        for (uint32_t nq = 0; nq < NQH; ++nq) {
            if (nq != local_q_head) {
                continue;
            }
            // DPRINT << "WRITER: "  << "nq=" << nq << ENDL();
            // for (uint32_t q_chunk = local_q_chunk_start; q_chunk < local_q_chunk_end; ++q_chunk) {
            uint32_t q_chunk = local_q_chunk_start;
            // for (uint32_t q_chunk = local_q_chunk_start; q_chunk < local_q_chunk_end; ++q_chunk) {
            for (uint32_t g = 0; g < 2; ++g) {
                if (g == 1) {
                    q_chunk = q_num_chunks - local_q_chunk_start - 1;
                }
                uint32_t q_head_offset = nq * St * DHt;
                uint32_t q_chunk_offset = q_chunk * Sq_chunk_t * DHt;
                out_tile_id = q_head_offset + q_chunk_offset;

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
