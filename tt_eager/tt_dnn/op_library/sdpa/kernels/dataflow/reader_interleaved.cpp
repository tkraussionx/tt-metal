// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "tt_eager/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t q_addr  = get_arg_val<uint32_t>(0);
    uint32_t k_addr  = get_arg_val<uint32_t>(1);
    uint32_t v_addr         = get_arg_val<uint32_t>(2);
    uint32_t mask_addr         = get_arg_val<uint32_t>(3);
    uint32_t B         = get_arg_val<uint32_t>(4);
    uint32_t NQH         = get_arg_val<uint32_t>(5);
    uint32_t NKH       = get_arg_val<uint32_t>(6);
    uint32_t St       = get_arg_val<uint32_t>(7);
    uint32_t DHt      = get_arg_val<uint32_t>(8);
    uint32_t S_chunk_t    = get_arg_val<uint32_t>(9);
    uint32_t num_chunks    = get_arg_val<uint32_t>(10);
    uint32_t scale_val    = get_arg_val<uint32_t>(11);

    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(0);

    DPRINT << "READER: scale_val: " << scale_val << ENDL();

    const uint32_t q_chunk_tiles = S_chunk_t * DHt;
    const uint32_t k_chunk_tiles = S_chunk_t * DHt;
    const uint32_t mask_chunk_tiles = S_chunk_t * S_chunk_t;

    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;
    constexpr uint32_t cb_mask_in = tt::CB::c_in3;
    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    generate_bcast_unary_scalar(cb_scale_in, scale_val);
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    constexpr uint32_t onetile = 1;
    const uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    const DataFormat q_data_format = get_dataformat(cb_q_in);
    const uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    const DataFormat k_data_format = get_dataformat(cb_k_in);
    const uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    const DataFormat v_data_format = get_dataformat(cb_v_in);
    const uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    const DataFormat mask_data_format = get_dataformat(cb_mask_in);



    const InterleavedAddrGenFast<is_dram> q_reader = {
        .bank_base_address = q_addr,
        .page_size = q_tile_bytes,
        .data_format = q_data_format
    };

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

    const InterleavedAddrGenFast<is_dram> mask_reader = {
        .bank_base_address = mask_addr,
        .page_size = mask_tile_bytes,
        .data_format = mask_data_format
    };

    uint32_t q_tile_id = 0;
    uint32_t k_tile_id = 0;
    uint32_t v_tile_id = 0;
    uint32_t mask_tile_id = 0;

    for (uint32_t nb = 0; nb < B; ++nb) {
        // DPRINT << "READER: "  << "nb=" << nb << ENDL();
        for (uint32_t nq = 0; nq < NQH; ++nq) {
            // DPRINT << "READER: "  << "nq=" << nq << ENDL();
            for (uint32_t q_chunk = 0; q_chunk < num_chunks; ++q_chunk) {
                // DPRINT << "READER: "  << "q_chunk=" << q_chunk << ENDL();
                // Read Q chunk
                cb_reserve_back(cb_q_in, q_chunk_tiles);
                uint32_t q_write_ptr = get_write_ptr(cb_q_in);

                for (uint32_t tile = 0; tile < q_chunk_tiles; ++tile) {
                    // DPRINT << "READER: "  << "q_tile_id=" << q_tile_id << ENDL();
                    noc_async_read_tile(q_tile_id, q_reader, q_write_ptr);
                    q_tile_id += 1;
                    q_write_ptr += q_tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_q_in, q_chunk_tiles);


                for (uint32_t k_chunk = 0; k_chunk < num_chunks; ++k_chunk) {
                    // DPRINT << "READER: "  << "k_chunk=" << k_chunk << ENDL();
                    // TODO: index based off of BATCH as well
                    k_tile_id = k_chunk * S_chunk_t;
                    v_tile_id = k_chunk * S_chunk_t * DHt;
                    // Read K chunk
                    cb_reserve_back(cb_k_in, k_chunk_tiles);
                    uint32_t k_write_ptr = get_write_ptr(cb_k_in);

                    for (uint32_t row = 0; row < DHt; ++row) {
                        for (uint32_t col = 0; col < S_chunk_t; ++col) {
                            // DPRINT << "READER: "  << "k_tile_id=" << k_tile_id << ENDL();
                            noc_async_read_tile(k_tile_id, k_reader, k_write_ptr);
                            k_tile_id += 1;
                            k_write_ptr += k_tile_bytes;
                        }
                        // Strid along columns to get to next row
                        k_tile_id -= S_chunk_t;
                        k_tile_id += St;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_k_in, k_chunk_tiles);


                    // Read mask chunk
                    cb_reserve_back(cb_mask_in, mask_chunk_tiles);
                    uint32_t mask_write_ptr = get_write_ptr(cb_mask_in);
                    mask_tile_id = nq * St * St /*head_offset*/ + q_chunk * S_chunk_t * St /*row_offset*/ + k_chunk * S_chunk_t /*col_offset*/;
                    for (uint32_t row = 0; row < S_chunk_t; ++row) {
                        for (uint32_t col = 0; col < S_chunk_t; ++col) {
                            // DPRINT << "READER: "  << "mask_tile_id=" << mask_tile_id << ENDL();
                            noc_async_read_tile(mask_tile_id, mask_reader, mask_write_ptr);
                            mask_tile_id += 1;
                            mask_write_ptr += mask_tile_bytes;
                        }
                        // Strid along columns to get to next row
                        mask_tile_id -= S_chunk_t;
                        mask_tile_id += St;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_mask_in, mask_chunk_tiles);



                    // Read V chunk
                    cb_reserve_back(cb_v_in, k_chunk_tiles);
                    uint32_t v_write_ptr = get_write_ptr(cb_v_in);

                    for (uint32_t tile = 0; tile < k_chunk_tiles; ++tile) {
                        // DPRINT << "READER: "  << "v_tile_id=" << v_tile_id << ENDL();
                        noc_async_read_tile(v_tile_id, v_reader, v_write_ptr);
                        v_tile_id += 1;
                        v_write_ptr += v_tile_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_v_in, k_chunk_tiles);
                }
            }
        }
    }
}
