// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include "dataflow_api.h"


void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t k_tensor_addr                       = get_arg_val<uint32_t>(0);
    uint32_t v_tensor_addr                       = get_arg_val<uint32_t>(1);
    uint32_t num_blocks                          = get_arg_val<uint32_t>(2);
    uint32_t kv_out_h_dim                        = get_arg_val<uint32_t>(3);
    uint32_t k_out_tensor_tile_id                = get_arg_val<uint32_t>(4);
    uint32_t v_out_tensor_tile_id                = get_arg_val<uint32_t>(5);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t out_is_dram               = get_compile_time_arg_val(0);
    constexpr uint32_t kv_out_h_tiles            = get_compile_time_arg_val(1);
    constexpr uint32_t kv_out_w_tiles            = get_compile_time_arg_val(2);
    constexpr uint32_t kv_out_HtWt               = get_compile_time_arg_val(3);
    constexpr uint32_t kv_out_c                  = get_compile_time_arg_val(4);


    constexpr uint32_t cb_id_v = 1; // cb for V heads tiles
    #ifdef TRANSPOSE_K_HEADS
    constexpr uint32_t cb_id_k = 16; // cb for K heads (filled by compute)
    #else
    constexpr uint32_t cb_id_k = 1; // cb for K heads (directly from reader)
    #endif
    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_v);
    const DataFormat data_format = get_dataformat(cb_id_v);

    constexpr bool out_is_dram_bool = out_is_dram == 1;

    const InterleavedAddrGenFast<out_is_dram_bool> sk = {
        .bank_base_address = k_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };
    const InterleavedAddrGenFast<out_is_dram_bool> sv = {
        .bank_base_address = v_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };

    constexpr uint32_t block_size = 1; // micro-block size for read/write; nothing to do with num_blocks
    // TODO: This might negatively impact perf
    constexpr uint32_t out_num_tiles_read = block_size; // always read and pop by micro-block size for generality
    uint32_t l1_read_addr;
    uint32_t k_out_tensor_current_tile_id; // need this to update k_out_tensor_tile_id
    uint32_t v_out_tensor_current_tile_id; // need this to update v_out_tensor_tile_id
    uint32_t out_tensor_current_tile_id_along_c;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // k + create k head --> outputs: [B, num_kv_heads, S, head_dim]
        #ifndef TRANSPOSE_K_HEADS
        out_tensor_current_tile_id_along_c = k_out_tensor_tile_id;
        #else
        k_out_tensor_current_tile_id = k_out_tensor_tile_id;
        #endif
        for (uint32_t c_dim = 0; c_dim < kv_out_c; c_dim++) {
            #ifndef TRANSPOSE_K_HEADS
            k_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            #endif
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                cb_wait_front(cb_id_k, out_num_tiles_read);
                l1_read_addr = get_read_ptr(cb_id_k);
                noc_async_write_tile(k_out_tensor_current_tile_id, sk, l1_read_addr);

                noc_async_write_barrier();
                cb_pop_front(cb_id_k, out_num_tiles_read);

                #ifndef TRANSPOSE_K_HEADS
                k_out_tensor_current_tile_id++;
                #else
                k_out_tensor_current_tile_id += q_out_h_tiles;
                #endif
            }
            #ifndef TRANSPOSE_K_HEADS
            out_tensor_current_tile_id_along_c += q_out_HtWt;
            #endif
        }

        // v + create v head --> outputs: [B, num_kv_heads, S, head_dim]
        out_tensor_current_tile_id_along_c = v_out_tensor_tile_id;
        for (uint32_t c_dim = 0; c_dim < kv_out_c; c_dim++) {
            v_out_tensor_current_tile_id = out_tensor_current_tile_id_along_c;
            for (uint32_t w_dim = 0; w_dim < q_out_w_tiles; w_dim++) {
                cb_wait_front(cb_id_v, out_num_tiles_read);
                l1_read_addr = get_read_ptr(cb_id_v);
                noc_async_write_tile(v_out_tensor_current_tile_id, sv, l1_read_addr);

                noc_async_write_barrier();
                cb_pop_front(cb_id_v, out_num_tiles_read);

                v_out_tensor_current_tile_id++;
            }
            out_tensor_current_tile_id_along_c += q_out_HtWt;
        }

        // Update out_tensor_tile_id for next h_dim or batch if we finish one CHtWt
        kv_out_h_dim++;
        if (kv_out_h_dim < kv_out_h_tiles) {
            #ifndef TRANSPOSE_K_HEADS
            k_out_tensor_tile_id += q_out_w_tiles;
            #else
            k_out_tensor_tile_id++;
            #endif
            v_out_tensor_tile_id += q_out_w_tiles;
        } else {
            // If we finish one batch, always roll over to next tile in memory
            // This is just the current_tile_id, except for K when we transpose heads
            // In this case, decrement k_out_tensor_current_tile_id by the stride (q_out_h_tiles) and add 1 to roll over
            #ifndef TRANSPOSE_K_HEADS
            k_out_tensor_tile_id = k_out_tensor_current_tile_id;
            #else
            k_out_tensor_tile_id = ++k_out_tensor_current_tile_id - q_out_h_tiles; // inc by 1 and decrement by stride
            #endif
            v_out_tensor_tile_id = v_out_tensor_current_tile_id;
            kv_out_h_dim = 0;
        }
    }

}
