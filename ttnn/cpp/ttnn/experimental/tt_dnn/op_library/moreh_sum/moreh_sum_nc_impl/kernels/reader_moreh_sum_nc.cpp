// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/experimental/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/experimental/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "debug/dprint.h"

inline uint32_t get_read_tile_id(uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size) {
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
}

// inline void print_cb_details(uint32_t cb_id) {
//         DPRINT << "cb_id " << cb_id << ": { "
//                 << "size: " << cb_interface[cb_id].fifo_size << ", "
//                 << "limit: " << cb_interface[cb_id].fifo_limit << ", "
//                 << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
//                 << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
//                 << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
//                 << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
//                 << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL();
// }

void kernel_main() {
    // compile-time args
    constexpr bool input_is_dram = (get_compile_time_arg_val(0) == 1);

    // runtime args
    ArgFetcher arg_fetcher;
    const auto input_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto num_input_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto num_output_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto dim = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto reduce_tile_size = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto inner_tile_size = arg_fetcher.get_next_arg_val<uint32_t>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = 0;

    // print_cb_details(0);
    // print_cb_details(24);

    #ifdef USE_FPU
    constexpr uint32_t cb_id_intermed0 = 24;
    constexpr uint32_t scaler = 0;
    generate_reduce_scaler(cb_id_intermed0, scaler);
    #endif
    // print_cb_details(24);

    uint32_t l1_write_addr_in0;
    uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    const auto input_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        auto read_tile_id = (dim == 0) ? (i) : (get_read_tile_id(i, reduce_tile_size, inner_tile_size));
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(read_tile_id, input_addrg, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            // print_cb_details(0);
            read_tile_id += inner_tile_size;
        }
    }
}
