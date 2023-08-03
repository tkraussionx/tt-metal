#include <cstdint>
#include <algorithm>
#include "dataflow_api.h"
// #include "debug_print.h"

#define MY_IF if(c_i > 3 && out_h_i > 80)

/**
 * Max-pool 2D. Highly Unoptimized!!
 */
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(1);
    int32_t out_nbatch = get_arg_val<int32_t>(9);
    int32_t out_nchannel = get_arg_val<int32_t>(10);
    int32_t out_h = get_arg_val<int32_t>(11);
    int32_t out_w = get_arg_val<int32_t>(12);
    uint32_t out_nbytes_w = get_arg_val<uint32_t>(14);
    uint32_t out_pagesize = get_arg_val<int32_t>(19);
    uint32_t out_pagesize_tile_aligned = get_arg_val<int32_t>(19);

    constexpr bool is_out_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    // ROW_MAJOR output
    const InterleavedAddrGen<is_out_dram> s_out = {
        .bank_base_address = out_addr,
        .page_size = out_pagesize
    };

    uint32_t out_row_id = 0;
    for (int32_t c_i = 0; c_i < out_nchannel; ++ c_i) {
        // for every output row
        for (int32_t out_h_i = 0; out_h_i < out_h; ++ out_h_i) {
            uint32_t out_l1_read_addr = get_read_ptr(out_cb_id);
            uint64_t out_noc_addr = get_noc_addr(out_row_id + out_h_i, s_out);
            cb_wait_front(out_cb_id, 1);
            noc_async_write(out_l1_read_addr, out_noc_addr, out_nbytes_w);
            noc_async_write_barrier();
            cb_pop_front(out_cb_id, 1);
        }
        out_row_id += out_h;
    }
} // kernel_main()
