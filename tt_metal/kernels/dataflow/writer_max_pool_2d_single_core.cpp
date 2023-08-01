#include <cstdint>
#include <algorithm>
#include "dataflow_api.h"
#include "debug_print.h"

#define MY_IF if(c_i > 3 && out_h_i > 80)

/**
 * Max-pool 2D. Highly Unoptimized!!
 */
void kernel_main() {
    // uint32_t in_addr = get_arg_val<uint32_t>(0);
    uint32_t out_addr = get_arg_val<uint32_t>(1);
    // uint32_t window_h = get_arg_val<uint32_t>(2);
    // uint32_t window_w = get_arg_val<uint32_t>(3);
    // uint32_t window_hw = get_arg_val<uint32_t>(4);
    // uint32_t stride_h = get_arg_val<uint32_t>(5);
    // uint32_t stride_w = get_arg_val<uint32_t>(6);
    // int32_t pad_h = get_arg_val<int32_t>(7);
    // int32_t pad_w = get_arg_val<int32_t>(8);
    int32_t out_nbatch = get_arg_val<int32_t>(9);
    int32_t out_nchannel = get_arg_val<int32_t>(10);
    int32_t out_h = get_arg_val<int32_t>(11);
    int32_t out_w = get_arg_val<int32_t>(12);
    // uint32_t in_nbytes_w = get_arg_val<uint32_t>(13);
    uint32_t out_nbytes_w = get_arg_val<uint32_t>(14);
    // int32_t in_nbatch = get_arg_val<int32_t>(15);
    // int32_t in_nchannel = get_arg_val<int32_t>(16);
    // int32_t in_h = get_arg_val<int32_t>(17);
    // int32_t in_w = get_arg_val<int32_t>(18);

    // constexpr bool is_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool is_out_dram = get_compile_time_arg_val(1) == 1;

    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    // ROW_MAJOR output
    const InterleavedAddrGen<is_out_dram> s_out = {
        .bank_base_address = out_addr,
        .page_size = out_nbytes_w
    };

    uint32_t out_row_id = 0;
    for (int32_t c_i = 0; c_i < out_nchannel; ++ c_i) {
        // for every output row
        for (int32_t out_h_i = 0; out_h_i < out_h; ++ out_h_i) {
            // uint32_t l1_read_addr = get_read_ptr(out_cb_id);
            // // wait for 1 row
            // MY_IF DPRINT << "QQQ " << ENDL();
            // cb_wait_front(out_cb_id, 1);
            // uint64_t out_noc_addr = get_noc_addr(out_row_id + out_h_i, s_out);
            // noc_async_write(l1_read_addr, out_noc_addr, out_nbytes_w);
            // noc_async_write_barrier();
            // // pop from CB
            // cb_pop_front(out_cb_id, 1);
            // MY_IF DPRINT << "PPPP :: " << (uint) c_i << " * " << (uint) out_h_i << ENDL();
        }
        out_row_id += out_h;
    }
} // kernel_main()
