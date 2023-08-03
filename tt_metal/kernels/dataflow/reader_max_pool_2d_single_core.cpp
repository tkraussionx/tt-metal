#include <cstdint>
#include <algorithm>
#include "dataflow_api.h"
// #include "debug_print.h"

// #define MY_IF if(c_i > 3 && out_h_i > 80)
#define MY_IF


SliceRange sr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 4, .w0 = 0, .w1 = 32, .ws = 1 };

uint16_t max_bfloat16(const uint16_t arr[], int n) {

    kernel_profiler::mark_time(8);

    uint16_t max_positive = 0;      // smallest +ve
    uint16_t max_negative = 0xff7f; // smallest -ve
    uint16_t any_non_zero = 0;      // flag to note if everything is 0
    uint16_t sign_bit_mask = 0x8000;
    for (int i = 0; i < n; ++i) {
        any_non_zero |= arr[i];
        // 0x800 mask for the sign bit
        if (arr[i] & sign_bit_mask) {
            // the number is negative, check if it is closer to 0
            max_negative = (max_negative > arr[i]) ? arr[i] : max_negative;
        } else {
            // the number if positive, check if it is larger than max_positive
            max_positive = (arr[i] > max_positive) ? arr[i] : max_positive;
        }
    }

    kernel_profiler::mark_time(9);

    // if all inputs are 0, return 0
    if (any_non_zero == 0) return 0;
    // return the max
    return max_positive > 0 ? max_positive : max_negative;
} // max_bfloat16()


/**
 * Max-pool 2D. Highly Unoptimized!!
 * TODO [AS]: reuse data moved to L1 instead of reading every time
 */
void kernel_main() {
    kernel_profiler::mark_time(7);

    uint32_t in_addr = get_arg_val<uint32_t>(0);
    uint32_t out_addr = get_arg_val<uint32_t>(1);
    uint32_t window_h = get_arg_val<uint32_t>(2);
    uint32_t window_w = get_arg_val<uint32_t>(3);
    uint32_t window_hw_ceil4 = get_arg_val<uint32_t>(4);
    uint32_t stride_h = get_arg_val<uint32_t>(5);
    uint32_t stride_w = get_arg_val<uint32_t>(6);
    int32_t pad_h = get_arg_val<int32_t>(7);
    int32_t pad_w = get_arg_val<int32_t>(8);
    int32_t out_nbatch = get_arg_val<int32_t>(9);
    int32_t out_nchannel = get_arg_val<int32_t>(10);
    int32_t out_h = get_arg_val<int32_t>(11);
    int32_t out_w = get_arg_val<int32_t>(12);
    uint32_t in_nbytes_w = get_arg_val<uint32_t>(13);
    uint32_t out_nbytes_w = get_arg_val<uint32_t>(14);
    int32_t in_nbatch = get_arg_val<int32_t>(15);
    int32_t in_nchannel = get_arg_val<int32_t>(16);
    int32_t in_h = get_arg_val<int32_t>(17);
    int32_t in_w = get_arg_val<int32_t>(18);
    uint32_t out_pagesize = get_arg_val<int32_t>(19);
    uint32_t out_pagesize_tile_aligned = get_arg_val<int32_t>(20);

    constexpr bool is_in_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    // ROW_MAJOR input
    const InterleavedAddrGen<is_in_dram> s_in = {
        .bank_base_address = in_addr,
        .page_size = in_nbytes_w
    };

    uint16_t bfloat16_arr[window_hw_ceil4];   // temporary array to hold at most one window worth of values

    uint32_t start_in_row_id = 0;
    uint32_t out_row_id = 0;

    // bool one_time_noc_wait = false;
    // bool one_time_cb_push = false;

    const uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
    const uint32_t in_l1_read_addr = get_read_ptr(in_cb_id);
    uint16_t* in_data_ptr = reinterpret_cast<uint16_t*>(in_l1_read_addr);

    for (int32_t c_i = 0; c_i < out_nchannel; ++ c_i) {
        int32_t start_h = - pad_h;
        int32_t read_nrows = window_h - pad_h; // initialize the number of input rows to read into L1
        int32_t in_row_id = start_in_row_id - pad_h;
        // for every output row
        for (int32_t out_h_i = 0; out_h_i < out_h; ++ out_h_i) {
            // read at most window_h input rows from DRAM into L1 (CB)
            // ignore rows from previous channel
            uint32_t curr_row_id = in_row_id < c_i * in_h ? c_i * in_h : in_row_id;
            uint32_t curr_row_l1_addr = in_l1_write_addr;
            // DPRINT << "OUT ROW " << (uint) out_h_i << " l1_write_ptr = " << curr_row_l1_addr << ENDL();
            for(int32_t h = 0; h < read_nrows; ++ h) {
                uint64_t in_noc_addr = get_noc_addr(curr_row_id, s_in);
                noc_async_read(in_noc_addr, curr_row_l1_addr, in_nbytes_w);
                curr_row_l1_addr += in_nbytes_w;
                ++ curr_row_id;
            }
            noc_async_read_barrier();
            // update reading counters for windows corresponding to the next output row
            in_row_id += stride_h;
            // use current data in L1 (CB) to construct output row
            int32_t start_w = - pad_w;
            const uint32_t out_l1_write_addr = get_write_ptr(out_cb_id);
            uint16_t* out_data_ptr = reinterpret_cast<uint16_t*>(out_l1_write_addr);
            cb_reserve_back(out_cb_id, 1);      // make sure one row is available to write in output cb
            // for every output col
            for (int32_t out_w_i = 0; out_w_i < out_w; ++ out_w_i) {
                uint16_t* curr_data_ptr = in_data_ptr;
                // start = {start_h, start_w}
                int32_t end_h = start_h + window_h;
                int32_t end_w = start_w + window_w;
                // populate the array to find max
                uint32_t arr_i = 0;
                int32_t start_h_max = start_h < 0 ? 0 : start_h;
                int32_t start_w_max = start_w < 0 ? 0 : start_w;
                int32_t end_h_min = end_h < in_h ? end_h : in_h;
                int32_t end_w_min = end_w < in_w ? end_w : in_w;
                // MY_IF DPRINT << "Window: " << (uint) start_h_max << "," << (uint) start_w_max << " --> " << (uint) end_h_min << "," << (uint) end_w_min << ENDL();

                kernel_profiler::mark_time(10);

                for (int32_t h = start_h_max; h < end_h_min; ++ h) {
                    for (int32_t w = start_w_max; w < end_w_min; ++ w) {
                        bfloat16_arr[arr_i ++] = curr_data_ptr[w];
                        // TODO [AS]: load 32b data at a time instead with uint32_t ptr (or 64b with uint64_t?)
                        // Need to make sure the window width boundary is taken care of since it may be odd.
                    }
                    curr_data_ptr += in_w;   // next row (num 16b elements)
                }

                kernel_profiler::mark_time(11);

                uint16_t max_val = max_bfloat16(bfloat16_arr, arr_i);
                out_data_ptr[out_w_i] = max_val;
                start_w += stride_w;
            }
            // push into output CB
            cb_push_back(out_cb_id, 1);
            start_h += stride_h;
            read_nrows = window_h;      // TODO: fix for general padding vals
        }
        start_in_row_id += in_h;
        out_row_id += out_h;
    }
} // kernel_main()
