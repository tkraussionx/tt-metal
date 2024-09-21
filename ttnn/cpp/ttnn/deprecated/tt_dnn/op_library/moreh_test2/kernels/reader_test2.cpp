#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    constexpr bool input1_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool cond_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool input2_is_dram = get_compile_time_arg_val(2) == 1;

    constexpr uint32_t cb_input1 = tt::CB::c_in0;
    constexpr uint32_t cb_cond = tt::CB::c_in1;
    constexpr uint32_t cb_input2 = tt::CB::c_in2;

    const uint32_t input1_addr = get_arg_val<uint32_t>(0);
    const uint32_t cond_addr = get_arg_val<uint32_t>(1);
    const uint32_t input2_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t tiles_offset = get_arg_val<uint32_t>(4);

    const uint32_t input1_tile_size = get_tile_size(cb_input1);
    const DataFormat input1_data_format = get_dataformat(cb_input1);
    const uint32_t cond_tile_size = get_tile_size(cb_cond);
    const DataFormat cond_data_format = get_dataformat(cb_cond);
    const uint32_t input2_tile_size = get_tile_size(cb_input2);
    const DataFormat input2_data_format = get_dataformat(cb_input2);

    const InterleavedAddrGenFast<input1_is_dram> input1_s = {
        .bank_base_address = input1_addr, .page_size = input1_tile_size, .data_format = input1_data_format};
    const InterleavedAddrGenFast<cond_is_dram> cond_s = {
        .bank_base_address = cond_addr, .page_size = cond_tile_size, .data_format = cond_data_format};
    const InterleavedAddrGenFast<input2_is_dram> input2_s = {
        .bank_base_address = input2_addr, .page_size = input2_tile_size, .data_format = input2_data_format};

    for (uint32_t i = tiles_offset; i < tiles_offset + num_tiles; i++) {
        cb_reserve_back(cb_input1, onetile);
        cb_reserve_back(cb_cond, onetile);
        cb_reserve_back(cb_input2, onetile);

        const uint32_t cb_input1_addr = get_write_ptr(cb_input1);
        const uint32_t cb_cond_addr = get_write_ptr(cb_cond);
        const uint32_t cb_input2_addr = get_write_ptr(cb_input2);
        noc_async_read_tile(i, input1_s, cb_input1_addr);
        noc_async_read_tile(i, cond_s, cb_cond_addr);
        noc_async_read_tile(i, input2_s, cb_input2_addr);
        noc_async_read_barrier();

        cb_push_back(cb_input1, onetile);
        cb_push_back(cb_cond, onetile);
        cb_push_back(cb_input2, onetile);
    }
}
