#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_output = tt::CB::c_out0;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t tiles_offset = get_arg_val<uint32_t>(2);

    const uint32_t tile_size = get_tile_size(cb_output);
    const DataFormat data_format = get_dataformat(cb_output);

    const InterleavedAddrGenFast<output_is_dram> s = {
        .bank_base_address = output_addr, .page_size = tile_size, .data_format = data_format};

    for (uint32_t i = tiles_offset; i < tiles_offset + num_tiles; i++) {
        cb_wait_front(cb_output, onetile);
        const uint32_t cb_output_addr = get_read_ptr(cb_output);
        noc_async_write_tile(i, s, cb_output_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output, onetile);
    }
}
