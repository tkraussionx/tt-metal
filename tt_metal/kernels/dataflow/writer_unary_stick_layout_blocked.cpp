#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {

    constexpr uint32_t cb_id_out0                      = 0;

    uint32_t dst_addr                 = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x                 = get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y                 = get_arg_val<uint32_t>(2);
    uint32_t num_blocks               = get_arg_val<uint32_t>(3);
    uint32_t block_size_tiles               = get_arg_val<uint32_t>(4);
    uint32_t block_size_bytes               = get_arg_val<uint32_t>(5);


    for (uint32_t b = 0; b < num_blocks; b++) {
        cb_wait_front(cb_id_out0, block_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);
        noc_async_write(l1_read_addr, dst_noc_addr, block_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, block_size_tiles);
    }

}
