#include "dataflow_kernel_api.h"

void kernel_main() {
    uint32_t src_addr           = dataflow::get_arg_val<uint32_t>(0);
    uint32_t src_noc_x          = dataflow::get_arg_val<uint32_t>(1);
    uint32_t src_noc_y          = dataflow::get_arg_val<uint32_t>(2);
    uint32_t src_buffer_size    = dataflow::get_arg_val<uint32_t>(3);

    uint32_t local_addr         = dataflow::get_arg_val<uint32_t>(4);

    uint32_t dst_addr           = dataflow::get_arg_val<uint32_t>(5);
    uint32_t dst_noc_x_start    = dataflow::get_arg_val<uint32_t>(6);
    uint32_t dst_noc_y_start    = dataflow::get_arg_val<uint32_t>(7);
    uint32_t dst_noc_x_end      = dataflow::get_arg_val<uint32_t>(8);
    uint32_t dst_noc_y_end      = dataflow::get_arg_val<uint32_t>(9);
    uint32_t num_dests          = dataflow::get_arg_val<uint32_t>(10);


    // Read src buffer into local L1 buffer
    uint64_t src_buffer_noc_addr = dataflow::get_noc_addr(src_noc_x, src_noc_y, src_addr);
    dataflow::noc_async_read(src_buffer_noc_addr, local_addr, src_buffer_size);
    dataflow::noc_async_read_barrier();

    // multicast local L1 buffer to all destination cores
    uint64_t dst_noc_multicast_addr = dataflow_internal::get_noc_multicast_addr(
        dst_noc_x_start,
        dst_noc_y_start,
        dst_noc_x_end,
        dst_noc_y_end,
        dst_addr);
    dataflow::noc_async_write_multicast_loopback_src(local_addr, dst_noc_multicast_addr, src_buffer_size, num_dests);
    dataflow::noc_async_write_barrier();
}
