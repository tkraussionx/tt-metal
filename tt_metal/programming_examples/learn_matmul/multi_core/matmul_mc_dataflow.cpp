#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {

    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src1_addr  = get_arg_val<uint32_t>(1);
    uint32_t Mt         = get_arg_val<uint32_t>(2);
    uint32_t Kt         = get_arg_val<uint32_t>(3);
    uint32_t Nt         = get_arg_val<uint32_t>(4);
    uint32_t dst_addr   = get_arg_val<uint32_t>(5);
    uint32_t core_x     = get_arg_val<uint32_t>(6);
    uint32_t core_y     = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_out0 = 16;


    const uint data_format = (uint) DataFormat::Float16_b;
    const uint tile_size_bytes = MUL_WITH_TILE_SIZE(data_format,1);

    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat src1_data_format = get_dataformat(cb_id_in1);

    const bool is_dram = true;

    const InterleavedAddrGenFast<is_dram> d1 = {
        .bank_base_address = dst_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };

    uint32_t k = 0;
    // for(uint32_t k = 0; k<Kt; k++)
    {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read(get_noc_addr(src0_addr), l1_write_addr_in0, tile_size_bytes);

        // // noc_async_read_barrier();

        cb_reserve_back(cb_id_in1, 1);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read(get_noc_addr(src1_addr), l1_write_addr_in1, tile_size_bytes);

        noc_async_read_barrier();

        // cb_wait_front(cb_id_out0,1);
        // noc_async_write_tile(0, d1, l1_write_addr_in1);
        // noc_async_write_barrier();
        // cb_pop_front(cb_id_out0, 1);

        cb_push_back(cb_id_in0, 1);
        cb_push_back(cb_id_in1, 1);
    }

    cb_wait_front(cb_id_out0,1);
    noc_async_write_tile(core_y*Nt + core_x, d1, get_read_ptr(cb_id_out0));
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, 1);
}
