#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {

    // same arg indices as in reader_binary_diff_lenghts for compat
    const uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr  = get_arg_val<uint32_t>(1);
    const uint32_t Mt         = get_arg_val<uint32_t>(2);
    const uint32_t Kt         = get_arg_val<uint32_t>(3);
    const uint32_t Nt         = get_arg_val<uint32_t>(4);
    const uint32_t dst_addr   = get_arg_val<uint32_t>(5);
    const uint32_t start_Mt = get_arg_val<uint32_t>(6);
    const uint32_t start_Nt = get_arg_val<uint32_t>(7);
    const uint32_t this_core_Mt = get_arg_val<uint32_t>(8);
    const uint32_t this_core_Nt = get_arg_val<uint32_t>(9);

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

    for(uint32_t current_Mt = 0; current_Mt<this_core_Mt;current_Mt++)
    {
        for(uint32_t current_Nt = 0; current_Nt<this_core_Nt;current_Nt++)
        {
            uint32_t k = 0;
            int current_in0_addr = src0_addr +current_Mt*Kt*tile_size_bytes;
            int current_in1_addr = src1_addr +current_Nt*Kt*tile_size_bytes;

            for(uint32_t k = 0; k<Kt; k++)
            {

                cb_reserve_back(cb_id_in0, 1);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read(get_noc_addr(current_in0_addr), l1_write_addr_in0, tile_size_bytes);


                cb_reserve_back(cb_id_in1, 1);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read(get_noc_addr(current_in1_addr), l1_write_addr_in1, tile_size_bytes);
                noc_async_read_barrier();

                cb_push_back(cb_id_in0, 1);
                cb_push_back(cb_id_in1, 1);
                current_in0_addr += tile_size_bytes;
                current_in1_addr += tile_size_bytes;

            }
            cb_wait_front(cb_id_out0,1);
            noc_async_write_tile((current_Mt+start_Mt)*Nt + current_Nt + start_Nt, d1, get_read_ptr(cb_id_out0));
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, 1);
        }
    }
}
