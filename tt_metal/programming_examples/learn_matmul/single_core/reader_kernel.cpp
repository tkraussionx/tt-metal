#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {

    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src1_addr  = get_arg_val<uint32_t>(1);
    uint32_t M         = get_arg_val<uint32_t>(2);
    uint32_t K         = get_arg_val<uint32_t>(3);
    uint32_t N         = get_arg_val<uint32_t>(4);
    uint32_t dst_addr   = get_arg_val<uint32_t>(5);

    uint32_t Mt = M/32;
    uint32_t Nt = N/32;
    uint32_t Kt = K/32;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_out0 = 16;

    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat src1_data_format = get_dataformat(cb_id_in1);



       const InterleavedAddrGenFast<false> s0 = {
        .bank_base_address = src0_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };

       const InterleavedAddrGenFast<false> s1 = {
        .bank_base_address = src1_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };

    const InterleavedAddrGenFast<false> d1 = {
        .bank_base_address = dst_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };

    for(uint32_t i=0; i<Mt; i++)
    {
        for(uint32_t j=0; j<Nt; j++)
        {
            for(uint32_t k = 0; k<Kt; k++)
            {
                cb_reserve_back(cb_id_in0, 1);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(i*Kt + k, s0, l1_write_addr_in0);

                // noc_async_read_barrier();
                cb_push_back(cb_id_in0, 1);

                cb_reserve_back(cb_id_in1, 1);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(k*Nt + j, s1, l1_write_addr_in1);

                noc_async_read_barrier();
                cb_push_back(cb_id_in1, 1);
            }
            // cb_reserve_back(cb_id_in0, 1);
            // uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            // noc_async_read_tile(i, s0, l1_write_addr_in0);

            // noc_async_read_barrier();
            // cb_push_back(cb_id_in0, 1);

             cb_wait_front(cb_id_out0,1);
            noc_async_write_tile(i*Nt + j, d1, get_read_ptr(cb_id_out0));
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, 1);
        }
    }
}
