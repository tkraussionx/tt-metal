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
    uint32_t dst0_addr   = get_arg_val<uint32_t>(5);
    uint32_t dst1_addr   = get_arg_val<uint32_t>(6);
    uint32_t start_Mt = get_arg_val<uint32_t>(7);
    uint32_t start_Nt = get_arg_val<uint32_t>(8);
    uint32_t this_core_Mt = get_arg_val<uint32_t>(9);
    uint32_t this_core_Nt = get_arg_val<uint32_t>(10);
    // const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    // const DataFormat src0_data_format = get_dataformat(cb_id_in0);
    // const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    // const DataFormat src1_data_format = get_dataformat(cb_id_in1);
    const bool is_dram = true;

    const uint data_format = (uint) DataFormat::Float16_b;
    const uint tile_size_bytes = MUL_WITH_TILE_SIZE(data_format,1);
    const InterleavedAddrGenFast<is_dram> s0 = {
        .bank_base_address = src0_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b
    };

    const InterleavedAddrGenFast<is_dram> s1 = {
        .bank_base_address = src1_addr,
        .page_size =  tile_size_bytes,
        .data_format = DataFormat::Float16_b
    };

    for(uint32_t current_Mt = start_Mt; current_Mt<(start_Mt+this_core_Mt);current_Mt++)
    {
        for(uint32_t k = 0; k < Kt; k++)
        {
            noc_async_read_tile(current_Mt*Kt + k, s0, dst0_addr);
            dst0_addr += tile_size_bytes;
        }
    }

    for(uint32_t current_Nt = start_Nt; current_Nt<(start_Nt+this_core_Nt);current_Nt++)
    {
        for(uint32_t k = 0; k < Kt; k++)
        {
            noc_async_read_tile(k*Nt + current_Nt, s1, dst1_addr);

            dst1_addr += tile_size_bytes;
        }
    }
    noc_async_read_barrier();

}
