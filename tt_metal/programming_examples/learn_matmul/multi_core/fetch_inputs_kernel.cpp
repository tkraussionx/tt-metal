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
    uint32_t core_x   = get_arg_val<uint32_t>(7);
    uint32_t core_y   = get_arg_val<uint32_t>(8);

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

    // for(uint32_t k = 0; k < 2; k++)
    {
        noc_async_read_tile(core_y*Kt, s0, dst0_addr);
        noc_async_read_barrier();
        noc_async_read_tile(core_x, s1, dst1_addr);
        noc_async_read_barrier();

        dst0_addr += tile_size_bytes;
        dst1_addr += tile_size_bytes;
    }
    DPRINT <<"Fetch Args : "<< src0_addr<<" "<<src1_addr<<" "<<Mt<<" "<<Kt<<" "<<Nt<<" "<<dst0_addr<<" "<<dst1_addr<<" "<<core_x<<" "<<core_y<<" "<< ENDL();



    // uint32_t Mt = M/32;
    // uint32_t Nt = N/32;
    // uint32_t Kt = K/32;

    // constexpr uint32_t cb_id_in0 = 0;
    // constexpr uint32_t cb_id_in1 = 1;
    // constexpr uint32_t cb_id_out0 = 16;

    // const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    // const DataFormat src0_data_format = get_dataformat(cb_id_in0);
    // const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    // const DataFormat src1_data_format = get_dataformat(cb_id_in1);

    // const bool is_dram = true;

    //    const InterleavedAddrGenFast<is_dram> s0 = {
    //     .bank_base_address = src0_addr,
    //     .page_size = src0_tile_bytes,
    //     .data_format = src0_data_format
    // };

    //    const InterleavedAddrGenFast<is_dram> s1 = {
    //     .bank_base_address = src1_addr,
    //     .page_size = src0_tile_bytes,
    //     .data_format = src0_data_format
    // };

    // const InterleavedAddrGenFast<is_dram> d1 = {
    //     .bank_base_address = dst_addr,
    //     .page_size = src0_tile_bytes,
    //     .data_format = src0_data_format
    // };

}
