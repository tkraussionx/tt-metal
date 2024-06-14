#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

const uint32_t physical_corex_map[]={1,2,3,4,6,7,8,9};
const uint32_t physical_corey_map[]={1,2,3,4,5,7,8,9};

void kernel_main() {

    // same arg indices as in reader_binary_diff_lenghts for compat
    const uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr  = get_arg_val<uint32_t>(1);
    const uint32_t Mt         = get_arg_val<uint32_t>(2);
    const uint32_t Kt         = get_arg_val<uint32_t>(3);
    const uint32_t Nt         = get_arg_val<uint32_t>(4);
    const uint32_t dst0_addr   = get_arg_val<uint32_t>(5);
    const uint32_t dst1_addr   = get_arg_val<uint32_t>(6);
    const uint32_t start_Mt = get_arg_val<uint32_t>(7);
    const uint32_t start_Nt = get_arg_val<uint32_t>(8);
    const uint32_t this_core_Mt = get_arg_val<uint32_t>(9);
    const uint32_t this_core_Nt = get_arg_val<uint32_t>(10);
    const uint32_t core_x = get_arg_val<uint32_t>(11);
    const uint32_t core_y = get_arg_val<uint32_t>(12);
    const uint32_t core_grid_x = get_arg_val<uint32_t>(13);
    const uint32_t core_grid_y = get_arg_val<uint32_t>(14);
    uint32_t in0_mcast_sender_semaphore = get_arg_val<uint32_t>(15);
    uint32_t in0_mcast_receiver_semaphore = get_arg_val<uint32_t>(16);
    uint32_t in1_mcast_sender_semaphore = get_arg_val<uint32_t>(17);
    uint32_t in1_mcast_receiver_semaphore = get_arg_val<uint32_t>(18);
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

    if(core_x==0)
    {
        uint32_t current_dst0_addr = dst0_addr;
        for(uint32_t current_Mt = start_Mt; current_Mt<(start_Mt+this_core_Mt);current_Mt++)
        {
            for(uint32_t k = 0; k < Kt; k++)
            {
                noc_async_read_tile(current_Mt*Kt + k, s0, current_dst0_addr);
                current_dst0_addr += tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        if(core_grid_x>1)
        {
            noc_semaphore_wait((volatile tt_l1_ptr uint32_t*)in0_mcast_sender_semaphore,core_grid_x-1); //NOC Wait 0
            noc_semaphore_set((volatile tt_l1_ptr uint32_t*)in0_mcast_sender_semaphore, 0);

            uint64_t in0_multicast_data_addr = get_noc_multicast_addr(
                physical_corex_map[1],
                physical_corey_map[core_y],
                physical_corex_map[core_grid_x-1],
                physical_corey_map[core_y],
                dst0_addr);

            noc_async_write_multicast(dst0_addr, in0_multicast_data_addr, tile_size_bytes*Kt*this_core_Mt, core_grid_x-1);

            uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
                physical_corex_map[1],
                physical_corey_map[core_y],
                physical_corex_map[core_grid_x-1],
                physical_corey_map[core_y],
                (uint32_t)in0_mcast_receiver_semaphore);
            noc_semaphore_set((volatile tt_l1_ptr uint32_t*)in0_mcast_receiver_semaphore, VALID);

            noc_async_write_multicast((uint32_t)in0_mcast_receiver_semaphore, in0_mcast_receiver_semaphore_noc_addr, 4, core_grid_x-1);
            // DPRINT_DATA0(DPRINT << "NOC 0  MCast Grid Start: " <<physical_corex_map[1]<<" "<<physical_corey_map[core_y]<<" End: "<<physical_corex_map[core_grid_x-1]<<" "<<physical_corey_map[core_y]<<ENDL());
            // DPRINT_DATA0(DPRINT<<"Num Rx "<<core_grid_x-1<<ENDL());
        }
    }
    else
    {
        noc_semaphore_set((volatile tt_l1_ptr uint32_t*)in0_mcast_receiver_semaphore, INVALID);

        const uint32_t in0_mcast_sender_noc_x = physical_corex_map[0];
        const uint32_t in0_mcast_sender_noc_y = physical_corey_map[core_y];
        uint64_t in0_mcast_sender_semaphore_noc_addr = get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y,(uint32_t) in0_mcast_sender_semaphore);
        noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1); //NOC Inc 0

        noc_semaphore_wait((volatile tt_l1_ptr uint32_t*)in0_mcast_receiver_semaphore, VALID);


    }

    if(core_y==0)
    {
        uint32_t current_dst1_addr = dst1_addr;
        for(uint32_t current_Nt = start_Nt; current_Nt<(start_Nt+this_core_Nt);current_Nt++)
        {
            for(uint32_t k = 0; k < Kt; k++)
            {
                noc_async_read_tile(k*Nt + current_Nt, s1, current_dst1_addr);
                current_dst1_addr += tile_size_bytes;
            }
        }
        noc_async_read_barrier();
        if(core_grid_y>1)
        {
            noc_semaphore_wait((volatile tt_l1_ptr uint32_t*)in1_mcast_sender_semaphore,core_grid_y-1); //NOC Wait 1
            noc_semaphore_set((volatile tt_l1_ptr uint32_t*)in1_mcast_sender_semaphore, 0);
            uint64_t in1_multicast_data_addr = get_noc_multicast_addr(
                physical_corex_map[core_x],
                physical_corey_map[1],
                physical_corex_map[core_x],
                physical_corey_map[core_grid_y-1],
                dst1_addr);
            noc_async_write_multicast(dst1_addr, in1_multicast_data_addr, tile_size_bytes*Kt*this_core_Nt, core_grid_y-1);
            uint64_t in1_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
                physical_corex_map[core_x],
                physical_corey_map[1],
                physical_corex_map[core_x],
                physical_corey_map[core_grid_y-1],
                (uint32_t)in1_mcast_receiver_semaphore);
            noc_semaphore_set((volatile tt_l1_ptr uint32_t*)in1_mcast_receiver_semaphore, VALID);

            noc_async_write_multicast((uint32_t)in1_mcast_receiver_semaphore, in1_mcast_receiver_semaphore_noc_addr, 4, core_grid_y-1);

            // DPRINT_DATA0(DPRINT << "NOC 0  NCast Grid Start: " <<physical_corex_map[core_x]<<physical_corey_map[1]<<" End:"<<physical_corex_map[core_x]<<physical_corey_map[core_grid_y-1]<<ENDL());
            // DPRINT_DATA0(DPRINT<<"Num Rx "<<core_grid_x-1<<ENDL());
        }

    }
    else
    {
        const uint32_t in1_mcast_sender_noc_x = physical_corex_map[core_x];
        const uint32_t in1_mcast_sender_noc_y = physical_corey_map[0];
        uint64_t in1_mcast_sender_semaphore_noc_addr = get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y,(uint32_t) in1_mcast_sender_semaphore);
        noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1); //NOC Inc 1

    }

}
