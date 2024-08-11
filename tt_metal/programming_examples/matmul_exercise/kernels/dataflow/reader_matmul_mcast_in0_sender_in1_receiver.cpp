// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr                    = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id           = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w                = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h                = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride       = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w                        = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h                        = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles                = get_arg_val<uint32_t>(7);

    // in1 tensor args
    uint32_t in1_tensor_addr                    = get_arg_val<uint32_t>(8);
    uint32_t in1_tensor_start_tile_id           = get_arg_val<uint32_t>(9);
    uint32_t in1_tensor_stride_w                = get_arg_val<uint32_t>(10);
    uint32_t in1_tensor_stride_h                = get_arg_val<uint32_t>(11);
    uint32_t in1_tensor_next_block_stride       = get_arg_val<uint32_t>(12);

    // in1 block args
    uint32_t in1_block_w                        = get_arg_val<uint32_t>(13);
    uint32_t in1_block_h                        = get_arg_val<uint32_t>(14);
    uint32_t in1_block_num_tiles                = get_arg_val<uint32_t>(15);

    // in0/in1 common args
    uint32_t num_blocks                         = get_arg_val<uint32_t>(16);

    // in0 mcast args
    uint32_t in0_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(17);
    uint32_t in0_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(18);
    uint32_t in0_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(19);
    uint32_t in0_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(20);
    uint32_t in0_mcast_num_dests                = get_arg_val<uint32_t>(21);
    uint32_t in0_mcast_sender_noc_x             = get_arg_val<uint32_t>(22);
    uint32_t in0_mcast_sender_noc_y             = get_arg_val<uint32_t>(23);
    uint32_t in0_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(24);
    uint32_t in0_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(25);

    // in1 mcast args
    uint32_t in1_mcast_dest_noc_start_x         = get_arg_val<uint32_t>(26);
    uint32_t in1_mcast_dest_noc_start_y         = get_arg_val<uint32_t>(27);
    uint32_t in1_mcast_dest_noc_end_x           = get_arg_val<uint32_t>(28);
    uint32_t in1_mcast_dest_noc_end_y           = get_arg_val<uint32_t>(29);
    uint32_t in1_mcast_num_dests                = get_arg_val<uint32_t>(30);
    uint32_t in1_mcast_sender_noc_x             = get_arg_val<uint32_t>(31);
    uint32_t in1_mcast_sender_noc_y             = get_arg_val<uint32_t>(32);
    uint32_t in1_mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(33);
    uint32_t in1_mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(34);

    constexpr bool in0_is_dram                        = get_compile_time_arg_val(0) == 1;
    constexpr bool in1_is_dram                        = get_compile_time_arg_val(1) == 1; // not used

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    uint32_t l1_write_addr_in0;


    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);
    *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready
    // to receive the mcast
    volatile tt_l1_ptr uint32_t* in0_mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_sender_semaphore_addr);


    volatile tt_l1_ptr uint32_t* in1_mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_mcast_receiver_semaphore_addr);

    const InterleavedAddrGenFast<in0_is_dram> s0 = {
        .bank_base_address = in0_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };

    uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
    for(uint32_t block = 0; block < num_blocks; block++) {
        //////////////////////////////////////////////////////////////////////////////////////////////
        // TODO: Implement the in0 sender logic
        //////////////////////////////////////////////////////////////////////////////////////////////
        // Operand 0
        /*



        */

        //////////////////////////////////////////////////////////////////////////////////////////////
        // TODO: Implement the in1 receiver logic
        //////////////////////////////////////////////////////////////////////////////////////////////
        // Operand 1
        /*


        */
    }
}
