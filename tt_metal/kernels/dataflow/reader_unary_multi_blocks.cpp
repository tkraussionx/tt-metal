#include <stdint.h>
#include "dataflow_api.h"
#include "debug_print.h"
void kernel_main() {
    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Arguments for in0
    uint32_t in0_addr_base  = get_arg_val<uint32_t>(1);
    uint32_t in0_noc_x = get_arg_val<uint32_t>(2);
    uint32_t in0_noc_y = get_arg_val<uint32_t>(3);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(4);
    uint32_t num_reads_per_block = get_arg_val<uint32_t>(5);
    uint32_t read_size_bytes = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = 0;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);
    //DPRINT << "A=" << num_blocks << ENDL();

    uint32_t index = 0;
    uint32_t x = 0;
    // DPRINT << "in0_noc_x=" << in0_noc_x << ENDL();
    // DPRINT << "in0_noc_y=" << in0_noc_y << ENDL();
    //DPRINT << "B=" << num_blocks << ENDL();
    for (uint32_t b = 0; b < num_blocks; b += 1) {
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        // Read from DRAM into L1 using DTX address map and push one block at a time to CB
        //DPRINT << "P" << ENDL();

        //DPRINT << "Q" << ENDL();
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        for(uint_32_t i = 0; i < num_reads_per_block; i++) {
            // There are 4 entries in the address map per read
            uint32_t src_address = address_map[index];
            uint32_t dst_address = address_map[index+1];
            uint32_t read_size = address_map[index+2];
            uint32_t pad = address_map[index+3];
            // if (index == 0) {
            // DPRINT << "S=" << src_address << ENDL();
            // DPRINT << "D=" << dst_address << ENDL();
            // DPRINT << "R=" << read_size << ENDL();
            // DPRINT << "pad=" << pad << ENDL();
            // }
            //if(pad == 1) {
                // //DPRINT << "PAD" << ENDL();
                // // source address is set to max. This refers to padding location.
                // // read zeroes from zero buffer
                // uint32_t dst_addr = l1_write_addr_in0 + dst_address;
                // //DPRINT << "dst_add=" << dst_addr;
                // // if (dst_addr < 200 * 1024) {
                // //     DPRINT << "PROBLEM" << ENDL();
                // // }
                // uint32_t pad_size = read_size;
                // if (pad_size <= l1_mem::address_map::ZEROS_SIZE) {
                //     // if(pad_size < 16) {
                //     //         DPRINT << "PROBLEM" << ENDL();
                //     // }
                //     noc_async_read(zeros_base_noc_addr, dst_addr, pad_size);
                // }
                // else {
                //     // padding size is bigger than the zero buffer size
                //     // read from zero buffer multiple times
                //     uint32_t zeros_to_read = pad_size;
                //     uint32_t zeros_read_size = l1_mem::address_map::ZEROS_SIZE;
                //     while(zeros_to_read != 0) {
                //         // if(zeros_read_size < 16) {
                //         //     DPRINT << "PROBLEM" << ENDL();
                //         // }
                //         noc_async_read(zeros_base_noc_addr, dst_addr, zeros_read_size);
                //         zeros_to_read -= zeros_read_size;
                //         if (zeros_to_read < zeros_read_size) {
                //             zeros_read_size = zeros_to_read;
                //         }
                //     }
                // }
            //}
            //else {
                //DPRINT << "NOT PAD" << ENDL();
                uint32_t src_addr = in0_addr_base + src_address;
                uint64_t src_noc_addr = get_noc_addr(in0_noc_x, in0_noc_y, src_addr);
                //DPRINT << "src_addr=" << src_addr;
                if(in0_noc_x > 1 || in0_noc_y > 1 || src_addr > 401408) {
                    //DPRINT << "Problem" << ENDL();
                }
                uint32_t dst_addr = l1_write_addr_in0 + dst_address;
                //DPRINT << "dst_add=" << dst_addr;

                // if (x == 0 && (dst_addr < 208 * 1024 || dst_addr + read_size > 1024 * 1024)) {
                //     DPRINT << "dst_addr=" << dst_addr << ENDL();
                //     DPRINT << "read_size=" << dst_addr << ENDL();
                //     DPRINT << "Problem" << ENDL();
                //     x = 1;
                // }
                // for(volatile int i = 0; i <50000; i++) {
                //     ;
                // }

                noc_async_read(src_noc_addr, dst_addr, read_size);
            //}
            //bytes_read += read_size;
            index += 3;
        }
        // if(bytes_read != in0_block_size_bytes) {
        //     DPRINT << "PROBLEM" << ENDL();
        // }
        noc_async_read_barrier();
        //DPRINT << "S" << ENDL();
        cb_push_back(cb_id_in0, in0_block_num_tiles);
    }
}
