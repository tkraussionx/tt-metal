// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

FORCE_INLINE uint16_t float16_to_uint16(uint16_t float16_val) {
    // Masks for float16 format
    const uint16_t exponent_mask = 0x7C00;  // Mask for the 5-bit exponent (bits 10-14)
    const uint16_t mantissa_mask = 0x03FF;  // Mask for the 10-bit mantissa (bits 0-9)
    const int exponent_bias = 15;           // Bias for the exponent in float16
    const int mantissa_shift = 10;          // Number of bits in the mantissa

    // Extract the exponent and mantissa
    uint16_t exponent = (float16_val & exponent_mask) >> mantissa_shift;  // Extract and shift to get the exponent
    uint16_t mantissa = float16_val & mantissa_mask;                      // Extract the mantissa

    // Handle special cases: zero and subnormal numbers
    if (exponent == 0) {
        return 0;  // Zero or subnormal numbers (treat as zero for simplicity)
    }

    // Calculate the resulting uint16_t
    // For normalized values, the formula is: (1 + mantissa / 1024) * 2^(exponent - 15)
    // Simplified as: (1 << (exponent - 15)) + (mantissa << (exponent - 25))

    // Shift exponent to remove the bias
    int shift_amount = exponent - exponent_bias;

    // Calculate the result using integer shifts
    uint16_t result = (1 << shift_amount) + (mantissa >> (mantissa_shift - shift_amount));

    return result;
}

void kernel_main() {

    constexpr bool SKIP = false;

    // Compile time args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(5);

    constexpr uint32_t batch = get_compile_time_arg_val(6);

    // All Gather specific
    constexpr uint32_t ring_size = get_compile_time_arg_val(7);
    uint32_t signal_semaphore_addr = get_semaphore(get_compile_time_arg_val(8));

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_y = get_arg_val<uint32_t>(rt_args_idx++);

    volatile tt_l1_ptr uint32_t* l1_signal_sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);
    uint64_t remote_signal_semaphore_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, signal_semaphore_addr);

    constexpr uint32_t cb_id_in0 = 0;
    // constexpr uint32_t cb_id_in2 = 2;  // Sharded cb

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t shard_size_bytes = shard_size_in_tiles* in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;

    // // Dprint some info
    // DPRINT << "[" << ring_idx << "] ring_size: " << ring_size << ENDL();
    // DPRINT << "[" << ring_idx << "] signal_semaphore_addr: " << signal_semaphore_addr << ENDL();
    DPRINT << "[" << ring_idx << "] to: (" << next_core_noc_x << ", " << next_core_noc_y << ")" << ENDL();
    // DPRINT << "[" << ring_idx << "] in0_block_num_tiles: " << in0_block_num_tiles << ENDL();
    // DPRINT << "[" << ring_idx << "] in0_single_tile_size_bytes: " << in0_single_tile_size_bytes << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_width_in_tiles: " << shard_width_in_tiles << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_height_in_tiles: " << shard_height_in_tiles << ENDL();
    DPRINT << "[" << ring_idx << "] shard_size_bytes: " << shard_size_bytes << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_read_stride: " << shard_read_stride << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_read_width: " << shard_read_width << ENDL();


    // cb_reserve_back(cb_id_in2, batch * shard_size_in_tiles);
    cb_reserve_back(cb_id_in0, batch * ring_size * shard_size_in_tiles);

    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t local_shard_read_addr = get_read_ptr(cb_id_in0);

    DPRINT << "[" << ring_idx << "] l1_write_addr_in0: " << l1_write_addr_in0 << ENDL();
    DPRINT << "[" << ring_idx << "] local_shard_read_addr: " << local_shard_read_addr << ENDL();

    for (uint32_t b = 0; b < batch; ++b) {

        for (uint32_t shard_cnt = 0; shard_cnt < ring_size && !SKIP; shard_cnt++) {

            uint32_t curr_shard_write_addr = l1_write_addr_in0 + shard_size_bytes * (shard_cnt + 1);
            uint64_t remote_curr_shard_write_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, curr_shard_write_addr);
            uint32_t curr_shard_read_addr = l1_write_addr_in0 + shard_size_bytes * shard_cnt;

            // Wait for signal from previous core that data has been added to this core's in0
            noc_semaphore_wait_min(l1_signal_sem_addr, shard_cnt + 1);

            if (shard_cnt == 0) { // Need to load the local shard from cb2 to cb0 in the correct place
                // noc_async_read(get_noc_addr(local_shard_read_addr), curr_shard_read_addr, shard_size_bytes);
                // noc_async_read_barrier();

                // noc_async_write_one_packet_set_state(remote_curr_shard_write_addr, shard_size_bytes);
            }
            if ((ring_idx == 1 && shard_cnt == 0) || (ring_idx == 0 && shard_cnt == 1)) {
                // Dprint all the elements inside the shard starting at curr_shard_read addr
                volatile tt_l1_ptr uint16_t* curr_shard_read_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(curr_shard_read_addr);

                for (uint32_t i = 1; i < 32*32*3; i++) {
                    uint16_t float_val = curr_shard_read_addr_ptr[i];
                    uint16_t uint16_val = float16_to_uint16(float_val);
                    DPRINT << uint16_val << ", ";
                }
            }

            // Do stuff for matmul fusion here


            /* Here, assume cb0 has the data the data ready in the correct place. */

            // Send data to next core
            if (shard_cnt < ring_size - 1) { // Skip sending the last shard

                // DPRINT << "shard_cnt write idx: " << shard_cnt + 1 << ENDL();
                DPRINT << "[" << ring_idx << "] Writing shard at address: " << remote_curr_shard_write_addr << ENDL();
                noc_async_write(curr_shard_read_addr, remote_curr_shard_write_addr, shard_size_bytes);
                // noc_async_write_barrier();
                // noc_async_write_one_packet_with_state(curr_shard_read_addr, remote_curr_shard_write_addr);

                // Signal the next core that data is ready
                noc_semaphore_inc(remote_signal_semaphore_addr, 1);
            }
       }
        cb_push_back(cb_id_in0, shard_size_in_tiles * ring_size);
    }
}
