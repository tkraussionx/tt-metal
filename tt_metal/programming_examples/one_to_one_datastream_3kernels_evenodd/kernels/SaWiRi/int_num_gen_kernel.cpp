// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
	uint32_t dram_buffer  = get_arg_val<uint32_t>(0);
	uint32_t semaphore_write_local_addr  = get_semaphore(get_arg_val<uint32_t>(1));
	uint32_t semaphore_read_local_addr = get_semaphore(get_arg_val<uint32_t>(2));
	uint32_t core_gen_x_cord = get_arg_val<uint32_t>(3);
	uint32_t core_gen_y_cord = get_arg_val<uint32_t>(4);
	uint32_t core_read_x_cord = get_arg_val<uint32_t>(5);
	uint32_t core_read_y_cord = get_arg_val<uint32_t>(6);
	uint32_t dram_noc_x = get_arg_val<uint32_t>(7);
	uint32_t dram_noc_y = get_arg_val<uint32_t>(8);
	uint32_t loop_start = get_arg_val<uint32_t>(9);

	// Fetch the semaphore address converted for NOC operations
	uint64_t semaphore_read_noc_addr = get_noc_addr(core_read_x_cord, core_read_y_cord, semaphore_read_local_addr);
	volatile tt_l1_ptr uint32_t* semaphore_write_l1_address = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_write_local_addr);

	uint64_t dram_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, dram_buffer);

	constexpr uint32_t cb_write = tt::CB::c_in0; // index=0

	uint32_t l1_write_addr = get_write_ptr(cb_write);
    uint32_t* write_ptr;

	// Write stream
	for (int integer = loop_start; integer < 1001; integer += 2) {

		// Write the integer value to the TT-CircularBuffer cb_write
		write_ptr = (uint32_t*) l1_write_addr;
		*write_ptr = integer;

		noc_semaphore_wait(semaphore_write_l1_address, 1); // wait for reader kernel to signal that it is ready to read

		// Write data from TT-CircularBuffer cb_write to DRAM
		noc_async_write(l1_write_addr, dram_noc_addr, 4);

 		noc_semaphore_set(semaphore_write_l1_address, 0); // reset generator kernel's semaphore, so that it must wait until reader finishes reading and signals for generator for the next value
 		noc_semaphore_inc(semaphore_read_noc_addr, 1); // signal to read kernel that the generated value is ready to be read
	}
}
