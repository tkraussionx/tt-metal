// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "risc_attribs.h"
//#include "debug/dprint.h"

FORCE_INLINE
bool cb_producer_space_available(int32_t num_pages) {
    uint32_t operand = 0;
    uint32_t pages_acked_ptr = (uint32_t) get_cb_tiles_acked_ptr(operand);

    // while the producer (write-side interface) is waiting for space to free up "tiles_pushed" is not changing
    // "tiles_pushed" is updated by the producer only when the tiles are pushed
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    DEBUG_STATUS('C', 'R', 'B', 'W');

    // uint16_t's here because Tensix updates the val at tiles_acked_ptr as uint16 in llk_pop_tiles
    // TODO: I think we could have TRISC update tiles_acked_ptr, and we wouldn't need uint16 here
    uint16_t pages_acked = (uint16_t)reg_read(pages_acked_ptr);
    uint16_t free_space_pages_wrap =
        cb_interface[operand].fifo_num_pages - (pages_received - pages_acked);
    free_space_pages = (int32_t)free_space_pages_wrap;
    return free_space_pages >= num_pages;
}

FORCE_INLINE
uint32_t min(uint32_t a, uint32_t b) { return (a < b) ? a: b; }

FORCE_INLINE
bool cb_consumer_space_available(bool db_buf_switch, int32_t num_pages) {

    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_acked = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_ack_addr(db_buf_switch));
    uint16_t pages_recv = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_recv_addr(db_buf_switch));
    uint32_t num_pages_consumer = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_num_pages_addr(db_buf_switch));

    uint16_t free_space_pages_wrap = num_pages_consumer - (pages_recv - pages_acked);
    int32_t free_space_pages = (int32_t)free_space_pages_wrap;
    DEBUG_STATUS('C', 'R', 'B', 'D');

    return free_space_pages >= num_pages;
}

FORCE_INLINE
void multicore_cb_push_back(uint64_t consumer_noc_encoding, uint32_t consumer_fifo_limit, uint32_t consumer_fifo_size, bool db_buf_switch, uint32_t page_size, uint32_t num_to_write) {
    // TODO(agrebenisan): Should create a multi-core CB interface... struct in L1
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_RECV_PTR = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_recv_addr(db_buf_switch));
    volatile tt_l1_ptr uint32_t* CQ_CONSUMER_CB_WRITE_PTR = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_wr_ptr_addr(db_buf_switch));

    *CQ_CONSUMER_CB_RECV_PTR += num_to_write;
    *CQ_CONSUMER_CB_WRITE_PTR += (page_size * num_to_write) >> 4;

    if ((*CQ_CONSUMER_CB_WRITE_PTR << 4) >= consumer_fifo_limit) {
        *CQ_CONSUMER_CB_WRITE_PTR -= consumer_fifo_size >> 4;
    }

    uint32_t pages_recv_addr = get_db_cb_recv_addr(db_buf_switch);
    noc_semaphore_set_remote(uint32_t(CQ_CONSUMER_CB_RECV_PTR), consumer_noc_encoding | pages_recv_addr);
}

FORCE_INLINE
void relay_command(bool db_buf_switch, uint64_t consumer_noc_encoding) {
    /*
        Relays the current command to the consumer.
    */

    uint64_t consumer_command_slot_addr = consumer_noc_encoding | get_command_slot_addr(db_buf_switch);
    noc_async_write(L1_UNRESERVED_BASE, consumer_command_slot_addr, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND);
    noc_async_write_barrier();
}



FORCE_INLINE
void produce(
    volatile tt_l1_ptr uint32_t* command_ptr,
    uint32_t num_srcs,
    uint32_t num_cores,
    uint32_t page_size,
    uint32_t producer_cb_size,
    uint32_t producer_cb_num_pages,
    uint32_t consumer_cb_size,
    uint32_t consumer_cb_num_pages,
    uint64_t consumer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages,
    bool db_buf_switch
    ) {
    /*
        This API prefetches data from host memory and writes data to the consumer core. On the consumer,
        we partition the data space into 2 via double-buffering. There are two command slots, and two
        data slots. The producer reads in data into its local buffer and checks whether it can write to
        the consumer. It continues like this in a loop, context switching between pulling in data and
        writing to the consumer.
    */
    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
    uint32_t l1_consumer_fifo_limit = get_db_buf_addr(db_buf_switch) + consumer_cb_size;

    uint hack_val = 0;

    for (uint32_t i = 0; i < num_srcs; i++) {
        const uint32_t bank_base_address = command_ptr[0];
        uint32_t num_pages = command_ptr[2];
        const uint32_t page_size = command_ptr[3];
        const uint32_t src_buf_type = command_ptr[4];
        //If reading from system memory can read all at once and not one core at a time
        if((BufferType)src_buf_type == BufferType::SYSTEM_MEMORY){
            num_cores = 1;
        }

        uint32_t total_reads_issued = 0;
        uint32_t total_num_writes_completed = 0;
        uint32_t total_num_reads_completed = 0;
        for(uint32_t core_id = 0; core_id < num_cores; core_id++){
            uint32_t core_id_x = 1;
            uint32_t core_id_y = 1;
            if(num_cores > 1){
                num_pages = command_ptr[6+(int)core_id*3];
                core_id_x = command_ptr[7+(int)core_id*3];
                core_id_y = command_ptr[8+(int)core_id*3];
            }
            uint32_t fraction_of_producer_cb_num_pages = consumer_cb_num_pages / 2;

            uint32_t num_to_read = min(num_pages, fraction_of_producer_cb_num_pages);
            //uint32_t num_pages_left_in_consumer_cb = (l1_consumer_fifo_limit + 1 - dst_addr) / page_size;
            uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages); // This must be a bigger number for perf.
            //num_to_write = min(num_to_write, num_pages_left_in_consumer_cb);
            uint32_t num_reads_issued_core = 0;
            uint32_t num_reads_completed_core = 0;
            uint32_t num_writes_completed_core = 0;

            while (num_writes_completed_core != num_pages) {
                // Context switch between reading in pages and sending them to the consumer.
                // These APIs are non-blocking to allow for context switching.
                DPRINT << "NUM TO READ " << num_to_read << ENDL();
                DPRINT << "NUM READS ISSUED CORE " << num_reads_issued_core << ENDL();
                DPRINT << "NUM WRITES COMPLETED CORE " << num_writes_completed_core << ENDL();
                DPRINT << "TOTAL NUM READS ISSUED  " << total_reads_issued << ENDL();
                DPRINT << "TOTAL NUM WRITES COMPLETED " << total_num_writes_completed << ENDL();
                DPRINT << "CB_PRODUCER_SPACE_AVAILABLE " << (uint32_t)cb_producer_space_available(num_to_read) << ENDL();
                DPRINT << "NUM_PAGES " << num_pages << ENDL();
                if (cb_producer_space_available(num_to_read) and num_reads_issued_core < num_pages) {
                    uint32_t l1_write_ptr = get_write_ptr(0);
                    //DPRINT << "GOT WRITE PTR " << ENDL();
                    if((BufferType)src_buf_type == BufferType::SYSTEM_MEMORY){
                        Buffer src_buffer((BufferType)src_buf_type, bank_base_address, page_size);
                        src_buffer.noc_async_read_buffer(l1_write_ptr, total_reads_issued, num_to_read, 0);

                    }else{
                        ShardedBuffer src_buffer(page_size, bank_base_address);
                        src_buffer.noc_async_read_buffer(l1_write_ptr, num_to_read, num_reads_issued_core, core_id_x, core_id_y);
                    }
                    //DPRINT << "FINISHED WRITE " << ENDL();
                    cb_push_back(0, num_to_read);
                    DPRINT << "FINISHED PUSH_BACK " << ENDL();
                    num_reads_issued_core += num_to_read;
                    total_reads_issued += num_to_read;
                    uint32_t num_pages_left = num_pages - num_reads_issued_core;
                    num_to_read = min(num_pages_left, fraction_of_producer_cb_num_pages);
                }
                DPRINT << "FINISHED READ_CB_SECTION " << ENDL();

                uint32_t dst_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_wr_ptr_addr(db_buf_switch))[0] << 4;
                DPRINT << "FINISHED GETTING DST_ADDR " << ENDL();
                uint32_t num_pages_left_in_consumer_cb = (l1_consumer_fifo_limit + 1 - dst_addr) / page_size;
                DPRINT << "NUM_PAGES_LEFT_IN_CONSUMER_CB_WITH_DST_ADDR " << num_pages_left_in_consumer_cb << ENDL();

                num_to_write = min(num_to_write, num_pages_left_in_consumer_cb);

                if (num_reads_issued_core > num_writes_completed_core and cb_consumer_space_available(db_buf_switch, num_to_write)) {
                    if (num_writes_completed_core == num_reads_completed_core) {
                        noc_async_read_barrier();
                        num_reads_completed_core = num_reads_issued_core;
                    }

                    uint64_t dst_noc_addr = consumer_noc_encoding | dst_addr;
                    uint32_t l1_read_ptr = get_read_ptr(0);
                    noc_async_write(l1_read_ptr, dst_noc_addr, page_size * num_to_write);
                    multicore_cb_push_back(consumer_noc_encoding, l1_consumer_fifo_limit, consumer_cb_size, db_buf_switch, page_size, num_to_write);
                    noc_async_write_barrier();
                    cb_pop_front(0, num_to_write);

                    DPRINT << "PROD NUM TO WRITE: " << num_to_write << " to " << dst_addr << ENDL();
                    if(num_to_write > 0){
                        // DPRINT << "PRODUCER WRITING " << num_to_write << " PAGES TO CONSUMER CB " << ENDL();
                        uint32_t * ptr = (uint32_t *)l1_read_ptr;
                        for (uint32_t i = 0; i < num_to_write; i++) {
                            if((BufferType)src_buf_type == BufferType::SYSTEM_MEMORY){
                                DPRINT << "WRITING PAGE TO SYSTEM MEMORY : " << hack_val++ << " FIRST ELEMENT " << DEC() << ptr[i*page_size/4] << ENDL();
                            }
                            else{
                                DPRINT << "WRITING PAGE TO SHARDED BUFFER : " << hack_val++ << " FIRST ELEMENT " << DEC() << ptr[i*page_size/4] << ENDL();

                            }
                        }
                    }

                    num_writes_completed_core += num_to_write;
                    total_num_writes_completed += num_to_write;
                    num_to_write = min(num_pages - num_writes_completed_core, producer_consumer_transfer_num_pages);
                }
            }
        }
        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
    }
}

FORCE_INLINE
void produce_sharded(
    volatile tt_l1_ptr uint32_t* command_ptr,
    uint32_t num_cores,
    uint32_t num_srcs, uint32_t page_size, uint32_t producer_cb_size, uint32_t producer_cb_num_pages,
    uint32_t consumer_cb_size, uint32_t consumer_cb_num_pages, uint64_t consumer_noc_encoding, uint32_t producer_consumer_transfer_num_pages, bool db_buf_switch) {
    /*
        This API prefetches data from host memory and writes data to the consumer core. On the consumer,
        we partition the data space into 2 via double-buffering. There are two command slots, and two
        data slots. The producer reads in data into its local buffer and checks whether it can write to
        the consumer. It continues like this in a loop, context switching between pulling in data and
        writing to the consumer.
    */
    command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
    uint32_t l1_consumer_fifo_limit = get_db_buf_addr(db_buf_switch) + consumer_cb_size;


    for (uint32_t i = 0; i < num_srcs; i++) {
        const uint32_t bank_base_address = command_ptr[0];
        const uint32_t page_size = command_ptr[3];
        const uint32_t src_buf_type = command_ptr[4];


        uint32_t total_reads_issued = 0;
        for(uint32_t core_id = 0; core_id < num_cores; core_id++){
            const uint32_t num_pages = command_ptr[6+core_id*3];
            const uint32_t core_id_x = command_ptr[7+core_id*3];
            const uint32_t core_id_y = command_ptr[8+core_id*3];
            uint32_t fraction_of_producer_cb_num_pages = consumer_cb_num_pages / 2;
            uint32_t num_to_read = min(num_pages, fraction_of_producer_cb_num_pages);
            DPRINT << "PRODUCER: core_id_x " << DEC() << core_id_x << " core_id_y " << core_id_y << " num_pages " << num_pages << ENDL();
            uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages); // This must be a bigger number for perf.
            uint32_t num_reads_completed_core = 0;
            uint32_t num_writes_completed_core = 0;
            uint32_t num_reads_issued_core = 0;
            while (num_writes_completed_core != (num_pages)) {
                // Context switch between reading in pages and sending them to the consumer.
                // These APIs are non-blocking to allow for context switching.
                if (cb_producer_space_available(num_to_read) and num_reads_issued_core < num_pages) {
                    cb_reserve_back(0, num_to_read);
                    uint32_t l1_write_ptr = get_write_ptr(0);
                    if((BufferType)src_buf_type == BufferType::SYSTEM_MEMORY){
                        Buffer src_buffer((BufferType)src_buf_type, bank_base_address, page_size);
                        src_buffer.noc_async_read_buffer(l1_write_ptr, total_reads_issued, num_to_read, 0);

                    }else{
                        ShardedBuffer src_buffer_sharded(page_size, bank_base_address);
                        src_buffer_sharded.noc_async_read_buffer(l1_write_ptr, num_to_read, num_reads_issued_core, core_id_x, core_id_y);
                    }
                    noc_async_read_barrier();
                    if(num_to_write > 0){
                        DPRINT << "PRODUCER READING " << num_to_write <<  ENDL();
                        DPRINT << "PRODUCER:  NUM_READS_ISSUED " << num_writes_completed_core << ENDL();
                        uint32_t * ptr = (uint32_t *)l1_write_ptr;
                        for (uint32_t i = 0; i < num_to_write; i++) {
                            DPRINT << "PRODUCER WRITING TO CB : PAGE: " << i << " FIRST ELEMENT " << DEC() << ptr[i*page_size/4] << ENDL();
                        }
                    }


                    cb_push_back(0, num_to_read);
                    num_reads_issued_core += num_to_read;
                    total_reads_issued += num_to_read;
                    uint32_t num_pages_left = num_pages - num_reads_issued_core;
                    num_to_read = min(num_pages_left, fraction_of_producer_cb_num_pages);
                }
                if (num_reads_issued_core > num_writes_completed_core and cb_consumer_space_available(db_buf_switch, num_to_write)) {
                    if (num_writes_completed_core == num_reads_completed_core) {
                        noc_async_read_barrier();
                        num_reads_completed_core = num_reads_issued_core;
                    }

                    uint32_t dst_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_db_cb_wr_ptr_addr(db_buf_switch))[0] << 4;
                    uint64_t dst_noc_addr = consumer_noc_encoding | dst_addr;

                    cb_wait_front(0, num_to_write);
                    uint32_t l1_read_ptr = get_read_ptr(0);

                    noc_async_write(l1_read_ptr, dst_noc_addr, page_size * num_to_write);
                    if(num_to_write > 0){
                        DPRINT << "PRODUCER WRITING " << num_to_write << " PAGES TO MULTICORE_CB " << ENDL();
                        DPRINT << "PRODUCER:  NUM_WRITES COMPLETED " << num_writes_completed_core << ENDL();
                        uint32_t * ptr = (uint32_t *)l1_read_ptr;
                        for (uint32_t i = 0; i < num_to_write; i++) {
                            DPRINT << "PRODUCER WRITING TO NOC : PAGE: " << i << " FIRST ELEMENT " << DEC() << ptr[i*page_size/4] << ENDL();
                        }
                    }

                    multicore_cb_push_back(consumer_noc_encoding, l1_consumer_fifo_limit, consumer_cb_size, db_buf_switch, page_size, num_to_write);
                    noc_async_write_barrier();
                    cb_pop_front(0, num_to_write);
                    num_writes_completed_core += num_to_write;
                    num_to_write = min(num_pages - num_writes_completed_core, producer_consumer_transfer_num_pages);
                }

            }
        }
        command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
    }
}
