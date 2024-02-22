// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/cq_prefetcher.hpp"
#include "debug/dprint.h"

void kernel_main() {
    bool db_buf_switch = false;
    constexpr uint32_t host_issue_queue_read_ptr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t issue_queue_start_addr = get_compile_time_arg_val(1);
    constexpr uint32_t issue_queue_size = get_compile_time_arg_val(2);
    constexpr uint32_t host_completion_queue_write_ptr_addr = get_compile_time_arg_val(3);
    constexpr uint32_t completion_queue_start_addr = get_compile_time_arg_val(4);
    constexpr uint32_t completion_queue_size = get_compile_time_arg_val(5);
    constexpr uint32_t host_finish_addr = get_compile_time_arg_val(6);
    constexpr uint32_t command_start_addr = get_compile_time_arg_val(7);
    constexpr uint32_t data_section_addr = get_compile_time_arg_val(8);
    constexpr uint32_t data_buffer_size = get_compile_time_arg_val(9);
    constexpr uint32_t consumer_cmd_base_addr = get_compile_time_arg_val(10);
    constexpr uint32_t consumer_data_buffer_size = get_compile_time_arg_val(11);
    constexpr tt::PullAndPushConfig pull_and_push_config = (tt::PullAndPushConfig)get_compile_time_arg_val(12);

    constexpr bool read_from_issue_queue = (pull_and_push_config == tt::PullAndPushConfig::LOCAL or pull_and_push_config == tt::PullAndPushConfig::PUSH_TO_REMOTE);

    setup_issue_queue_read_interface(issue_queue_start_addr, issue_queue_size);
    setup_completion_queue_write_interface(completion_queue_start_addr, completion_queue_size);

    // Initialize the producer/consumer DB semaphore
    // This represents how many buffers the producer can write to.
    // At the beginning, it can write to two different buffers.
    uint64_t my_noc_encoding = uint64_t(NOC_XY_ENCODING(my_x[0], my_y[0])) << 32;
    uint64_t pull_noc_encoding = uint64_t(NOC_XY_ENCODING(PULL_NOC_X, PULL_NOC_Y)) << 32;
    uint64_t push_noc_encoding = uint64_t(NOC_XY_ENCODING(PUSH_NOC_X, PUSH_NOC_Y)) << 32;
    uint64_t dispatch_noc_encoding = uint64_t(NOC_XY_ENCODING(DISPATCH_NOC_X, DISPATCH_NOC_Y)) << 32;
    uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_ENCODING(PCIE_NOC_X, PCIE_NOC_Y)) << 32;

    volatile tt_l1_ptr uint32_t* push_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));  // Should be initialized to num commnd slots by host
    volatile tt_l1_ptr uint32_t *pull_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));  // Should be initialized to 0 by host

    volatile tt_l1_ptr uint32_t *debug =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));

    // DPRINT << "SEM INIT: " << push_semaphore_addr[0] << ENDL();

    PullAndRelayCfg src_pr_cfg;
    PullAndRelayCfg dst_pr_cfg;

    volatile db_cb_config_t* local_multicore_cb_cfg = get_local_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);
    volatile db_cb_config_t* dispatch_multicore_cb_cfg = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, db_buf_switch);

    // TODO: Need to support this for REMOTE_PULL_AND_PUSH to allow sending programs to dispatch core
    if constexpr (pull_and_push_config == tt::PullAndPushConfig::LOCAL) {
        // DPRINT << "Programming cb for programs on dispatch core" << ENDL();
        constexpr uint32_t remote_cb_num_pages = (MEM_L1_SIZE - L1_UNRESERVED_BASE - DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND * 2) / DeviceCommand::PROGRAM_PAGE_SIZE;
        constexpr uint32_t remote_cb_size = remote_cb_num_pages * DeviceCommand::PROGRAM_PAGE_SIZE;

        // Dispatch core has a constant CB
        program_consumer_cb<false>(
            local_multicore_cb_cfg,
            dispatch_multicore_cb_cfg,
            dispatch_noc_encoding,
            remote_cb_num_pages,
            DeviceCommand::PROGRAM_PAGE_SIZE,
            remote_cb_size);
    }

    DPRINT << "MY_X: " << (uint32_t) my_x[0] << ", MY_Y: " << (uint32_t) my_y[0] << ENDL();
    while (true) {
        if constexpr (read_from_issue_queue) {
            DPRINT << "WAIT FRONT" << ENDL();
            issue_queue_wait_front();
            DPRINT << "DONE WAITING" << ENDL();
            uint32_t rd_ptr = (cq_read_interface.issue_fifo_rd_ptr << 4);
            uint64_t src_noc_addr = pcie_core_noc_encoding | rd_ptr;
            noc_async_read(src_noc_addr, command_start_addr, min(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND, issue_queue_size - rd_ptr));
            // DPRINT << "BARRIER" << ENDL();
        } else {
            DPRINT << "WAIT FOR DST TO GET CMD" << ENDL();
            db_acquire(pull_semaphore_addr, my_noc_encoding); // dst routers increments this semaphore when cmd is available in the dst router
            DPRINT << "DONE WAIT FOR DST TO GET CMD" << ENDL();
            uint64_t src_noc_addr = pull_noc_encoding | eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
            noc_async_read(src_noc_addr, command_start_addr, DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND); // read in cmd header only
            debug[0] = 109;
        }
        noc_async_read_barrier();

        // Producer information
        volatile tt_l1_ptr uint32_t* command_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(command_start_addr);
        volatile tt_l1_ptr CommandHeader* header = (CommandHeader*)command_ptr;

        uint32_t issue_data_size = header->issue_data_size;
        uint32_t num_buffer_transfers = header->num_buffer_transfers;
        uint32_t stall = header->stall;
        uint32_t page_size = header->page_size;
        uint32_t producer_cb_size = header->producer_cb_size;
        uint32_t consumer_cb_size = header->consumer_cb_size;
        uint32_t producer_cb_num_pages = header->producer_cb_num_pages;
        uint32_t consumer_cb_num_pages = header->consumer_cb_num_pages;
        uint32_t num_pages = header->num_pages;
        uint32_t producer_consumer_transfer_num_pages = header->producer_consumer_transfer_num_pages;
        bool is_program = header->is_program_buffer;
        DeviceCommand::WrapRegion wrap = (DeviceCommand::WrapRegion)header->wrap;

        if constexpr (read_from_issue_queue) { // don't wrap issue queue on completion path
            if (wrap == DeviceCommand::WrapRegion::ISSUE) {
                // Basically popfront without the extra conditional
                cq_read_interface.issue_fifo_rd_ptr = cq_read_interface.issue_fifo_limit - cq_read_interface.issue_fifo_size;  // Head to beginning of command queue
                cq_read_interface.issue_fifo_rd_toggle = not cq_read_interface.issue_fifo_rd_toggle;
                notify_host_of_issue_queue_read_pointer<host_issue_queue_read_ptr_addr>();
                continue; // issue wraps are not relayed forward
            }
        }

        // DPRINT << "PROGRAM LOCAL CB" << ENDL();
        program_local_cb(data_section_addr, producer_cb_num_pages, page_size, producer_cb_size);

        if constexpr (pull_and_push_config != tt::PullAndPushConfig::PULL_FROM_REMOTE) { // completion queue writer for R chip does not relay commands/data
            DPRINT << "WAIT CONSUMER SPACE push sem " << push_semaphore_addr[0] << ENDL();
            wait_consumer_space_available(push_semaphore_addr);
            debug[0] = 110;
        }

        // TODO: update this to work for remote pull and push when enqueuing programs is supported
        if constexpr (pull_and_push_config == tt::PullAndPushConfig::LOCAL) {
            if (stall) {
                DPRINT << "WAIT CONSUMER IDLE" << ENDL();
                debug[0] = 111;
                wait_consumer_idle<2>(push_semaphore_addr);
            }
        }

        uint32_t completion_data_size = header->completion_data_size;
        if constexpr (pull_and_push_config == tt::PullAndPushConfig::PULL_FROM_REMOTE) {
            DPRINT << "Pull and push to the completion queue" << ENDL();
            completion_queue_reserve_back(completion_data_size);
            write_event(uint32_t(&header->event));

            if (wrap == DeviceCommand::WrapRegion::COMPLETION) {
                cq_write_interface.completion_fifo_wr_ptr = completion_queue_start_addr >> 4;     // Head to the beginning of the completion region
                cq_write_interface.completion_fifo_wr_toggle = not cq_write_interface.completion_fifo_wr_toggle;
                notify_host_of_completion_queue_write_pointer<host_completion_queue_write_ptr_addr>();
                noc_async_write_barrier(); // Barrier for now
            }
        }

        if constexpr (pull_and_push_config == tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH or pull_and_push_config == tt::PullAndPushConfig::PULL_FROM_REMOTE) { // signal to dst ethernet router that command was pulled in and data can be sent
            DPRINT << "Signal to dst router that we can consume data" << ENDL();
            noc_semaphore_inc(pull_noc_encoding | eth_get_semaphore(0), 1);
            noc_async_write_barrier(); // Barrier for now
        }

        if constexpr (pull_and_push_config == tt::PullAndPushConfig::PULL_FROM_REMOTE) {
            DPRINT << "num buffer transfers " << num_buffer_transfers << ENDL();
            debug[0] = num_buffer_transfers + 10;
            if (num_buffer_transfers == 1) { // reading data from buffer on device and sending to host
                volatile tt_l1_ptr uint32_t* buffer_transfer_ptr = command_ptr + DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
                uint32_t src_bank_base_address = buffer_transfer_ptr[0];
                uint32_t dst_bank_base_address = buffer_transfer_ptr[1];
                uint32_t num_pages = buffer_transfer_ptr[2];
                uint32_t src_buf_type = buffer_transfer_ptr[4];
                uint32_t dst_buf_type = buffer_transfer_ptr[5];
                uint32_t src_page_index = buffer_transfer_ptr[6];
                uint32_t dst_page_index = buffer_transfer_ptr[7];

                // doing a write so we pull from eth router cb and write to buffer
                src_pr_cfg.cb_buff_cfg.remote_noc_encoding = pull_noc_encoding;
                uint32_t l1_consumer_fifo_limit_16B = (get_cb_start_address<true>() + consumer_cb_size) >> 4;
                src_pr_cfg.cb_buff_cfg.fifo_limit_16B = l1_consumer_fifo_limit_16B;
                // how to program the local cb when there is only one cb data slot
                src_pr_cfg.cb_buff_cfg.remote_rd_addr_16B = (get_cb_start_address<true>() >> 4);
                src_pr_cfg.cb_buff_cfg.remote_total_size_16B = consumer_cb_size >> 4;
                src_pr_cfg.num_pages_to_read = producer_consumer_transfer_num_pages;
                src_pr_cfg.page_size = page_size;

                dst_pr_cfg.buff_cfg.buffer.init((BufferType)dst_buf_type, dst_bank_base_address, page_size);
                dst_pr_cfg.buff_cfg.page_id = dst_page_index;

                dst_pr_cfg.page_size = page_size;
                dst_pr_cfg.num_pages_to_write = producer_consumer_transfer_num_pages;

                DPRINT << "Read from eth cb and write to sysmem buffer - pull sem is " << pull_semaphore_addr[0] << ENDL();

                pull_and_relay<PullAndRelayType::CIRCULAR_BUFFER, PullAndRelayType::BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages); // write all the data
            }
            uint32_t finish = header->finish;
            if (finish) {
                notify_host_complete<host_finish_addr>();
            }
            completion_queue_push_back<completion_queue_start_addr, host_completion_queue_write_ptr_addr>(completion_data_size);
            continue;
        }

        // This should be cleaned up, logic kind of awkward

        if (is_program) {
            relay_command<command_start_addr, consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch, dispatch_noc_encoding);
            update_producer_consumer_sync_semaphores(my_noc_encoding, dispatch_noc_encoding, push_semaphore_addr, get_semaphore(0));
            while (true); // TODO: SUPPORT ME
            pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages);
        } else if (num_buffer_transfers == 1) {
            volatile tt_l1_ptr uint32_t* buffer_transfer_ptr = command_ptr + DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;
            uint32_t src_bank_base_address = buffer_transfer_ptr[0];
            uint32_t dst_bank_base_address = buffer_transfer_ptr[1];
            uint32_t num_pages = buffer_transfer_ptr[2];
            uint32_t src_buf_type = buffer_transfer_ptr[4];
            uint32_t dst_buf_type = buffer_transfer_ptr[5];
            uint32_t src_page_index = buffer_transfer_ptr[6];
            uint32_t dst_page_index = buffer_transfer_ptr[7];

            if ( (pull_and_push_config == tt::PullAndPushConfig::PUSH_TO_REMOTE and (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY) or (pull_and_push_config == tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH and (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY) ) {
                // read from buffer and write to cb on eth core

                volatile db_cb_config_t* remote_multicore_cb_cfg = get_remote_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, db_buf_switch);

                debug[0] = 112;
                DPRINT << "Programming consumer CB" << ENDL();
                program_consumer_cb<true>(
                    local_multicore_cb_cfg,
                    remote_multicore_cb_cfg,
                    push_noc_encoding,
                    consumer_cb_num_pages,
                    page_size,
                    consumer_cb_size
                );

                src_pr_cfg.buff_cfg.buffer.init((BufferType)src_buf_type, src_bank_base_address, page_size);
                src_pr_cfg.buff_cfg.page_id = src_page_index;
                src_pr_cfg.num_pages_to_read = producer_cb_num_pages / 2;

                src_pr_cfg.page_size = page_size;

                dst_pr_cfg.cb_buff_cfg.remote_noc_encoding = push_noc_encoding;
                dst_pr_cfg.cb_buff_cfg.local_multicore_cb_cfg = local_multicore_cb_cfg;
                dst_pr_cfg.cb_buff_cfg.remote_multicore_cb_cfg = remote_multicore_cb_cfg;
                uint32_t l1_consumer_fifo_limit_16B = (get_cb_start_address<true>() + consumer_cb_size) >> 4;
                dst_pr_cfg.cb_buff_cfg.fifo_limit_16B = l1_consumer_fifo_limit_16B;

                dst_pr_cfg.page_size = page_size;
                dst_pr_cfg.num_pages_to_write = producer_consumer_transfer_num_pages;

                DPRINT << "Read from src buffer and write to cb" << ENDL();

                if (pull_and_push_config == tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH) {
                    header->fwd_path = 0;
                }

                relay_command<command_start_addr, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, consumer_data_buffer_size>(db_buf_switch, push_noc_encoding);
                debug[0] = 113;
                update_producer_consumer_sync_semaphores(my_noc_encoding, push_noc_encoding, push_semaphore_addr, eth_get_semaphore(0));
                debug[0] = 114;
                pull_and_relay<PullAndRelayType::BUFFER, PullAndRelayType::CIRCULAR_BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages);
                debug[0] = 115;
                DPRINT << "DONE RELAY TO SRC ETH" << ENDL();

            } else if ( pull_and_push_config == tt::PullAndPushConfig::PUSH_TO_REMOTE and (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY ) {
                DPRINT << "sending read cmd to src router" << ENDL();
                relay_command<command_start_addr, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, consumer_data_buffer_size>(db_buf_switch, push_noc_encoding);
                update_producer_consumer_sync_semaphores(my_noc_encoding, push_noc_encoding, push_semaphore_addr, eth_get_semaphore(0));
                DPRINT << "done sending rd to src eth" << ENDL();
            } else if ( pull_and_push_config == tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH and  (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY ) {

                // remote pull and relay
                // doing a write so we pull from eth router cb and write to buffer
                src_pr_cfg.cb_buff_cfg.remote_noc_encoding = pull_noc_encoding;
                uint32_t l1_consumer_fifo_limit_16B = (get_cb_start_address<true>() + consumer_cb_size) >> 4;
                src_pr_cfg.cb_buff_cfg.fifo_limit_16B = l1_consumer_fifo_limit_16B;
                // how to program the local cb when there is only one cb data slot
                src_pr_cfg.cb_buff_cfg.remote_rd_addr_16B = (get_cb_start_address<true>() >> 4);
                src_pr_cfg.cb_buff_cfg.remote_total_size_16B = consumer_cb_size >> 4;
                src_pr_cfg.num_pages_to_read = producer_consumer_transfer_num_pages;
                src_pr_cfg.page_size = page_size;

                dst_pr_cfg.buff_cfg.buffer.init((BufferType)dst_buf_type, dst_bank_base_address, page_size);
                dst_pr_cfg.buff_cfg.page_id = dst_page_index;

                dst_pr_cfg.page_size = page_size;
                dst_pr_cfg.num_pages_to_write = producer_consumer_transfer_num_pages;

                DPRINT << "Read from eth cb and write to dst buffer - pull sem is " << pull_semaphore_addr[0] << ENDL();

                pull_and_relay<PullAndRelayType::CIRCULAR_BUFFER, PullAndRelayType::BUFFER>(src_pr_cfg, dst_pr_cfg, num_pages); // write all the data

                DPRINT << "DONE WRITE R BUFFER" << ENDL();

                // done doing the write now we have to send the write buffer cmd back to the src router
                header->num_buffer_transfers = 0; // make sure src router doesn't expect any data incoming
                header->fwd_path = 0;
                relay_command<command_start_addr, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, consumer_data_buffer_size>(db_buf_switch, push_noc_encoding);
                update_producer_consumer_sync_semaphores(my_noc_encoding, push_noc_encoding, push_semaphore_addr, eth_get_semaphore(0));
            }
        } else {
            // not a program, and no buffer transfers
            if (pull_and_push_config == tt::PullAndPushConfig::REMOTE_PULL_AND_PUSH) {
                header->fwd_path = 0;
            }
            // send command to src router
            relay_command<command_start_addr, eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE, consumer_data_buffer_size>(db_buf_switch, push_noc_encoding);
            update_producer_consumer_sync_semaphores(my_noc_encoding, push_noc_encoding, push_semaphore_addr, eth_get_semaphore(0));
        }

        // DPRINT << "Done data movement" << ENDL();

        if constexpr (read_from_issue_queue) {
            issue_queue_pop_front<host_issue_queue_read_ptr_addr>(DeviceCommand::NUM_BYTES_IN_DEVICE_COMMAND + issue_data_size);
        }

        if constexpr (pull_and_push_config == tt::PullAndPushConfig::LOCAL) { // update this because remote pull and push needs to swap between cmd slots on the remote dispatcher
            db_buf_switch = not db_buf_switch; // only one cmd slot on ethernet core
        }
    }
}
