// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/erisc_async_datamover.hpp"

// Args Schema:
// 1) handshake addr
// 2) sender channels offset (indicates for the erisc channels, where the senders start
//    so sender and receivers don't clash when paired with sender/receiver on the other
//    end of the link.)
// 3) sender num channels (How many erisc channels to use. ALso how many buffers to instantiate)
//    Informs how many times to iterate through the next group of args
//    4) sender_buffer_address
//    5) sender_num_messages_to_send
//    6) sender_channel_size
//    7) sender_semaphores_base_address
//    8) worker_semaphore_address
//    9) sender_num_workers
//       Informs how many worker X/Y coords to accept in the next loop. Each X/Y pair is 2 uint16s
//       10) worker_coord(s)
// ...
// Repeat from step 2 for receiver side

// Intended only for (performance) test use cases
FORCE_INLINE void eth_setup_handshake2(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        DPRINT << "eth_send_bytes\n";
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        DPRINT << "eth_wait_for_receiver_done\n";
        eth_wait_for_receiver_done();
    } else {
        DPRINT << "eth_wait_for_bytes\n";
        eth_wait_for_bytes(16);
        DPRINT << "wait eth_receiver_done\n";
        eth_receiver_channel_done(0);
    }
}

void kernel_main() {
    // COMPILE TIME ARGS
    // If true, will enable this erisc's sender functionality
    constexpr bool enable_sender_side = get_compile_time_arg_val(0) != 0;

    // If true, will enable this erisc's receiver functionality
    constexpr bool enable_receiver_side = get_compile_time_arg_val(1) != 0;

    std::array<erisc::datamover::ChannelBuffer, erisc_info_t::MAX_CONCURRENT_TRANSACTIONS> buffer_channels;

    //
    std::array<uint32_t, erisc_info_t::MAX_CONCURRENT_TRANSACTIONS> printed_receiver_done;
    // for (uint32_t i = 0; i < erisc_info_t::MAX_CONCURRENT_TRANSACTIONS; i++) {
    //     printed_receiver_done[i] = 0;
    // }

    // SENDER ARGS
    uint32_t args_offset = 0;
    uint32_t handshake_addr = get_arg_val<uint32_t>(args_offset++);

    // kernel_profiler::mark_time(80);

    uint8_t const& sender_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint32_t const& sender_num_channels = get_arg_val<uint32_t>(args_offset++);
    uint8_t num_senders_with_no_work = 0;;
    for (uint32_t channel = 0; channel < sender_num_channels; channel++) {
        uint32_t const& sender_buffer_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const& sender_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const& sender_channel_size = get_arg_val<uint32_t>(args_offset++);
        // TODO(snijjar): we can consider computing this instead using a helper in erisc_datamover_api.h
        // The erisc's local l1 copy of the semaphore workers remotely increment
        uint32_t const& sender_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        // worker's semaphore L1 address
        uint32_t const& worker_semaphore_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const& sender_num_workers = get_arg_val<uint32_t>(args_offset++);
        uint32_t const& workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += sender_num_workers;
        new (&buffer_channels[sender_channels_start + channel]) erisc::datamover::ChannelBuffer(
            sender_channels_start + channel,
            sender_buffer_address,
            sender_channel_size,
            worker_semaphore_address,
            sender_num_workers,
            sender_num_messages_to_send,
            // sender_buffer_address,
            (volatile tt_l1_ptr uint32_t *const)sender_semaphores_base_address,
            (const erisc::datamover::WorkerXY *)workers_xy_list_addr,
            true);
        if (sender_num_messages_to_send == 0) {
            num_senders_with_no_work++;
        }
    }

    // Receiver args
    uint8_t const& receiver_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint32_t const& receiver_num_channels = get_arg_val<uint32_t>(args_offset++);
    uint8_t num_receivers_with_no_work = 0;
    for (uint32_t channel = 0; channel < receiver_num_channels; channel++) {
        uint32_t const& receiver_buffers_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const& receiver_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const& receiver_channel_size = get_arg_val<uint32_t>(args_offset++);
        // TODO(snijjar): we can consider computing this instead using a helper in erisc_datamover_api.h
        uint32_t const& receiver_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const& worker_semaphore_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const& receiver_num_workers = get_arg_val<uint32_t>(args_offset++);
        uint32_t const& workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += receiver_num_workers;
        new (&buffer_channels[receiver_channels_start + channel]) erisc::datamover::ChannelBuffer(
            receiver_channels_start + channel,
            receiver_buffers_base_address,
            receiver_channel_size,
            worker_semaphore_address,
            receiver_num_workers,
            receiver_num_messages_to_send,
            // receiver_buffers_base_address,  // remote_eth_buffer_address,
            (volatile tt_l1_ptr uint32_t *const)receiver_semaphores_base_address,
            (const erisc::datamover::WorkerXY *)workers_xy_list_addr,
            false);

        if (receiver_num_messages_to_send == 0) {
            num_receivers_with_no_work++;
        }
    }

    // Handshake with other erisc to make sure it's safe to start sending/receiving
    // Chose an arbitrary ordering mechanism to guarantee one of the erisc's will always be "sender" and the other
    // will always be "receiver" (only for handshake purposes)
    bool act_as_sender_in_handshake =
        (sender_channels_start < receiver_channels_start || receiver_num_channels == 0) && sender_num_channels > 0;
    // DPRINT << "EDM handshaking with other eth. enable_sender_side " << (uint32_t)act_as_sender_in_handshake << "\n";
    erisc::datamover::eth_setup_handshake(handshake_addr, act_as_sender_in_handshake);

    // DPRINT << "EDM args(" << (uint32_t)act_as_sender_in_handshake << "): sender_channels_start "
    //        << (uint32_t)sender_channels_start << "\n";
    // DPRINT << "EDM args(" << (uint32_t)act_as_sender_in_handshake << "): sender_num_channels " << sender_num_channels << "\n";
    // args_offset = 3;
    // for (uint32_t channel = 0; channel < sender_num_channels; channel++) {
    //     uint32_t sender_buffer_address = get_arg_val<uint32_t>(args_offset++);
    //     uint32_t sender_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
    //     // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
    //     // Each channel currently constrained to the same buffer size
    //     std::uint32_t sender_channel_size = get_arg_val<uint32_t>(args_offset++);
    //     // TODO(snijjar): we can consider computing this instead using a helper in erisc_datamover_api.h
    //     // The erisc's local l1 copy of the semaphore workers remotely increment
    //     uint32_t sender_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
    //     // worker's semaphore L1 address
    //     uint32_t worker_semaphore_address = get_arg_val<uint32_t>(args_offset++);
    //     uint32_t sender_num_workers = get_arg_val<uint32_t>(args_offset++);
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tsender_num_messages_to_send=" << sender_num_messages_to_send << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tsender_channel_size=" << sender_channel_size << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tsender_semaphores_base_address=" << sender_semaphores_base_address << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tworker_semaphore_address=" << worker_semaphore_address << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tsender_num_workers=" << sender_num_workers << "\n";
    //     args_offset += sender_num_workers;
    // }

    // DPRINT << "EDM args(" << (uint32_t)act_as_sender_in_handshake << "): receiver_channels_start "<< (uint32_t)receiver_channels_start << "\n";
    // DPRINT << "EDM args(" << (uint32_t)act_as_sender_in_handshake << "): receiver_num_channels " << receiver_num_channels
    //        << "\n";
    // args_offset += 2;
    // for (uint32_t channel = 0; channel < receiver_num_channels; channel++) {
    //     uint32_t receiver_buffers_base_address = get_arg_val<uint32_t>(args_offset++);
    //     uint32_t receiver_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
    //     // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
    //     // Each channel currently constrained to the same buffer size
    //     uint32_t receiver_channel_size = get_arg_val<uint32_t>(args_offset++);
    //     // TODO(snijjar): we can consider computing this instead using a helper in erisc_datamover_api.h
    //     uint32_t receiver_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
    //     uint32_t worker_semaphore_address = get_arg_val<uint32_t>(args_offset++);
    //     uint32_t receiver_num_workers = get_arg_val<uint32_t>(args_offset++);
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \treceiver_num_messages_to_send=" << receiver_num_messages_to_send << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \treceiver_channel_size=" << receiver_channel_size << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \treceiver_semaphores_base_address=" << receiver_semaphores_base_address << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tworker_semaphore_address=" << worker_semaphore_address << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \treceiver_num_workers=" << receiver_num_workers << "\n";
    //     args_offset += receiver_num_workers;
    // }


    constexpr uint32_t SWITCH_INTERVAL = 100000;
    uint32_t did_nothing_count = 0;

    uint32_t num_senders_complete = !enable_sender_side ? sender_num_channels : num_senders_with_no_work;
    uint32_t num_receivers_complete = !enable_receiver_side ? receiver_num_channels : num_receivers_with_no_work;
    uint32_t curr_sender = sender_channels_start;
    uint32_t curr_receiver = receiver_channels_start;
    uint32_t receiver_channel_end = receiver_channels_start + receiver_num_channels;
    uint32_t sender_channel_end = sender_channels_start + sender_num_channels;
    // {
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tNOC_INDEX " << (uint32_t)noc_index << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tmy_x,my_y " << (uint32_t)((my_x[0] << 16) | my_y[0]) << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tnum_senders_complete " << num_senders_complete << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tsender_num_channels " << sender_num_channels << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \tnum_receivers_complete " << num_receivers_complete << "\n";
    //     DPRINT << "EDM arg (" << (uint32_t)act_as_sender_in_handshake << "): \treceiver_num_channels " << receiver_num_channels << "\n";
    // }
    while ((enable_sender_side && num_senders_complete != sender_num_channels) ||
           (enable_receiver_side && num_receivers_complete != receiver_num_channels)) {
        // DPRINT << "EDM arg (" << (uint32_t)enable_sender_side << "): \tnum_receivers_complete "
        //        << num_receivers_complete << "\n";
        // DPRINT << "EDM arg (" << (uint32_t)enable_sender_side << "):\tnum_senders_complete " << num_senders_complete
        //        << "\n";
        bool did_something = false;

        // Note, may want to interleave the sender and receiver calls to reduce latency for each
        //////////////////////////////////////
        // SENDER
        //////////////////////////////////////
        if constexpr (enable_sender_side) {
            erisc::datamover::ChannelBuffer &current_sender = buffer_channels[curr_sender];
            if (!current_sender.is_done()) {
                did_something =
                    erisc::datamover::sender_noc_receive_payload_ack_check_sequence(current_sender) || did_something;

                did_something = erisc::datamover::sender_eth_send_data_sequence(current_sender) || did_something;

                did_something = erisc::datamover::sender_notify_workers_if_buffer_available_sequence(
                                    current_sender, num_senders_complete) ||
                                did_something;

                did_something =
                    erisc::datamover::sender_eth_check_receiver_ack_sequence(current_sender) || did_something;
            }

            curr_sender += 1;
            curr_sender = (curr_sender >= sender_channel_end) ? sender_channels_start : curr_sender;
        }
        //////////////////////////////////////
        // RECEIVER
        //////////////////////////////////////

        if constexpr (enable_receiver_side) {
            erisc::datamover::ChannelBuffer &current_receiver = buffer_channels[curr_receiver];
            if (!current_receiver.is_done()) {
                bool received = erisc::datamover::receiver_eth_accept_payload_sequence(current_receiver);
                did_something = received || did_something;

                did_something =
                    erisc::datamover::receiver_eth_notify_workers_payload_available_sequence(current_receiver) ||
                    did_something;

                did_something = erisc::datamover::receiver_noc_read_worker_completion_check_sequence(
                                    current_receiver, num_receivers_complete) ||
                                did_something;

            }

            curr_receiver += 1;
            curr_receiver = (curr_receiver >= receiver_channel_end) ? receiver_channels_start : curr_receiver;
        }
        //////////////////////////////////////

        if (did_something) {
            did_nothing_count = 0;
        } else {
            if (did_nothing_count++ > SWITCH_INTERVAL) {
                did_nothing_count = 0;
                // kernel_profiler::mark_time(15);
                run_routing();
            }
        }
    }

    // kernel_profiler::mark_time(16);
    // if (enable_sender_side) {
    //     DPRINT << "1 ERISC DATAMOVER DONE!\n";
    // } else {
    //     DPRINT << "0 ERISC DATAMOVER DONE!\n";
    // }
}
