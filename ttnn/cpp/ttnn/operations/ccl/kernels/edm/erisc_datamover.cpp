// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "dataflow_api.h"
#include "eth_l1_address_map.h"

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_async_datamover.hpp"

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
//    8) worker_semaphore_id
//    9) sender_num_workers
//       Informs how many worker X/Y coords to accept in the next loop. Each X/Y pair is 2 uint16s
//       10) worker_coord(s)
// ...
// Repeat from step 2 for receiver side

using ttnn::ccl::WorkerXY;

// COMPILE TIME ARGS
// If true, will enable this erisc's sender functionality
constexpr bool enable_sender_side = get_compile_time_arg_val(0) != 0;

// If true, will enable this erisc's receiver functionality
constexpr bool enable_receiver_side = get_compile_time_arg_val(1) != 0;

constexpr uint32_t num_senders = get_compile_time_arg_val(2);
constexpr uint32_t num_receivers = get_compile_time_arg_val(3);

static constexpr ttnn::ccl::EriscDataMoverBufferSharingMode edm_buffer_sharing_mode =
    static_cast<ttnn::ccl::EriscDataMoverBufferSharingMode>(get_compile_time_arg_val(4));

static constexpr ttnn::ccl::EriscDataMoverTerminationMode terminate_on_worker_signal =
    static_cast<ttnn::ccl::EriscDataMoverTerminationMode>(get_compile_time_arg_val(5));

static constexpr bool use_compile_time_designated_handshake_sender = false;//get_compile_time_arg_val(6) != 0;
static constexpr bool is_handshake_sender = get_compile_time_arg_val(7) != 0;

static constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(8);
static constexpr uint32_t chip_id = get_compile_time_arg_val(9);

static_assert(num_buffers_per_channel > 0, "compile time argument [9]: num_buffers_per_channel must be > 0");

using EDM_CONFIG_T = erisc::datamover::
    EriscDatamoverConfig<edm_buffer_sharing_mode, terminate_on_worker_signal, num_buffers_per_channel>;
using ChannelBufferT = erisc::datamover::ChannelBuffer<EDM_CONFIG_T>;

FORCE_INLINE void execute_edm_channel_state(ChannelBufferT &edm_channel,
                                            uint32_t &num_senders_complete,
                                            uint32_t &num_receivers_complete,
                                            uint32_t sender_num_channels,
                                            uint32_t receiver_num_channels,
                                            bool &senders_in_progress,
                                            bool &receivers_in_progress,
                                            uint32_t eth_transaction_ack_word_addr) {
    DeviceZoneScopedN("EXEC");
    switch (edm_channel.get_state()) {
        case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_ETH:
            erisc::datamover::receiver_eth_accept_payload_sequence(
                edm_channel, num_receivers_complete, eth_transaction_ack_word_addr);
            receivers_in_progress =
                receivers_in_progress && num_receivers_complete != receiver_num_channels;
            break;

        case ChannelBufferT::STATE::RECEIVER_WAITING_FOR_WORKER:
            erisc::datamover::receiver_noc_read_worker_completion_check_sequence(
                edm_channel, num_receivers_complete);
            receivers_in_progress =
                receivers_in_progress && num_receivers_complete != receiver_num_channels;
            break;

        //////////////
        case ChannelBufferT::STATE::SENDER_WAITING_FOR_WORKER:
            erisc::datamover::sender_noc_receive_payload_ack_check_sequence(
                edm_channel, num_senders_complete);
            senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
            break;

        case ChannelBufferT::STATE::SENDER_READY_FOR_ETH_TRANSFER:
            erisc::datamover::sender_eth_send_data_sequence(edm_channel);
            break;

        case ChannelBufferT::STATE::SENDER_WAITING_FOR_ETH:  // Can remove due to short circuit
            erisc::datamover::sender_eth_check_receiver_ack_sequence(
                edm_channel, num_senders_complete);
            senders_in_progress = senders_in_progress && num_senders_complete != sender_num_channels;
            break;
            /////////////////////////

        default:
            ASSERT(false);
            break;
    };
}

static constexpr bool SAVE_CODESPACE = true;

template <uint8_t max>
uint8_t wrap_increment(uint8_t &val) {
    static constexpr bool max_is_pow2 = ((max - 1) & max) == 0;
    static constexpr bool max_is_2 = (max == 2);
    if constexpr (max_is_2) {
        return 1 - val;
    } else if constexpr (max_is_pow2) {
        static constexpr bool max_pow2_incr_mask = (max - 1);
        uint8_t next_val = val + 1;
        return next_val & max_pow2_incr_mask;
    } else {
        static constexpr uint8_t last_index = max - 1;
        return val == last_index ? 0 : val + 1;  // goto next
    }
}

void kernel_main() {
    DEBUG_STATUS("STRT");
    std::array<ChannelBufferT, num_senders + num_receivers> buffer_channels;

    // SENDER ARGS
    uint32_t args_offset = 0;
    uint32_t handshake_addr = get_arg_val<uint32_t>(args_offset++);

    uint8_t const sender_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint32_t const sender_num_channels = num_senders;
    uint8_t num_senders_with_no_work = 0;
    for (uint32_t channel = 0; channel < sender_num_channels; channel++) {
        uint32_t const sender_buffer_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const sender_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const sender_channel_size = get_arg_val<uint32_t>(args_offset++);
        // The erisc's local l1 copy of the semaphore workers remotely increment
        uint32_t const sender_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        // worker's semaphore L1 address
        const uint32_t worker_semaphore_address = get_semaphore(get_arg_val<uint32_t>(args_offset++));
        const uint32_t sender_num_workers = get_arg_val<uint32_t>(args_offset++);
        const uint32_t workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += sender_num_workers;
        new (&buffer_channels[sender_channels_start + channel]) ChannelBufferT(
            sender_channels_start + channel,
            sender_buffer_address,
            sender_channel_size,
            worker_semaphore_address,
            sender_num_workers,
            sender_num_messages_to_send,
            (volatile tt_l1_ptr uint32_t *const)sender_semaphores_base_address,
            (const WorkerXY *)workers_xy_list_addr,
            true);
        if constexpr (terminate_on_worker_signal == EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) {
            if (sender_num_messages_to_send == 0) {
                num_senders_with_no_work++;
            }
        }
    }

    // Receiver args
    uint8_t const receiver_channels_start = get_arg_val<uint32_t>(args_offset++);
    uint32_t const receiver_num_channels = num_receivers;
    uint8_t num_receivers_with_no_work = 0;
    for (uint32_t channel = 0; channel < receiver_num_channels; channel++) {
        uint32_t const receiver_buffers_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_num_messages_to_send = get_arg_val<uint32_t>(args_offset++);
        // Each channel buffer is at buffer_base + (channel_id * sender_channel_size)
        // Each channel currently constrained to the same buffer size
        uint32_t const receiver_channel_size = get_arg_val<uint32_t>(args_offset++);
        uint32_t const receiver_semaphores_base_address = get_arg_val<uint32_t>(args_offset++);
        uint32_t const worker_semaphore_address = get_semaphore(get_arg_val<uint32_t>(args_offset++));
        uint32_t const receiver_num_workers = get_arg_val<uint32_t>(args_offset++);
        const uint32_t workers_xy_list_addr = get_arg_addr(args_offset);
        args_offset += receiver_num_workers;
        new (&buffer_channels[receiver_channels_start + channel]) ChannelBufferT(
            receiver_channels_start + channel,
            receiver_buffers_base_address,
            receiver_channel_size,
            worker_semaphore_address,
            receiver_num_workers,
            receiver_num_messages_to_send,
            (volatile tt_l1_ptr uint32_t *const)receiver_semaphores_base_address,
            (const WorkerXY *)workers_xy_list_addr,
            false);

        if constexpr (terminate_on_worker_signal == EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED) {
            if (receiver_num_messages_to_send == 0) {
                num_receivers_with_no_work++;
            }
        }
    }

    // Handshake with other erisc to make sure it's safe to start sending/receiving
    // Chose an arbitrary ordering mechanism to guarantee one of the erisc's will always be "sender" and the other
    // will always be "receiver" (only for handshake purposes)
    DEBUG_STATUS("HSW");
    bool act_as_sender_in_handshake =
        (sender_channels_start < receiver_channels_start || receiver_num_channels == 0) && sender_num_channels > 0;
    erisc::datamover::eth_setup_handshake(handshake_addr, act_as_sender_in_handshake);
    DEBUG_STATUS("HSD");
    uint32_t eth_transaction_ack_word_addr = handshake_addr + 16;

    constexpr uint32_t SWITCH_INTERVAL = 2000000000;

    uint32_t num_senders_complete = !enable_sender_side ? sender_num_channels : num_senders_with_no_work;
    uint32_t num_receivers_complete = !enable_receiver_side ? receiver_num_channels : num_receivers_with_no_work;
    bool senders_in_progress = num_senders_complete != sender_num_channels;
    bool receivers_in_progress = num_receivers_complete != receiver_num_channels;

    {
        uint32_t idle_count = 0;
        static constexpr uint8_t advanceable_oob_idx = num_receivers + num_senders;
        static constexpr uint8_t advanceable_oob_idx_is_pow2 = ((advanceable_oob_idx - 1) & advanceable_oob_idx) == 0;
        static constexpr bool advanceable_oob_incr_mask = (advanceable_oob_idx - 1);
        bool any_channels_advanceable = false;
        std::array<uint32_t, advanceable_oob_idx> advanceable_channels;
        uint8_t advanceable_index = 0;
        for (uint32_t i = 0; i < advanceable_oob_idx; i++) {
            advanceable_channels[i] = advanceable_oob_idx;
        }

        uint8_t i = 0;
        static constexpr ChannelBufferT::STATE receiver_advanceable_without_eth_state = erisc::datamover::get_receiver_state_that_can_advance_without_eth<EDM_CONFIG_T>();
        while (senders_in_progress || receivers_in_progress) {
            {  // No more channels are advanceable so start checking through all of them to see if any are advanceable

                uint8_t index = advanceable_index;
                for (uint8_t ch = 0; ch < advanceable_oob_idx; ch++) {
                    ChannelBufferT &edm_channel = buffer_channels[ch];
                    if (edm_channel.is_done()) {
                        continue;
                    }
                    bool advanceable = channel_can_make_progress(edm_channel);
                    if (advanceable) {
                        switch (edm_channel.get_state()) {
                            case ChannelBufferT::STATE::SENDER_WAITING_FOR_ETH:
                                // Since we don't depend on tx q space being available, we can just complete this state
                                // quickly right now If this ends up causing enough bubbles over eth, we can add them to
                                // a separate list that we check after eth tx q is full or no other channels are
                                // advanceable
                                erisc::datamover::sender_eth_check_receiver_ack_sequence(
                                    edm_channel, num_senders_complete);
                                // Technically only need to do this for message count termination mode
                                senders_in_progress =
                                    senders_in_progress && num_senders_complete != sender_num_channels;
                            break;

                            case receiver_advanceable_without_eth_state:
                                erisc::datamover::receiver_eth_notify_workers_payload_available_sequence(edm_channel);
                            break;

                            default:
                                // if constexpr (!SAVE_CODESPACE && !eth_txq_is_busy()) {
                                if (!eth_txq_is_busy()) {
                                    execute_edm_channel_state(edm_channel,
                                                num_senders_complete,
                                                num_receivers_complete,
                                                sender_num_channels,
                                                receiver_num_channels,
                                                senders_in_progress,
                                                receivers_in_progress,
                                                eth_transaction_ack_word_addr);
                                    index = wrap_increment<advanceable_oob_idx>(index);
                                } else {
                                    any_channels_advanceable = true;
                                    advanceable_channels[index] = ch;
                                    index = wrap_increment<advanceable_oob_idx>(index);
                                }
                            break;
                        };
                    }
                }
            }

            while (any_channels_advanceable) {

                idle_count = 0;
                if (!eth_txq_is_busy()) {
                    uint8_t channel = advanceable_channels[advanceable_index];
                    ChannelBufferT &edm_channel = buffer_channels[channel];

                    execute_edm_channel_state(edm_channel,
                                            num_senders_complete,
                                            num_receivers_complete,
                                            sender_num_channels,
                                            receiver_num_channels,
                                            senders_in_progress,
                                            receivers_in_progress,
                                            eth_transaction_ack_word_addr);
                    advanceable_channels[advanceable_index] = advanceable_oob_idx;
                    advanceable_index = wrap_increment<advanceable_oob_idx>(advanceable_index);
                    any_channels_advanceable = advanceable_channels[advanceable_index] < advanceable_oob_idx;
                } else {
                    // if constexpr (!SAVE_CODESPACE) {
                    //     ChannelBufferT &edm_channel = buffer_channels[i];
                    //     if (erisc::datamover::state_has_no_dependence_on_eth(edm_channel) && channel_can_make_progress(edm_channel)) {
                    //         if (edm_channel.get_state() == ChannelBufferT::STATE::SENDER_WAITING_FOR_ETH) {
                    //             // Since we don't depend on tx q space being available, we can just complete this state
                    //             // quickly right now If this ends up causing enough bubbles over eth, we can add them to
                    //             // a separate list that we check after eth tx q is full or no other channels are
                    //             // advanceable
                    //             erisc::datamover::sender_eth_check_receiver_ack_sequence(
                    //                 edm_channel, num_senders_complete);
                    //             // Technically only need to do this for message count termination mode
                    //             senders_in_progress =
                    //                 senders_in_progress && num_senders_complete != sender_num_channels;
                    //         } else if (edm_channel.get_state() == receiver_advanceable_without_eth_state) {
                    //             erisc::datamover::receiver_eth_notify_workers_payload_available_sequence(edm_channel);
                    //         }
                    //     }
                    //     i = wrap_increment<advanceable_oob_idx>(i);
                    // }

                }
            }

            idle_count++;

            if (idle_count > SWITCH_INTERVAL) {
                idle_count = 0;
                run_routing();
            }
        }
    }

    {
        for (uint32_t s = 0; s < num_senders + num_receivers; s++) {
            auto &channel = buffer_channels[s];
            // We need to explicitly check for channel send done because we may
            // advance sender channel state as soon as we receive an ack. Since we
            // may be the last active channel, and advance to done state just from ack
            // from the receiver ("I got a payload"), then we need to wait for done
            // at the very end here. Otherise if we invoke another erisc op back-to-back,
            // we may mess up transaction state because it's possible for receiver of this
            // op to send the completion done after that one has already started.
            uint32_t wait_count = 0;
            uint32_t wait_max = 5000000;
            for (uint8_t buffer_index = 0; buffer_index < num_buffers_per_channel; buffer_index++) {
                wait_count = 0;
                channel.buffer_index = buffer_index;
                if (!channel.is_sender_side) {
                    if (!channel.eth_is_receiver_channel_send_done()) {
                        channel.eth_receiver_channel_done();
                    }
                }
            }
            for (uint8_t buffer_index = 0; buffer_index < num_buffers_per_channel; buffer_index++) {
                if (channel.is_sender_side) {
                    while (!channel.eth_is_receiver_channel_send_done()) {
                        wait_count++;
                        if (wait_count > wait_max) {
                            DEBUG_STATUS("STK");
                            run_routing();
                            wait_count = 0;
                        }
                    }
                }
            }
        }
    }
    DEBUG_STATUS("DONE");
}
