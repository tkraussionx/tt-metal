// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/routing/flow_control/flow_control_enums.hpp"
#include "tt_metal/impl/routing/routing_packet_queue_ctrl.hpp"

#include <cstddef>

namespace tt_metal {
namespace routing {
namespace messaging {



template <class sender_impl_t>
struct sender {
    using remote_addr_t = sender_impl_t::remote_addr_t;

    // Contemplating specializing the message_src type or setting it to universal byte* at
    // possible expense of compiler not being able to reason about alignment
    void send_message(uint32_t *message_src, size_t message_size_words, remote_addr_t dest_addr) {
        static_cast<sender_impl_t*>(this)->send_message(message_src, message_size_words, dest_addr);
    }

    bool command_queue_full() const {
        return static_cast<sender_impl_t*>(this)->command_queue_full();
    }

    // TODO: add a sync/barrier function
};

////////////////////////////////////////
// Noc Payload Writer
#include "tt_metal/hw/inc/dataflow_api.h" // for async_write and noc_cmd_buf_ready
struct noc_sender : public sender<host_sender> {
    using remote_addr_t = remote_payload_addr_t<TransportMedium::NOC>::type;

    void send_message(uint32_t *message_src, size_t message_size_words, remote_addr_t dest_addr) {
        noc_async_write(reinterpret_cast<uint32_t>(message_src), dest_addr, message_size_words);
    }

    bool command_queue_full() const {
        return !noc_cmd_buf_ready(noc_index, NCRISC_WR_CMD_BUF);
    }
};
////////////////////////////////////////


////////////////////////////////////////
// Ethernet Payload Writer
#include "tt_metal/hw/inc/ethernet/dataflow_api.h" // for async_write and noc_cmd_buf_ready
struct noc_sender : public sender<host_sender> {
    using remote_addr_t = remote_payload_addr_t<TransportMedium::NOC>::type;

    void send_message(uint32_t *message_src, size_t message_size_words, remote_addr_t dest_addr) {
        eth_send_bytes_over_channel_payload_only(reinterpret_cast<uint32_t>(message_src), dest_addr, message_size_words);
    }

    bool command_queue_full() const {
        return eth_txq_is_busy();
    }
};

////////////////////////////////////////


////////////////////////////////////////
// Host Payload Writer
#include <algorithm> // for host
struct host_sender : public sender<host_sender> {
    using remote_addr_t = remote_payload_addr_t<TransportMedium::SHARED_MEM>::type;

    void send_message(uint32_t *message_src, size_t message_size_words, remote_addr_t dest_addr) {
        std::copy_n(message_src, message_size_words, reinterpret_cast<decltype(message_src)>(dest_addr));
    }

    bool command_queue_full() const {
        return false;
    }
};
////////////////////////////////////////


struct message_writer {

}


} // namespace messaging
} // namespace routing
} // namespace tt_metal
