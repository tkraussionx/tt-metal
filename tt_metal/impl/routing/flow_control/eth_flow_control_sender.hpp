// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/ethernet/dataflow_api.h"  // for async_write and noc_cmd_buf_ready
#include "tt_metal/impl/routing/flow_control/flow_control_sender.hpp"
#include "tt_metal/impl/routing/flow_control/queue_iterator.hpp"

namespace tt_metal {
namespace routing {
namespace flow_control {

// Note that we can't write a non-stream-based
// Start here for EDM migration
template <size_t q_capacity = 0>
struct PacketizedEthernetStreamBasedFlowControlSender
    : public FlowControlSender<PacketizedEthernetStreamBasedFlowControlSender<q_capacity>, q_capacity> {
    PacketizedEthernetStreamBasedFlowControlSender(size_t remote_wrptr_reg_address, RemoteQueuePtrManager<q_capacity> const &q_ptrs) :
        FlowControlSender<PacketizedEthernetStreamBasedFlowControlSender<q_capacity>, q_capacity>(q_ptrs),
        remote_wrptr_reg_address(remote_wrptr_reg_address) {}
    PacketizedEthernetStreamBasedFlowControlSender(size_t remote_wrptr_reg_address, size_t q_size, RemoteQueuePtrManager<q_capacity> const &q_ptrs) :
        FlowControlSender<PacketizedEthernetStreamBasedFlowControlSender<q_capacity>, q_capacity>(q_size, q_ptrs),
        remote_wrptr_reg_address(remote_wrptr_reg_address) {}

    // implements the credit sending mechanics
    void send_credits_impl(size_t wrptr) { eth_write_remote_reg(remote_wrptr_reg_address, wrptr); }

    const size_t remote_wrptr_reg_address;
};

}  // namespace flow_control
}  // namespace routing
}  // namespace tt_metal
