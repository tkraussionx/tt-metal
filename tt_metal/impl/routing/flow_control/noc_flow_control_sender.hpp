// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/routing/flow_control/flow_control_sender.hpp"
#include "tt_metal/hw/inc/dataflow_api.h"  // for async_write and noc_cmd_buf_ready

namespace tt_metal {
namespace routing {
namespace flow_control {

template <size_t q_capacity = 0>
struct PacketizedNocFlowControlSender
    : public FlowControlSender<PacketizedNocFlowControlSender<q_capacity>, q_capacity> {
    PacketizedNocFlowControlSender(size_t remote_noc_x, size_t remote_noc_y, size_t remote_wrptr_address, RemoteQueuePtrManager<q_capacity> const &q_ptrs) :
        FlowControlSender<PacketizedNocFlowControlSender, q_capacity>(q_ptrs),
        remote_wrptr_address(get_noc_addr(remote_noc_x, remote_noc_y, remote_wrptr_address)) {}

    PacketizedNocFlowControlSender(
        size_t remote_noc_x, size_t remote_noc_y, size_t remote_wrptr_address, size_t q_size, RemoteQueuePtrManager<q_capacity> const &q_ptrs) :
        FlowControlSender<PacketizedNocFlowControlSender, q_capacity>(q_size, q_ptrs),
        remote_wrptr_address(get_noc_addr(remote_noc_x, remote_noc_y, remote_wrptr_address)) {}

    // implements the credit sending mechanics
    void send_credits(size_t n) { noc_inline_dw_write(remote_wrptr_address, n); }

    uint64_t remote_wrptr_address;
};

}  // namespace flow_control
}  // namespace routing
}  // namespace tt_metal
