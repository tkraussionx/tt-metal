// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


constexpr uint32_t PACKET_WORD_SIZE_BYTES = 16;
constexpr uint32_t MAX_SWITCH_FAN_IN = 4;
constexpr uint32_t MAX_SWITCH_FAN_OUT = 4;

enum DispatchPacketFlag : uint32_t {
    PACKET_CMD_START = (0x1 << 1),
    PACKET_CMD_END = (0x1 << 2),
    PACKET_MULTI_CMD = (0x1 << 3),
};

enum DispatchRemoteNetworkType : uint32_t {
    NOC0 = 0,
    NOC1 = 1,
    ETH = 2,
    NONE = 3
};

struct dispatch_packet_header_t {

    uint32_t packet_size_words;
    uint32_t packet_dest;
    uint32_t packet_flags;
    uint32_t num_completed_cmds;

    inline bool check_packet_flags(uint32_t flags) const {
        return (packet_flags & flags) == flags;
    }

    inline void set_packet_flags(uint32_t flags) {
        packet_flags |= flags;
    }

    inline void clear_packet_flags(uint32_t flags) {
        packet_flags &= ~flags;
    }
};
