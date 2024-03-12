// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


constexpr uint32_t PACKET_WORD_SIZE_BYTES = 16;
constexpr uint32_t MAX_SWITCH_FAN_IN = 4;
constexpr uint32_t MAX_SWITCH_FAN_OUT = 4;

constexpr uint32_t MAX_SRC_ENDPOINTS = 32;
constexpr uint32_t MAX_DEST_ENDPOINTS = 32;

constexpr uint32_t INPUT_QUEUE_START_ID = 0;
constexpr uint32_t OUTPUT_QUEUE_START_ID = MAX_SWITCH_FAN_IN;

constexpr uint32_t PACKET_QUEUE_REMOTE_READY_FLAG = 0xA;
constexpr uint32_t PACKET_QUEUE_REMOTE_FINISHED_FLAG = 0xB;

constexpr uint32_t PACKET_QUEUE_TEST_STARTED = 0xFFFFFF00;
constexpr uint32_t PACKET_QUEUE_TEST_PASS = 0xFFFFFF01;
constexpr uint32_t PACKET_QUEUE_TEST_TIMEOUT = 0xFFFFFF02;
constexpr uint32_t PACKET_QUEUE_TEST_DATA_MISMATCH = 0xFFFFFF03;

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

inline bool is_remote_network_type_noc(DispatchRemoteNetworkType type) {
    return type == NOC0 || type == NOC1;
}

struct dispatch_packet_header_t {

    uint32_t packet_size_words;
    uint16_t packet_src;
    uint16_t packet_dest;
    uint16_t packet_flags;
    uint16_t num_cmds;
    uint32_t tag;

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
