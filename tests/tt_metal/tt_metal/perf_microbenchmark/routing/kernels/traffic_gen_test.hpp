// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include <cstdint>
#include <vector>

inline const char *packet_queue_test_status_to_string(uint32_t status) {
    switch (status) {
    case PACKET_QUEUE_TEST_STARTED:
        return "STARTED";
    case PACKET_QUEUE_TEST_PASS:
        return "DONE/OK";
    case PACKET_QUEUE_TEST_TIMEOUT:
        return "TIMEOUT";
    case PACKET_QUEUE_TEST_DATA_MISMATCH:
        return "DATA_MISMATCH";
    default:
        return "UNKNOWN";
    }
}

inline uint64_t get_64b_result(uint32_t* buf, uint32_t index) {
    return (((uint64_t)buf[index]) << 32) | buf[index+1];
}

inline uint64_t get_64b_result(const std::vector<uint32_t>& vec, uint32_t index) {
    return (((uint64_t)vec[index]) << 32) | vec[index+1];
}

#define TX_TEST_IDX_TOT_DATA_WORDS PQ_TEST_MISC_INDEX + 1
#define TX_TEST_IDX_NPKT PQ_TEST_MISC_INDEX + 3
#define TX_TEST_IDX_WORDS_FLUSHED PQ_TEST_MISC_INDEX + 5
#define TX_TEST_IDX_FEW_DATA_WORDS_SENT_ITER PQ_TEST_MISC_INDEX + 7
#define TX_TEST_IDX_MANY_DATA_WORDS_SENT_ITER PQ_TEST_MISC_INDEX + 9
#define TX_TEST_IDX_ZERO_DATA_WORDS_SENT_ITER PQ_TEST_MISC_INDEX + 11
// #define TX_TEST_IDX_ PQ_TEST_MISC_INDEX +
// #define TX_TEST_IDX_ PQ_TEST_MISC_INDEX +

enum class pkt_dest_size_choices_t {
    RANDOM=0,
    SAME_START_RNDROBIN_FIX_SIZE=1 // max packet size used
};
