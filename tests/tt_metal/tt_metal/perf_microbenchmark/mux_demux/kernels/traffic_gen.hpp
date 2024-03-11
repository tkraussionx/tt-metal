// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

inline uint32_t prng_next(uint32_t n) {
    uint32_t x = n;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline void zero_queue_data(uint32_t queue_start_addr_words, uint32_t queue_size_words) {
    tt_l1_ptr uint32_t* queue_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(queue_start_addr_words*PACKET_WORD_SIZE_BYTES);
    for (uint32_t i = 0; i < queue_size_words*PACKET_WORD_SIZE_BYTES/4; i++) {
        queue_ptr[i] = 0;
    }
}

typedef struct {

    uint64_t data_words_input;
    uint32_t packet_index;
    uint32_t packet_rnd_seed;
    uint32_t curr_packet_size_words;
    uint32_t curr_packet_words_remaining;
    uint32_t curr_packet_dest;
    bool last_packet;

    inline void init(uint32_t prng_seed, uint32_t endpoint_id) {
        this->packet_rnd_seed = prng_seed ^ endpoint_id;
        this->curr_packet_words_remaining = 0;
        this->last_packet = false;
        this->packet_index = 0;
    }

    inline void rnd_packet_update(uint32_t num_dest_endpoints,
                                  uint32_t dest_endpoint_start_id,
                                  uint32_t max_packet_size_words,
                                  uint64_t total_data_words) {
        this->curr_packet_dest = ((this->packet_rnd_seed >> 16) & (num_dest_endpoints-1)) + dest_endpoint_start_id;
        this->curr_packet_size_words = this->packet_rnd_seed & (max_packet_size_words-1);
        this->curr_packet_words_remaining = this->curr_packet_size_words + 1;
        this->data_words_input += (this->curr_packet_size_words+1);
        this->last_packet = this->data_words_input >= total_data_words;
        this->packet_index++;
    }

    inline void next_packet_rnd(uint32_t num_dest_endpoints,
                                uint32_t dest_endpoint_start_id,
                                uint32_t max_packet_size_words,
                                uint64_t total_data_words) {
        this->packet_rnd_seed = prng_next(this->packet_rnd_seed);
        this->rnd_packet_update(num_dest_endpoints, dest_endpoint_start_id,
                                max_packet_size_words, total_data_words);
    }

    inline void next_packet_rnd_to_dest(uint32_t num_dest_endpoints,
                                        uint32_t dest_endpoint_id,
                                        uint32_t dest_endpoint_start_id,
                                        uint32_t max_packet_size_words,
                                        uint64_t total_data_words) {
        uint32_t rnd = this->packet_rnd_seed;
        uint32_t dest;
        do {
            rnd = prng_next(rnd);
            dest = (rnd >> 16) & (num_dest_endpoints-1);
        } while (dest != (dest_endpoint_id - dest_endpoint_start_id));
        this->packet_rnd_seed = rnd;
        this->rnd_packet_update(num_dest_endpoints, dest_endpoint_start_id,
                                max_packet_size_words, total_data_words);
    }

    inline bool start_of_packet() {
        return this->curr_packet_words_remaining == this->curr_packet_size_words + 1;
    }

    inline bool packet_active() {
        return this->curr_packet_words_remaining != 0;
    }

    inline bool last_packet_done () {
        return this->last_packet && (this->curr_packet_words_remaining == 0);

    }

    void debug_log_object() {
        debug_log(this->packet_index);
        debug_log(this->packet_rnd_seed);
        debug_log(this->curr_packet_size_words);
        debug_log(this->curr_packet_dest);
        debug_log(this->curr_packet_words_remaining);
        debug_log(this->last_packet);
    }

} input_queue_rnd_state_t;


inline void fill_packet_data(tt_l1_ptr uint32_t* start_addr, uint32_t num_words, uint32_t start_val) {
    tt_l1_ptr uint32_t* addr = start_addr + (PACKET_WORD_SIZE_BYTES/4 - 1);
    for (uint32_t i = 0; i < num_words; i++) {
        *addr = start_val++;
        addr += (PACKET_WORD_SIZE_BYTES/4);
    }
}


inline bool check_packet_data(tt_l1_ptr uint32_t* start_addr, uint32_t num_words, uint32_t start_val,
                              uint32_t& mismatch_addr, uint32_t& mismatch_val, uint32_t& expected_val) {
    tt_l1_ptr uint32_t* addr = start_addr + (PACKET_WORD_SIZE_BYTES/4 - 1);
    for (uint32_t i = 0; i < num_words; i++) {
        if (*addr != start_val) {
            mismatch_addr = reinterpret_cast<uint32_t>(addr);
            mismatch_val = *addr;
            expected_val = start_val;
            return false;
        }
        start_val++;
        addr += (PACKET_WORD_SIZE_BYTES/4);
    }
    return true;
}
