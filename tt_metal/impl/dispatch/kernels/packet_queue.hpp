// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "cq_cmds.hpp"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"


constexpr uint32_t NUM_WR_CMD_BUFS = 4;

constexpr uint32_t DEFAULT_MAX_NOC_SEND_WORDS = (NUM_WR_CMD_BUFS-1)*(NOC_MAX_BURST_WORDS*NOC_WORD_BYTES)/PACKET_WORD_SIZE_BYTES;
constexpr uint32_t DEFAULT_MAX_ETH_SEND_WORDS = 2*1024;

constexpr uint32_t NUM_PTR_REGS_PER_QUEUE = 3;

extern tt_l1_ptr uint32_t* debug_buf;
extern uint32_t debug_buf_index;
extern uint32_t debug_buf_size;

inline void debug_set_buf(tt_l1_ptr uint32_t* buf, uint32_t size) {
    debug_buf = buf;
    debug_buf_index = 0;
    debug_buf_size = size;
    for (uint32_t i = 0; i < debug_buf_size; i++) {
        debug_buf[i] = 0;
    }
}

inline void debug_log(uint32_t val) {
    debug_buf[debug_buf_index++] = val;
    if (debug_buf_index >= debug_buf_size) {
        debug_buf_index = 0;
    }
}

inline void debug_advance_index(uint32_t n) {
    debug_buf_index += n;
    if (debug_buf_index >= debug_buf_size) {
        debug_buf_index =- debug_buf_size;
    }
}

inline void debug_set_index(uint32_t n) {
    debug_buf_index = n;
}

inline void debug_log_index(uint32_t index, uint32_t val) {
    debug_buf[index] = val;
}


typedef struct dispatch_packet_header_t dispatch_packet_header_t;

static_assert(sizeof(dispatch_packet_header_t) == PACKET_WORD_SIZE_BYTES);


class packet_queue_state_t;
class packet_input_queue_state_t;
class packet_output_queue_state_t;

class packet_queue_state_t {

public:

    uint32_t queue_id;
    uint32_t queue_start_addr_words;
    uint32_t queue_size_words;
    uint32_t ptr_offset_mask;
    uint32_t queue_size_mask;

    bool queue_is_input;

    uint32_t remote_x, remote_y; // remote source for input queues, remote dest for output queues
    uint32_t remote_queue_id;
    DispatchRemoteNetworkType remote_update_network_type;

    // For read/write pointers, we use stream credit registers with auto-increment.
    // Pointers are in 16B units, and we assume buffer size is power of 2 so we get
    // automatic wrapping. (If needed, we can fix the pointer advance functions later
    // to handle non-power-of-2 buffer sizes.)

    inline uint32_t get_queue_local_wptr() const {
        return NOC_STREAM_READ_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    }

    inline void reset_queue_local_wptr() {
        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        NOC_STREAM_WRITE_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id,
            STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);
    }

    inline void advance_queue_local_wptr(uint32_t num_words) {
        NOC_STREAM_WRITE_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            num_words << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }

    inline uint32_t get_queue_local_rptr_sent() const {
        return NOC_STREAM_READ_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id+1,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    }

    inline uint32_t get_queue_local_rptr_cleared() const {
        return NOC_STREAM_READ_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id+2,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    }

    inline void reset_queue_local_rptr_sent()  {
        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        NOC_STREAM_WRITE_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id+1,
            STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);
    }

    inline void reset_queue_local_rptr_cleared()  {
        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        NOC_STREAM_WRITE_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id+2,
            STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);
    }

    inline void advance_queue_local_rptr_sent(uint32_t num_words)  {
        NOC_STREAM_WRITE_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id+1,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            num_words << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }

    inline void advance_queue_local_rptr_cleared(uint32_t num_words)  {
        NOC_STREAM_WRITE_REG(
            NUM_PTR_REGS_PER_QUEUE*this->queue_id+2,
            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
            num_words << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }

    inline uint32_t get_queue_data_num_words_available_to_send() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_sent()) & this->queue_size_mask;
    }

    inline uint32_t get_queue_data_num_words_occupied() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_cleared()) & this->queue_size_mask;
    }

    inline uint32_t get_queue_data_num_words_free() const {
        return this->queue_size_words - this->get_queue_data_num_words_occupied();
    }

    inline uint32_t get_num_words_sent_not_cleared() const {
        return (this->get_queue_local_rptr_sent() - this->get_queue_local_rptr_cleared()) & this->queue_size_mask;
    }

    inline uint32_t get_num_words_written_not_sent() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_sent()) & this->queue_size_mask;
    }

    inline uint32_t get_queue_wptr_offset_words() const {
        return this->get_queue_local_wptr() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_sent_offset_words() const {
        return this->get_queue_local_rptr_sent() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_cleared_offset_words() const {
        return this->get_queue_local_rptr_cleared() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_sent_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_rptr_cleared_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_rptr_cleared_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_wptr_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_words_before_rptr_sent_wrap() const {
        return queue_size_words - this->get_queue_rptr_sent_offset_words();
    }

    inline uint32_t get_queue_words_before_rptr_cleared_wrap() const {
        return queue_size_words - this->get_queue_rptr_cleared_offset_words();
    }

    inline uint32_t get_queue_words_before_wptr_wrap() const {
        return queue_size_words - this->get_queue_wptr_offset_words();
    }

    inline void remote_reg_update(uint32_t reg_addr, uint32_t val, uint32_t cmd_buf = 0) {

        if (this->remote_update_network_type == DispatchRemoteNetworkType::NONE) {
            return;
        }
        else if (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) {
            eth_write_remote_reg(reg_addr, val);
        } else {
            uint64_t dest_addr = NOC_XY_ADDR(this->remote_x, this->remote_y, reg_addr);
            noc_fast_posted_write_dw_inline(
                this->remote_update_network_type,
                cmd_buf,
                val,
                dest_addr,
                0xF, // byte-enable
                NOC_UNICAST_WRITE_VC,
                false // mcast
            );
        }
    }

    inline void advance_queue_remote_wptr(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*this->remote_queue_id,
                                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr_sent(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*this->remote_queue_id+1,
                                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr_cleared(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*this->remote_queue_id+2,
                                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_reg_update(reg_addr, val);
    }

    inline void reset_ready_flag() {
        NOC_STREAM_WRITE_REG(this->queue_id, STREAM_REMOTE_SRC_REG_INDEX, 0);
    }

    inline void send_remote_ready_notification() {
        uint32_t reg_addr = STREAM_REG_ADDR(this->remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
        this->remote_reg_update(reg_addr, PACKET_QUEUE_REMOTE_READY_FLAG);
    }

    inline void send_remote_finished_notification() {
        uint32_t reg_addr = STREAM_REG_ADDR(this->remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
        this->remote_reg_update(reg_addr, PACKET_QUEUE_REMOTE_FINISHED_FLAG);
    }

    inline bool is_remote_ready() const {
        return NOC_STREAM_READ_REG(this->queue_id, STREAM_REMOTE_SRC_REG_INDEX) ==
                PACKET_QUEUE_REMOTE_READY_FLAG;
    }

    inline uint32_t get_remote_ready_status() const {
        return NOC_STREAM_READ_REG(this->queue_id, STREAM_REMOTE_SRC_REG_INDEX);
    }

    inline bool is_remote_finished() const {
        return NOC_STREAM_READ_REG(this->queue_id, STREAM_REMOTE_SRC_REG_INDEX) ==
                PACKET_QUEUE_REMOTE_FINISHED_FLAG;
    }

    void yield() {
        // TODO: implement yield for ethernet here
    }

    void debug_log_object() {
        debug_log(this->queue_id);
        debug_log(this->queue_start_addr_words);
        debug_log(this->queue_size_words);
        debug_log(this->remote_x);
        debug_log(this->remote_y);
        debug_log(this->remote_queue_id);
        debug_log(this->get_queue_local_wptr());
        debug_log(this->get_queue_local_rptr_sent());
        debug_log(this->get_queue_local_rptr_cleared());
        debug_log(static_cast<uint32_t>(this->remote_update_network_type));
    }
};


class packet_input_queue_state_t : public packet_queue_state_t {

protected:

    bool curr_packet_valid;
    tt_l1_ptr dispatch_packet_header_t* curr_packet_header_ptr;
    uint32_t curr_packet_src;
    uint32_t curr_packet_dest;
    uint32_t curr_packet_size_words;
    uint32_t curr_packet_words_sent;
    uint32_t curr_packet_tag;
    uint32_t curr_packet_flags;

    inline void advance_next_packet() {
        if(this->get_queue_data_num_words_available_to_send() > 0) {
            tt_l1_ptr dispatch_packet_header_t* next_packet_header_ptr =
                reinterpret_cast<tt_l1_ptr dispatch_packet_header_t*>(
                    (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES
                );
            this->curr_packet_header_ptr = next_packet_header_ptr;
            this->curr_packet_dest = next_packet_header_ptr->packet_dest;
            this->curr_packet_src = next_packet_header_ptr->packet_src;
            this->curr_packet_size_words = next_packet_header_ptr->packet_size_words;
            this->curr_packet_tag = next_packet_header_ptr->tag;
            this->curr_packet_flags = next_packet_header_ptr->packet_flags;
            this->curr_packet_words_sent = 0;
            this->curr_packet_valid = true;
       }
    }

public:

    inline void init(uint32_t queue_id,
                     uint32_t queue_start_addr_words,
                     uint32_t queue_size_words,
                     uint32_t remote_x,
                     uint32_t remote_y,
                     uint32_t remote_queue_id,
                     DispatchRemoteNetworkType remote_update_network_type) {

        this->queue_id = queue_id;
        this->queue_start_addr_words = queue_start_addr_words;
        this->queue_size_words = queue_size_words;
        tt_l1_ptr uint32_t* queue_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(queue_start_addr_words*PACKET_WORD_SIZE_BYTES);
        for (uint32_t i = 0; i < queue_size_words*PACKET_WORD_SIZE_BYTES/4; i++) {
            queue_ptr[i] = 0;
        }
        this->ptr_offset_mask = queue_size_words - 1;
        this->queue_size_mask = (queue_size_words << 1) - 1;
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_queue_id = remote_queue_id;
        this->remote_update_network_type = remote_update_network_type;
        this->curr_packet_valid = false;
        this->reset_queue_local_rptr_sent();
        this->reset_queue_local_rptr_cleared();
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }

    inline tt_l1_ptr dispatch_packet_header_t* get_curr_packet_header_ptr() {
        return this->curr_packet_header_ptr;
    }

    inline uint32_t get_curr_packet_dest() const {
        return this->curr_packet_dest;
    }

    inline uint32_t get_curr_packet_src() const {
        return this->curr_packet_src;
    }

    inline uint32_t get_curr_packet_size_words() const {
        return this->curr_packet_size_words;
    }

    inline uint32_t get_curr_packet_words_remaining() const {
        return this->curr_packet_size_words - this->curr_packet_words_sent;
    }

    inline uint32_t get_curr_packet_tag() const {
        return this->curr_packet_tag;
    }

    inline uint32_t get_curr_packet_flags() const {
        return this->curr_packet_flags;
    }

    inline bool partial_packet_sent() const {
        return this->curr_packet_valid && (this->curr_packet_words_sent > 0);
    }

    inline bool wait_for_src_ready(uint64_t timeout_cycles = 0) {
        uint64_t start_timestamp = 0;
        if (timeout_cycles > 0) {
             start_timestamp = c_tensix_core::read_wall_clock();
        }
        while (!this->is_remote_ready()) {
            if (timeout_cycles > 0) {
                uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - start_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    return false;
                }
            }
            this->send_remote_ready_notification();
            this->yield();
        }
        return true;
    }

    inline bool input_queue_full_packet_available_to_send(uint32_t& num_words_available_to_send) {
        num_words_available_to_send = this->get_queue_data_num_words_available_to_send();
        if (num_words_available_to_send == 0) {
            return false;
        }
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        uint32_t curr_packet_words_remaining = this->get_curr_packet_words_remaining();
        return num_words_available_to_send >= curr_packet_words_remaining;
    }

    inline uint32_t input_queue_curr_packet_num_words_available_to_send() {
        uint32_t num_words = this->get_queue_data_num_words_available_to_send();
        if (num_words == 0) {
            return 0;
        }
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        uint32_t curr_packet_words_remaining = this->get_curr_packet_words_remaining();
        num_words = std::min(num_words, curr_packet_words_remaining);
        return num_words;
    }

    inline void input_queue_advance_words_sent(uint32_t num_words) {
        if (num_words > 0) {
            this->advance_queue_local_rptr_sent(num_words);
            this->advance_queue_remote_rptr_sent(num_words);
            this->curr_packet_words_sent += num_words;
            uint32_t curr_packet_words_remaining = this->get_curr_packet_words_remaining();
            if (curr_packet_words_remaining == 0) {
                this->curr_packet_valid = false;
                this->advance_next_packet();
            }
        }
    }

    inline void input_queue_advance_words_cleared(uint32_t num_words) {
        if (num_words > 0) {
            this->advance_queue_local_rptr_cleared(num_words);
            this->advance_queue_remote_rptr_cleared(num_words);
        }
    }

    inline void input_queue_clear_all_words_sent() {
        this->input_queue_advance_words_cleared(this->get_num_words_sent_not_cleared());
    }

    void debug_log_object() {
        debug_log(0xabcdef00);
        packet_queue_state_t::debug_log_object();
        debug_log(0xFFFFFFFF);
        debug_log(this->curr_packet_valid);
        debug_log(this->curr_packet_tag);
        debug_log(this->curr_packet_dest);
        debug_log(this->curr_packet_size_words);
        debug_log(this->curr_packet_words_sent);
    }

};


class packet_output_queue_state_t : public packet_queue_state_t {

    uint32_t max_noc_send_words;
    uint32_t max_eth_send_words;

    struct {

        packet_input_queue_state_t* input_queue_array;
        uint32_t input_queue_words_in_flight[2][MAX_SWITCH_FAN_IN];
        uint32_t total_words_in_flight[2];
        uint32_t curr_index;

        void init(packet_input_queue_state_t* input_queue_array) {
            this->input_queue_array = input_queue_array;
            for (uint32_t i = 0; i < MAX_SWITCH_FAN_IN; i++) {
                this->input_queue_words_in_flight[0][i] = 0;
                this->input_queue_words_in_flight[1][i] = 0;
            }
            this->curr_index = 0;
            this->total_words_in_flight[0] = 0;
            this->total_words_in_flight[1] = 0;
        }

        uint32_t get_curr_total_words_in_flight() const {
            return this->total_words_in_flight[this->curr_index];
        }

        uint32_t get_prev_total_words_in_flight() const {
            return this->total_words_in_flight[this->curr_index ^ 1];
        }

        uint32_t prev_words_in_flight_flush() {
            uint32_t prev_index = this->curr_index ^ 1;
            uint32_t total_words_in_flight = 0;
            if (this->total_words_in_flight[prev_index] > 0) {
                for (uint32_t i = 0; i < MAX_SWITCH_FAN_IN; i++) {
                    uint32_t words_in_flight = this->input_queue_words_in_flight[prev_index][i];
                    total_words_in_flight += words_in_flight;
                    if (words_in_flight > 0) {
                        this->input_queue_array[i].input_queue_advance_words_cleared(words_in_flight);
                        this->input_queue_words_in_flight[prev_index][i] = 0;
                    }
                }
                this->total_words_in_flight[prev_index] = 0;
            }
            this->curr_index = prev_index;
            return total_words_in_flight;
        }

        void register_words_in_flight(uint32_t input_queue_id, uint32_t num_words) {
            this->input_queue_words_in_flight[this->curr_index][input_queue_id] += num_words;
            this->total_words_in_flight[this->curr_index] += num_words;
            this->input_queue_array[input_queue_id].input_queue_advance_words_sent(num_words);
        }

        void debug_log_object() {
            debug_log(this->curr_index);
            for (uint32_t i = 0; i < 2; i++) {
                debug_log(0xaa000000 + i);
                debug_log(this->total_words_in_flight[i]);
                for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                    debug_log(0xbb000000 + j);
                    debug_log(this->input_queue_words_in_flight[i][j]);
                }
            }
        }

    } input_queue_status;

public:

    inline void init(uint32_t queue_id,
                     uint32_t queue_start_addr_words,
                     uint32_t queue_size_words,
                     uint32_t remote_x,
                     uint32_t remote_y,
                     uint32_t remote_queue_id,
                     DispatchRemoteNetworkType remote_update_network_type,
                     packet_input_queue_state_t* input_queue_array) {

        this->queue_id = queue_id;
        this->queue_start_addr_words = queue_start_addr_words;
        this->queue_size_words = queue_size_words;
        this->ptr_offset_mask = queue_size_words - 1;
        this->queue_size_mask = (queue_size_words << 1) - 1;
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_queue_id = remote_queue_id;
        this->remote_update_network_type = remote_update_network_type;
        this->max_noc_send_words = DEFAULT_MAX_NOC_SEND_WORDS;
        this->max_eth_send_words = DEFAULT_MAX_ETH_SEND_WORDS;
        this->input_queue_status.init(input_queue_array);
        this->reset_queue_local_rptr_sent();
        this->reset_queue_local_rptr_cleared();
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }

    inline void set_max_noc_send_words(uint32_t max_noc_send_words) {
        this->max_noc_send_words = max_noc_send_words;
    }

    inline void set_max_eth_send_words(uint32_t max_eth_send_words) {
        this->max_eth_send_words = max_eth_send_words;
    }

    inline bool wait_for_dest_ready(uint64_t timeout_cycles = 0) {
        uint64_t start_timestamp = 0;
        if (timeout_cycles > 0) {
            start_timestamp = c_tensix_core::read_wall_clock();
        }
        while (!this->is_remote_ready()) {
            if (timeout_cycles > 0) {
                uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - start_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    return false;
                }
            }
            this->yield();
        }
        this->send_remote_ready_notification();
        return true;
    }

    inline uint32_t output_max_num_words_to_forward() const {
        return (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) ?
            this->max_eth_send_words : this->max_noc_send_words;
    }

    inline void send_data_to_remote(uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {
        if ((this->remote_update_network_type == DispatchRemoteNetworkType::NONE) ||
            (num_words == 0)) {
            return;
        } else if (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) {
            internal_::eth_send_packet(0, src_addr/PACKET_WORD_SIZE_BYTES, dest_addr/PACKET_WORD_SIZE_BYTES, num_words);
        } else {
            uint64_t noc_dest_addr = NOC_XY_ADDR(this->remote_x, this->remote_y, dest_addr);
            uint32_t cmd_buf = 0;
            uint32_t words_remaining = num_words;
            while (words_remaining > 0) {
                uint32_t words_to_send = std::min(words_remaining, ((uint32_t)NOC_MAX_BURST_WORDS));
                uint32_t bytes_to_send = words_to_send*PACKET_WORD_SIZE_BYTES;
                while (!noc_cmd_buf_ready(this->remote_update_network_type, cmd_buf));
                ncrisc_noc_fast_write_posted(
                    this->remote_update_network_type,
                    cmd_buf,
                    src_addr,
                    noc_dest_addr,
                    bytes_to_send,
                    NOC_UNICAST_WRITE_VC,
                    false, // mcast
                    false, // linked
                    1 // num_dests
                );
                src_addr += bytes_to_send;
                dest_addr += bytes_to_send;
                words_remaining -= words_to_send;
                cmd_buf = (cmd_buf + 1) % NUM_WR_CMD_BUFS;
            }
        }
    }

    inline void remote_wptr_update(uint32_t num_words) {
        this->advance_queue_remote_wptr(num_words);
    }

    inline uint32_t prev_words_in_flight_check_flush() {
        if (this->get_num_words_written_not_sent() <= this->input_queue_status.get_curr_total_words_in_flight()) {
            return this->input_queue_status.prev_words_in_flight_flush();
        } else {
            return 0;
        }
    }

    inline bool output_barrier(uint64_t timeout_cycles = 0) {
        uint64_t start_timestamp = 0;
        if (timeout_cycles > 0) {
            start_timestamp = c_tensix_core::read_wall_clock();
        }
        while (this->get_queue_data_num_words_occupied() > 0) {
            if (timeout_cycles > 0) {
                uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - start_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    return false;
                }
            }
            this->yield();
        }
        this->input_queue_status.prev_words_in_flight_flush();
        this->input_queue_status.prev_words_in_flight_flush();
        return true;
    }

    inline uint32_t get_num_words_to_send(uint32_t input_queue_index) {

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);

        uint32_t num_words_available_in_input = input_queue_ptr->input_queue_curr_packet_num_words_available_to_send();
        uint32_t num_words_before_input_rptr_wrap = input_queue_ptr->get_queue_words_before_rptr_sent_wrap();
        num_words_available_in_input = std::min(num_words_available_in_input, num_words_before_input_rptr_wrap);
        uint32_t num_words_free_in_output = this->get_queue_data_num_words_free();
        uint32_t num_words_to_forward = std::min(num_words_available_in_input, num_words_free_in_output);

        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t output_buf_words_before_wptr_wrap = this->get_queue_words_before_wptr_wrap();
        num_words_to_forward = std::min(num_words_to_forward, output_buf_words_before_wptr_wrap);
        num_words_to_forward = std::min(num_words_to_forward, this->output_max_num_words_to_forward());

        return num_words_to_forward;
    }


    // blocks until all words are forwarded, or returns false in case of timeout
    inline bool forward_packet_from_input(uint32_t input_queue_index, uint64_t timeout_cycles = 0) {

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);
        uint32_t packet_words_remaining = input_queue_ptr->get_curr_packet_words_remaining();

        uint32_t words_forwarded = 0;
        uint64_t start_timestamp = 0;
        if (timeout_cycles > 0) {
            start_timestamp = c_tensix_core::read_wall_clock();
        }
        while (words_forwarded != packet_words_remaining) {
            if (timeout_cycles > 0) {
                uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - start_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    return false;
                }
            }
            uint32_t num_words_to_forward = this->get_num_words_to_send(input_queue_index);
            if (num_words_to_forward > 0) {
                uint32_t src_addr =
                    (input_queue_ptr->queue_start_addr_words +
                    input_queue_ptr->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES;
                uint32_t dest_addr =
                    (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;
                this->send_data_to_remote(src_addr, dest_addr, num_words_to_forward);
                this->input_queue_status.register_words_in_flight(input_queue_index, num_words_to_forward);
                this->advance_queue_local_wptr(num_words_to_forward);
                this->remote_wptr_update(num_words_to_forward);
                words_forwarded += num_words_to_forward;
            }
        }
        return true;
    }

    inline uint32_t forward_data_from_input(uint32_t input_queue_index, uint32_t max_num_words = 0) {

        uint32_t num_words_to_forward = this->get_num_words_to_send(input_queue_index);
        if (max_num_words > 0) {
            num_words_to_forward = std::min(num_words_to_forward, max_num_words);
        }
        if (num_words_to_forward == 0) {
            return 0;
        }

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);

        uint32_t src_addr =
            (input_queue_ptr->queue_start_addr_words +
             input_queue_ptr->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES;
        uint32_t dest_addr =
            (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;

        this->send_data_to_remote(src_addr, dest_addr, num_words_to_forward);
        this->input_queue_status.register_words_in_flight(input_queue_index, num_words_to_forward);
        this->advance_queue_local_wptr(num_words_to_forward);
        this->remote_wptr_update(num_words_to_forward);

        return num_words_to_forward;
    }

    void debug_log_object() {
        debug_log(0xabcdef01);
        packet_queue_state_t::debug_log_object();
        debug_log(0xeeeeeeee);
        debug_log(this->output_max_num_words_to_forward());
        debug_log(0xffffffff);
        this->input_queue_status.debug_log_object();
    }
};


/**
 *  Polling for ready signal from the remote peers of all input and output queues.
 *  Blocks until all are ready, but doesn't block polling on each individual queue.
 *  Returns false in case of timeout.
 */
bool wait_all_src_dest_ready(packet_input_queue_state_t* input_queue_array, uint32_t num_input_queues,
                             packet_output_queue_state_t* output_queue_array, uint32_t num_output_queues,
                             uint64_t timeout_cycles = 0) {

    bool all_src_dest_ready = false;
    bool src_ready[MAX_SWITCH_FAN_IN] = {false};
    bool dest_ready[MAX_SWITCH_FAN_OUT] = {false};

    // debug_log(0x33330000 | num_input_queues);
    // for (uint32_t i = 0; i < num_input_queues; i++) {
    //     debug_log(input_queue_array[i].get_remote_ready_status());
    // }
    // debug_log(0x44440000 | num_output_queues);
    // for (uint32_t i = 0; i < num_output_queues; i++) {
    //     debug_log(output_queue_array[i].get_remote_ready_status());
    // }
    uint32_t iters = 0;

    uint64_t start_timestamp = c_tensix_core::read_wall_clock();
    while (!all_src_dest_ready) {
        iters++;
        if (timeout_cycles > 0) {
            uint64_t cycles_since_start = c_tensix_core::read_wall_clock() - start_timestamp;
            if (cycles_since_start > timeout_cycles) {
                return false;
            }
        }
        all_src_dest_ready = true;
        for (uint32_t i = 0; i < num_input_queues; i++) {
            if (!src_ready[i]) {
                src_ready[i] = input_queue_array[i].is_remote_ready();
                if (!src_ready[i]) {
                    input_queue_array[i].send_remote_ready_notification();
                    all_src_dest_ready = false;
                } else {
                    // debug_log(0x11110000 | i);
                    // debug_log(input_queue_array[i].get_remote_ready_status());
                }
            }
        }
        for (uint32_t i = 0; i < num_output_queues; i++) {
            if (!dest_ready[i]) {
                dest_ready[i] = output_queue_array[i].is_remote_ready();
                if (dest_ready[i]) {
                    output_queue_array[i].send_remote_ready_notification();
                    // debug_log(0x22220000 | i);
                    // debug_log(output_queue_array[i].get_remote_ready_status());
                } else {
                    all_src_dest_ready = false;
                }
            }
        }
    }
    // debug_log(0x55555555);
    // debug_log(iters);
    return true;
}
