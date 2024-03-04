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

extern volatile uint32_t* debug_buf;
extern uint32_t debug_buf_index;
extern uint32_t debug_buf_size;

inline void debug_set_buf(volatile uint32_t* buf, uint32_t size) {
    debug_buf = buf;
    debug_buf_index = 0;
    debug_buf_size = size;
}

inline void debug_log(uint32_t val) {
    debug_buf[debug_buf_index++] = val;
    if (debug_buf_index >= debug_buf_size) {
        debug_buf_index = 0;
    }
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

    bool queue_is_input;

    uint32_t remote_x, remote_y; // remote source for input queues, remote dest for output queues
    uint32_t remote_queue_id;
    DispatchRemoteNetworkType remote_update_network_type;

    // For read/write pointers, we use stream credit registers with auto-increment.
    // Pointers are in 16B units, and we assume buffer size is power of 2 so we get
    // automatic wrapping. (If needed, we can fix the pointer advance functions later
    // to handle non-power-of-2 buffer sizes.)

    inline uint32_t get_queue_local_wptr() const {
        return NOC_STREAM_READ_REG(2*this->queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    }

    inline void reset_queue_local_wptr() {
        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        NOC_STREAM_WRITE_REG(2*this->queue_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);
    }

    inline void advance_queue_local_wptr(uint32_t num_words) {
        NOC_STREAM_WRITE_REG(2*this->queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX, num_words << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }

    inline uint32_t get_queue_local_rptr() const {
        return NOC_STREAM_READ_REG(2*this->queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    }

    inline void reset_queue_local_rptr()  {
        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        NOC_STREAM_WRITE_REG(2*this->queue_id+1, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);
    }

    inline void advance_queue_local_rptr(uint32_t num_words)  {
        NOC_STREAM_WRITE_REG(2*this->queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX, num_words << REMOTE_DEST_BUF_WORDS_FREE_INC);
    }

    inline uint32_t get_queue_data_num_words() const {
        return this->get_queue_local_wptr() - this->get_queue_local_rptr();
    }

    inline uint32_t get_queue_data_num_words_before_wrap() const {
        return this->get_queue_local_wptr() - this->get_queue_local_rptr();
    }

    inline uint32_t get_queue_wptr_offset_words() const {
        return this->get_queue_local_wptr() & (this->queue_size_words - 1);
    }

    inline uint32_t get_queue_rptr_offset_words() const {
        return this->get_queue_local_rptr() & (this->queue_size_words - 1);
    }

    inline uint32_t get_queue_words_before_wrap() const {
        return queue_size_words - this->get_queue_rptr_offset_words();
    }

    inline uint32_t get_queue_free_space_num_words() const {
        return this->queue_size_words - this->get_queue_data_num_words();
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
        uint32_t reg_addr = STREAM_REG_ADDR(2*this->remote_queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = STREAM_REG_ADDR(2*this->remote_queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_reg_update(reg_addr, val);
    }

    inline void reset_ready_flag() {
        NOC_STREAM_WRITE_REG(this->queue_id, STREAM_REMOTE_SRC_REG_INDEX, 0);
    }

    inline void send_remote_ready_notification() {
        uint32_t reg_addr = STREAM_REG_ADDR(this->remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
        this->remote_reg_update(reg_addr, 0x1);
    }

    inline bool is_remote_ready() const {
        return NOC_STREAM_READ_REG(this->queue_id, STREAM_REMOTE_SRC_REG_INDEX) != 0;
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
        debug_log(this->get_queue_local_rptr());
        debug_log(static_cast<uint32_t>(this->remote_update_network_type));
    }
};


class packet_input_queue_state_t : public packet_queue_state_t {

protected:

    bool curr_packet_valid;
    uint32_t curr_packet_dest;
    uint32_t curr_packet_size_words;
    uint32_t curr_packet_words_sent;

    uint32_t words_in_flight;

    inline void advance_next_packet() {
        if(this->get_queue_data_num_words() == 0) {
            this->curr_packet_valid = false;
        } else {
            volatile tt_l1_ptr dispatch_packet_header_t* next_packet_header_ptr =
                reinterpret_cast<volatile tt_l1_ptr dispatch_packet_header_t*>(
                    (this->queue_start_addr_words + this->get_queue_rptr_offset_words())*PACKET_WORD_SIZE_BYTES
                );
            this->curr_packet_dest = next_packet_header_ptr->packet_dest;
            this->curr_packet_size_words = next_packet_header_ptr->packet_size_words;
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
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_queue_id = remote_queue_id;
        this->remote_update_network_type = remote_update_network_type;
        this->curr_packet_valid = false;
        this->reset_queue_local_rptr();
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }

    inline void wait_for_src_ready() {
        while (!this->is_remote_ready()) {
            this->send_remote_ready_notification();
            this->yield();
        }
    }

    inline uint32_t input_queue_data_num_words_available() {
        uint32_t num_words = this->get_queue_data_num_words();
        if (num_words != 0 && !this->curr_packet_valid) {
            this->advance_next_packet();
        }
        uint32_t words_before_wrap = this->get_queue_words_before_wrap();
        num_words = std::min(num_words, words_before_wrap);
        uint32_t curr_packet_words_remaining = this->curr_packet_size_words - this->curr_packet_words_sent;
        num_words = std::min(num_words, curr_packet_words_remaining);
        return num_words;
    }

    inline void input_queue_increment_words_in_flight(uint32_t num_words) {
        if (num_words > 0) {
            this->words_in_flight += num_words;
            this->advance_queue_local_rptr(num_words);
            uint32_t curr_packet_words_remaining = this->curr_packet_size_words - this->curr_packet_words_sent;
            if (num_words == curr_packet_words_remaining) {
                this->advance_next_packet();
            }
        }
    }

    inline void input_queue_flush_in_flight_words(uint32_t num_words) {
        if (num_words > 0) {
            this->words_in_flight -= num_words;
            this->advance_queue_remote_rptr(num_words);
        }
    }

    inline void input_queue_flush_in_flight_data() {
        if (this->words_in_flight > 0) {
            this->advance_queue_remote_rptr(this->words_in_flight);
            this->words_in_flight = 0;
        }
    }

    void debug_log_object() {
        packet_queue_state_t::debug_log_object();
        debug_log(0xFFFFFFFF);
        debug_log(this->curr_packet_valid);
        debug_log(this->curr_packet_dest);
        debug_log(this->curr_packet_size_words);
        debug_log(this->curr_packet_words_sent);
        debug_log(this->words_in_flight);
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

        void prev_words_in_flight_flush() {
            uint32_t prev_index = this->curr_index ^ 1;
            if (this->total_words_in_flight[prev_index] > 0) {
                for (uint32_t i = 0; i < MAX_SWITCH_FAN_IN; i++) {
                    uint32_t words_in_flight = this->input_queue_words_in_flight[prev_index][i];
                    if (words_in_flight > 0) {
                        this->input_queue_array[i].input_queue_flush_in_flight_words(i);
                        this->input_queue_words_in_flight[prev_index][i] = 0;
                    }
                }
                this->total_words_in_flight[prev_index] = 0;
            }
            this->curr_index = prev_index;
        }

        void register_words_in_flight(uint32_t input_queue_id, uint32_t num_words) {
            this->input_queue_words_in_flight[this->curr_index][input_queue_id] += num_words;
            this->total_words_in_flight[this->curr_index] += num_words;
            this->input_queue_array[input_queue_id].input_queue_increment_words_in_flight(num_words);
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
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_queue_id = remote_queue_id;
        this->remote_update_network_type = remote_update_network_type;
        this->max_noc_send_words = DEFAULT_MAX_NOC_SEND_WORDS;
        this->max_eth_send_words = DEFAULT_MAX_ETH_SEND_WORDS;
        this->input_queue_status.init(input_queue_array);
        this->reset_queue_local_rptr();
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }

    inline void set_max_noc_send_words(uint32_t max_noc_send_words) {
        this->max_noc_send_words = max_noc_send_words;
    }

    inline void set_max_eth_send_words(uint32_t max_eth_send_words) {
        this->max_eth_send_words = max_eth_send_words;
    }

    inline void wait_for_dest_ready() {
        while (!this->is_remote_ready()) {
            this->yield();
        }
        this->send_remote_ready_notification();
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

    inline void output_queue_data_barrier() {
        while (this->get_queue_data_num_words() > 0) {
            this->yield();
        }
    }

    inline void prev_words_in_flight_barrier() {
        while (this->get_queue_data_num_words() > this->input_queue_status.get_curr_total_words_in_flight()) {
            this->yield();
        }
        this->input_queue_status.prev_words_in_flight_flush();
    }


    inline uint32_t get_num_words_to_send(uint32_t input_queue_index, uint32_t max_num_words = UINT32_MAX) {

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);

        uint32_t num_words_available_in_input = input_queue_ptr->input_queue_data_num_words_available();
        uint32_t num_words_free_in_output = this->get_queue_free_space_num_words();
        uint32_t num_words_to_forward = std::min(num_words_available_in_input, num_words_free_in_output);

        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t output_buf_words_before_wrap = this->get_queue_words_before_wrap();
        num_words_to_forward = std::min(num_words_to_forward, output_buf_words_before_wrap);
        num_words_to_forward = std::min(num_words_to_forward, this->output_max_num_words_to_forward());
        num_words_to_forward = std::min(num_words_to_forward, max_num_words);

        return num_words_to_forward;
    }


    inline uint32_t forward_data_from_input(uint32_t input_queue_index, uint32_t max_num_words = UINT32_MAX) {

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);

        uint32_t num_words_to_forward = this->get_num_words_to_send(input_queue_index, max_num_words);
        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t src_addr =
            (input_queue_ptr->queue_start_addr_words +
             input_queue_ptr->get_queue_rptr_offset_words())*PACKET_WORD_SIZE_BYTES;

        uint32_t dest_addr =
            (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;

        this->send_data_to_remote(src_addr, dest_addr, num_words_to_forward);

        this->input_queue_status.register_words_in_flight(input_queue_index, num_words_to_forward);

        this->advance_queue_local_wptr(num_words_to_forward);
        this->remote_wptr_update(num_words_to_forward);

        return num_words_to_forward;
    }

    void debug_log_object() {
        packet_queue_state_t::debug_log_object();
        debug_log(0xeeeeeeee);
        debug_log(this->output_max_num_words_to_forward());
        uint32_t data_sent = 0;
        if (this->remote_update_network_type < DispatchRemoteNetworkType::ETH) {
            data_sent = NOC_STATUS_READ_REG(this->remote_update_network_type, NIU_MST_NONPOSTED_WR_REQ_SENT);
        }
        debug_log(data_sent);
        debug_log(0xFFFFFFFF);
        this->input_queue_status.debug_log_object();
    }

};
