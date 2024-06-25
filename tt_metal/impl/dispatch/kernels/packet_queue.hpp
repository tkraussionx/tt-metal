// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "packet_queue_ctrl.hpp"
#include "risc_attribs.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_io_kernel_helpers.hpp"


constexpr uint32_t NUM_WR_CMD_BUFS = 4;

constexpr uint32_t DEFAULT_MAX_NOC_SEND_WORDS = (NUM_WR_CMD_BUFS-1)*(NOC_MAX_BURST_WORDS*NOC_WORD_BYTES)/PACKET_WORD_SIZE_BYTES;
constexpr uint32_t DEFAULT_MAX_ETH_SEND_WORDS = 2*1024;

constexpr uint32_t NUM_PTR_REGS_PER_QUEUE = 3;


inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

inline uint64_t get_timestamp_32b() {
    return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

void zero_l1_buf(tt_l1_ptr uint32_t* buf, uint32_t size_bytes) {
    for (uint32_t i = 0; i < size_bytes/4; i++) {
        buf[i] = 0;
    }
}

static FORCE_INLINE
void write_test_results(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE
void set_64b_result(uint32_t* buf, uint64_t val, uint32_t index = 0) {
    if (buf != nullptr) {
        buf[index] = val >> 32;
        buf[index+1] = val & 0xFFFFFFFF;
    }
}

typedef struct dispatch_packet_header_t dispatch_packet_header_t;

static_assert(sizeof(dispatch_packet_header_t) == PACKET_WORD_SIZE_BYTES);


class packet_queue_state_t;
class packet_input_queue_state_t;
class packet_output_queue_state_t;

class packet_queue_state_t {

    volatile uint32_t* local_wptr_val;
    volatile uint32_t* local_rptr_sent_val;
    volatile uint32_t* local_rptr_cleared_val;
    volatile uint32_t* local_wptr_update;
    volatile uint32_t* local_rptr_sent_update;
    volatile uint32_t* local_rptr_cleared_update;
    volatile uint32_t* local_wptr_reset;
    volatile uint32_t* local_rptr_sent_reset;
    volatile uint32_t* local_rptr_cleared_reset;

    uint32_t remote_ready_status_addr;
    volatile uint32_t* local_ready_status_ptr;

    uint32_t remote_wptr_update_addr;
    uint32_t remote_rptr_sent_update_addr;
    uint32_t remote_rptr_cleared_update_addr;

protected:

    // cb mode true => go from unpacketized to packetized domain. aka add packet header to messages
    bool cb_mode;
    uint32_t cb_mode_page_size_words;
    uint8_t cb_mode_local_sem_id;
    uint8_t cb_mode_remote_sem_id;

public:

    uint8_t queue_id;
    uint32_t queue_start_addr_words;
    uint32_t queue_size_words;
    uint32_t ptr_offset_mask;
    uint32_t queue_size_mask;

    bool queue_is_input;

    uint8_t remote_x, remote_y; // remote source for input queues, remote dest for output queues
    uint8_t remote_queue_id;
    DispatchRemoteNetworkType remote_update_network_type;

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint32_t queue_size_words,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type,
              bool cb_mode,
              uint8_t cb_mode_local_sem_id,
              uint8_t cb_mode_remote_sem_id,
              uint8_t cb_mode_log_page_size) {

        this->queue_id = queue_id;
        this->queue_start_addr_words = queue_start_addr_words;
        this->queue_size_words = queue_size_words;
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_queue_id = remote_queue_id;
        this->remote_update_network_type = remote_update_network_type;

        this->cb_mode = cb_mode;
        this->cb_mode_local_sem_id = cb_mode_local_sem_id;
        this->cb_mode_remote_sem_id = cb_mode_remote_sem_id;
        this->cb_mode_page_size_words = (((uint32_t)0x1) << cb_mode_log_page_size)/PACKET_WORD_SIZE_BYTES;

        // Misc. register definitions below.

        // For read/write pointers, we use stream credit registers with auto-increment.
        // Pointers are in 16B units, and we assume buffer size is power of 2 so we get
        // automatic wrapping. (If needed, we can fix the pointer advance functions later
        // to handle non-power-of-2 buffer sizes.)

        // For source/destination ready synchronization signals, we use misc. registers in
        // streams that behave like scratch registers and are reset to 0.

        this->local_wptr_val = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        this->local_rptr_sent_val = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        this->local_rptr_cleared_val = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+2, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));

        this->local_wptr_update = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        this->local_rptr_sent_update = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        this->local_rptr_cleared_update = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+2, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));

        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        this->local_wptr_reset = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
        this->local_rptr_sent_reset = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+1, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
        this->local_rptr_cleared_reset = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+2, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));

        this->remote_wptr_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*remote_queue_id,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_rptr_sent_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*remote_queue_id+1,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_rptr_cleared_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*remote_queue_id+2,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);

        this->remote_ready_status_addr = STREAM_REG_ADDR(remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
        this->local_ready_status_ptr = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(queue_id, STREAM_REMOTE_SRC_REG_INDEX));
    }

    inline uint8_t get_queue_id() const {
        return this->queue_id;
    }

    inline uint32_t get_queue_local_wptr() const {
        return *this->local_wptr_val;
    }

    inline void reset_queue_local_wptr() {
        *this->local_wptr_reset = 0;
    }

    inline void advance_queue_local_wptr(uint32_t num_words) {
        *this->local_wptr_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    inline uint32_t get_queue_local_rptr_sent() const {
        return *this->local_rptr_sent_val;
    }

    inline uint32_t get_queue_local_rptr_cleared() const {
        return *this->local_rptr_cleared_val;
    }

    inline void reset_queue_local_rptr_sent()  {
        *this->local_rptr_sent_reset = 0;
    }

    inline void reset_queue_local_rptr_cleared()  {
        *this->local_rptr_cleared_reset = 0;
    }

    inline void advance_queue_local_rptr_sent(uint32_t num_words)  {
        *this->local_rptr_sent_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    inline void advance_queue_local_rptr_cleared(uint32_t num_words)  {
        *this->local_rptr_cleared_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    // /*virtual*/ inline uint32_t get_queue_data_num_words_available_to_send() const {
    //     return (this->get_queue_local_wptr() - this->get_queue_local_rptr_sent()) & this->queue_size_mask;
    // }

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

    inline void remote_reg_update(uint32_t reg_addr, uint32_t val) {

        if ((this->remote_update_network_type == DispatchRemoteNetworkType::NONE) || this->cb_mode) {
            return;
        }
        else if (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) {
            eth_write_remote_reg(reg_addr, val);
        } else {
            uint64_t dest_addr = get_noc_addr(this->remote_x, this->remote_y, reg_addr);
            noc_inline_dw_write(dest_addr, val);
        }
    }

    inline void advance_queue_remote_wptr(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_wptr_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr_sent(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_rptr_sent_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr_cleared(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_rptr_cleared_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void reset_ready_flag() {
        *this->local_ready_status_ptr = 0;
    }

    inline void send_remote_ready_notification() {
        this->remote_reg_update(this->remote_ready_status_addr,
                                PACKET_QUEUE_REMOTE_READY_FLAG);
    }

    inline void set_remote_ready_status_addr(uint8_t remote_queue_id) {
        this->remote_ready_status_addr = STREAM_REG_ADDR(remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
    }

    inline void send_remote_finished_notification() {
        this->remote_reg_update(this->remote_ready_status_addr,
                                PACKET_QUEUE_REMOTE_FINISHED_FLAG);
    }

    inline bool is_remote_ready() const {
        return *this->local_ready_status_ptr == PACKET_QUEUE_REMOTE_READY_FLAG;
    }

    inline uint32_t get_remote_ready_status() const {
        return *this->local_ready_status_ptr;
    }

    inline bool is_remote_finished() const {
        return *this->local_ready_status_ptr == PACKET_QUEUE_REMOTE_FINISHED_FLAG;
    }

    inline uint32_t cb_mode_get_local_sem_val() {
        if (!this->cb_mode) {
            return 0;
        }
        volatile tt_l1_ptr uint32_t* local_sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(this->cb_mode_local_sem_id));
        // semaphore underflow is currently used to signal path teardown with minimal prefetcher changes
        uint32_t val = *local_sem_addr;
        if (val & 0x80000000) {
            val &= 0x7fffffff;
            *this->local_ready_status_ptr = PACKET_QUEUE_REMOTE_FINISHED_FLAG;
        }
        return val;
    }

    inline bool cb_mode_local_sem_downstream_complete() {
        if (!this->cb_mode) {
            return false;
        }
        volatile tt_l1_ptr uint32_t* local_sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(this->cb_mode_local_sem_id));
        // semaphore underflow is currently used to signal path teardown with minimal prefetcher changes
        uint32_t val = *local_sem_addr;
        return (val & 0x80000000);
    }

    inline void cb_mode_inc_local_sem_val(uint32_t val) {
        if (this->cb_mode) {
            uint32_t sem_l1_addr = get_semaphore(this->cb_mode_local_sem_id);
            uint64_t sem_noc_addr = get_noc_addr(sem_l1_addr);
            noc_semaphore_inc(sem_noc_addr, val);
            noc_async_atomic_barrier();
        }
    }

    inline void cb_mode_inc_remote_sem_val(uint32_t val) {
        uint32_t sem_l1_addr = get_semaphore(this->cb_mode_remote_sem_id);
        uint64_t sem_noc_addr = get_noc_addr(remote_x, remote_y, sem_l1_addr);
        if (this->cb_mode && (val > 0)) {
            noc_semaphore_inc(sem_noc_addr, val);
        }
    }

    inline uint32_t cb_mode_rptr_sent_advance_page_align() {
        uint32_t rptr_val = this->get_queue_local_rptr_sent();
        uint32_t page_size_words_mask = this->cb_mode_page_size_words - 1;
        uint32_t num_words_past_page_boundary = rptr_val & page_size_words_mask;
        uint32_t input_pad_words_skipped = 0;
        if (num_words_past_page_boundary > 0) {
            input_pad_words_skipped = this->cb_mode_page_size_words - num_words_past_page_boundary;
            this->advance_queue_local_rptr_sent(input_pad_words_skipped);
        }
        return input_pad_words_skipped;
    }

    inline void cb_mode_local_sem_wptr_update() {
        uint32_t local_sem_val = this->cb_mode_get_local_sem_val();
        for (uint32_t i = 0; i < local_sem_val; i++) {
            this->advance_queue_local_wptr(this->cb_mode_page_size_words);
        }
        this->cb_mode_inc_local_sem_val(-local_sem_val);
    }

    inline void cb_mode_local_sem_rptr_cleared_update() {
        uint32_t local_sem_val = this->cb_mode_get_local_sem_val();
        for (uint32_t i = 0; i < local_sem_val; i++) {
            this->advance_queue_local_rptr_cleared(this->cb_mode_page_size_words);
        }
        this->cb_mode_inc_local_sem_val(-local_sem_val);
    }

    void yield() {
        // TODO: implement yield for ethernet here
    }

    void dprint_object() {
        DPRINT << "  id: " << DEC() << static_cast<uint32_t>(this->queue_id) << ENDL();
        DPRINT << "  start_addr: 0x" << HEX() << static_cast<uint32_t>(this->queue_start_addr_words*PACKET_WORD_SIZE_BYTES) << ENDL();
        DPRINT << "  size_bytes: 0x" << HEX() << static_cast<uint32_t>(this->queue_size_words*PACKET_WORD_SIZE_BYTES) << ENDL();
        DPRINT << "  remote_x: " << DEC() << static_cast<uint32_t>(this->remote_x) << ENDL();
        DPRINT << "  remote_y: " << DEC() << static_cast<uint32_t>(this->remote_y) << ENDL();
        DPRINT << "  remote_queue_id: " << DEC() << static_cast<uint32_t>(this->remote_queue_id) << ENDL();
        DPRINT << "  remote_update_network_type: " << DEC() << static_cast<uint32_t>(this->remote_update_network_type) << ENDL();
        DPRINT << "  ready_status: 0x" << HEX() << this->get_remote_ready_status() << ENDL();
        DPRINT << "  local_wptr: 0x" << HEX() << this->get_queue_local_wptr() << ENDL();
        DPRINT << "  local_rptr_sent: 0x" << HEX() << this->get_queue_local_rptr_sent() << ENDL();
        DPRINT << "  local_rptr_cleared: 0x" << HEX() << this->get_queue_local_rptr_cleared() << ENDL();
    }
};


class packet_input_queue_state_t : public packet_queue_state_t {
public:
    fabric_receiver_stream_state_t stream_state;

protected:
    bool curr_packet_valid;
    tt_l1_ptr dispatch_packet_header_t* curr_packet_header_ptr;
    //
    uint16_t curr_packet_src;
    //
    uint16_t curr_packet_dest;
    //
    uint32_t curr_packet_size_words;
    //
    uint32_t end_of_cmd;
    // I think this is only for padding packets to packet boundary by skipping through the
    // buffer by that many words
    uint32_t curr_packet_words_sent;
    // for packet switching. Obtainable from payload header
    uint32_t curr_packet_tag;
    // for packet switching. Obtainable from payload header
    uint16_t curr_packet_flags;

    //
    uint32_t packetizer_page_words_cleared;

    /*
     * When data is available to send from this queue, this function updates the internal `curr_packet_*`
     * fields hold the information about the payload (e.g. size, src, dest, etc.)
     * This function doesn't do any sort of sending of the actual payload
     */
    inline void advance_next_packet() {
        bool rx_from_stream = this->remote_update_network_type == DispatchRemoteNetworkType::STREAM;
        bool packet_available = (rx_from_stream && messages_are_available(stream_state)) || (this->get_queue_data_num_words_available_to_send() > 0);
        // if (rx_from_stream) {
        //     // DPRINT << "RR: messages_available: {}" << (uint32_t)messages_are_available(stream_state) << "\n";
        //     DPRINT << "RR: n_words_available: " << (uint32_t)this->get_queue_data_num_words_available_to_send() << "\n";
        // }
        if(packet_available) {
            if (rx_from_stream) {
                DPRINT << "RR: pkt avail\n";
                // advance_remote_receiver_stream_to_next_message(
                //     stream_state, get_next_available_stream_message_size_in_bytes(stream_state));
            }
            // uses get_queue_rptr_sent_offset_words
            // -> Under the hood uses local_rptr_sent_val, which is actually a pointer...
            //    I expect this is the current offset...
            tt_l1_ptr dispatch_packet_header_t* next_packet_header_ptr =
                reinterpret_cast<tt_l1_ptr dispatch_packet_header_t*>(
                    rx_from_stream ?
                    stream_state.get_current_local_buffer_address() :
                    (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES
                );
            this->curr_packet_header_ptr = next_packet_header_ptr;
            uint32_t packet_size_bytes = curr_packet_header_ptr->get_packet_size_bytes();
            // DPRINT << "\tpacket_size_bytes=" << packet_size_bytes << "\n";
            if (rx_from_stream) {
                // Indicates we are a remote receiver receving from a stream.
                //
                // Streams expect sizes to be in num 16B words, which isn't bytes, so if we are
                // reading a packet header that came from a stream it means we need to convert
                // it back to size in bytes, from size in 16B words.
                packet_size_bytes = packet_size_bytes << 4;
                // DPRINT << "\t\tpacket_size_bytes=" << packet_size_bytes << "\n";
            }

            this->end_of_cmd = curr_packet_header_ptr->get_is_end_of_cmd();
            this->curr_packet_size_words = packet_size_bytes/PACKET_WORD_SIZE_BYTES;
            if (packet_size_bytes % PACKET_WORD_SIZE_BYTES) {
                this->curr_packet_size_words++;
            }
            if (this->cb_mode) {
                // prefetcher has size in bytes
                next_packet_header_ptr->packet_dest = this->curr_packet_dest;
                next_packet_header_ptr->packet_src = this->curr_packet_src;
                next_packet_header_ptr->tag = this->curr_packet_tag;
                next_packet_header_ptr->packet_flags = this->curr_packet_flags;
            } else {
                this->curr_packet_dest = next_packet_header_ptr->packet_dest;
                this->curr_packet_src = next_packet_header_ptr->packet_src;
                this->curr_packet_tag = next_packet_header_ptr->tag;
                this->curr_packet_flags = next_packet_header_ptr->packet_flags;
            }
            this->curr_packet_words_sent = 0;
            this->curr_packet_valid = true;

            // Now that we've extracted the header information, we can advance the packet?
            // but this will clear the packet... what's the point of grabbing the header?
            // then it must mean we're pointing to the next packet (which may not have arrived yet?)
            // so what's going on here unless it's just to book keep so keep

       }
    }

public:

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint32_t queue_size_words,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type,
              bool packetizer_input = false,
              uint8_t packetizer_input_log_page_size = 0,
              uint8_t packetizer_input_sem_id = 0,
              uint8_t packetizer_input_remote_sem_id = 0,
              uint16_t packetizer_input_src = 0,
              uint16_t packetizer_input_dest = 0) {

        packet_queue_state_t::init(queue_id, queue_start_addr_words, queue_size_words,
                                   remote_x, remote_y, remote_queue_id, remote_update_network_type,
                                   packetizer_input, packetizer_input_sem_id,
                                   packetizer_input_remote_sem_id,
                                   packetizer_input_log_page_size);

        tt_l1_ptr uint32_t* queue_ptr =
            reinterpret_cast<tt_l1_ptr uint32_t*>(queue_start_addr_words*PACKET_WORD_SIZE_BYTES);
        // zero_l1_buf(queue_ptr, queue_size_words*PACKET_WORD_SIZE_BYTES);

        this->packetizer_page_words_cleared = 0;

        if (packetizer_input) {
            this->curr_packet_src = packetizer_input_src;
            this->curr_packet_dest = packetizer_input_dest;
            this->curr_packet_flags = 0;
            this->curr_packet_tag = 0xabcd;
        }

        this->ptr_offset_mask = queue_size_words - 1;
        this->queue_size_mask = (queue_size_words << 1) - 1;
        this->curr_packet_valid = false;
        this->reset_queue_local_rptr_sent();
        this->reset_queue_local_rptr_cleared();
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }

    uint32_t get_queue_data_num_words_available_to_send() const {
        if (this->remote_update_network_type == DispatchRemoteNetworkType::STREAM) {
            // DPRINT << "RR stream_id: " << (uint32_t)this->stream_state.local_stream_id << "\n";
            // DPRINT << "RR stream: " << (uint32_t)this->stream_state.local_stream_id << "\n";
            uint32_t n = fw_managed_rx_stream_num_bytes_available(this->stream_state.local_stream_id, this->stream_state);
            if (n > 0) {
                DPRINT << "RR local_msg_info_ptr: " << (uint32_t)this->stream_state.local_msg_info_ptr << "\n";
                DPRINT << "RR wrptr: " << NOC_STREAM_READ_REG(this->stream_state.local_stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX) << "\n";
            }
            // DPRINT << "XxX: " << n << "\n";
            if (!((n & 0xF) == 0)) {
                DPRINT << "n=" << n << "\n";
            }
            ASSERT((n & 0xF) == 0);
            return n >> 4;
        } else {
            uint32_t size = (this->get_queue_local_wptr() - this->get_queue_local_rptr_sent()) & this->queue_size_mask;
            return size;
        }
    }

    inline uint32_t get_end_of_cmd() const {
        return this->end_of_cmd;
    }

    inline bool is_packetizer_input() const {
        return this->cb_mode;
    }

    // E.g. send 18k to core
    // prefetcher is in unpacketized domain.
    // prefetcher gets raw data
    // prefetcher prepends a header to the payload with size information when sending
    // to packetized domain

    inline bool get_curr_packet_valid() {
        if (this->cb_mode) {
            this->cb_mode_local_sem_wptr_update();
        }
        // stream -> auto const &[msg_addr, msg_size_bytes] = get_next_message_info(stream_id, stream_state);
        if (!this->curr_packet_valid && (this->get_queue_data_num_words_available_to_send() > 0)){
            // stream:
            //  auto const &[msg_addr, msg_size_bytes] = get_next_message_info(stream_id, stream_state);
            //  ASSERT(msg_size_bytes > 0);
            //  ASSERT(msg_size_bytes <= stream_state.local_buffer_size);
            //  copy_message_to_cb_blocking(cb, msg_addr, msg_size_bytes, stream_state);
            //
            // flushes the stream buffer - sends credits to relay stream
            // stream_relay_tiles(stream_id, 1, msg_size_bytes >> 4);
            this->advance_next_packet();
        }
        return this->curr_packet_valid;
    }

    inline tt_l1_ptr dispatch_packet_header_t* get_curr_packet_header_ptr() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_header_ptr;
    }

    inline uint32_t get_curr_packet_dest() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_dest;
    }

    inline uint32_t get_curr_packet_src() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_src;
    }

    inline uint32_t get_curr_packet_size_words() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_size_words;
    }

    inline uint32_t get_curr_packet_words_remaining() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_size_words - this->curr_packet_words_sent;
    }

    inline uint32_t get_curr_packet_tag() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_tag;
    }

    inline uint32_t get_curr_packet_flags() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_flags;
    }

    inline bool partial_packet_sent() const {
        return this->curr_packet_valid && (this->curr_packet_words_sent > 0);
    }

    inline bool curr_packet_start() const {
        return this->curr_packet_valid && (this->curr_packet_words_sent == 0);
    }

    inline bool input_queue_full_packet_available_to_send(uint32_t& num_words_available_to_send) {
        num_words_available_to_send = this->get_queue_data_num_words_available_to_send();
        if (num_words_available_to_send == 0) {
            return false;
        }
        return num_words_available_to_send >= this->get_curr_packet_words_remaining();
    }

    inline uint32_t input_queue_curr_packet_num_words_available_to_send() {
        if (this->cb_mode) {
            ASSERT(this->remote_update_network_type != DispatchRemoteNetworkType::STREAM);
            this->cb_mode_local_sem_wptr_update();
        }
        uint32_t num_words = this->get_queue_data_num_words_available_to_send();
        if (num_words == 0) {
            return 0;
        }
        num_words = std::min(num_words, this->get_curr_packet_words_remaining());
        return num_words;
    }

    // returns the number of words skipped for page padding if in packetizer mode
    inline uint32_t input_queue_advance_words_sent(uint32_t num_words) {
        uint32_t input_pad_words_skipped = 0;
        if (num_words > 0) {
            this->advance_queue_local_rptr_sent(num_words);
            this->advance_queue_remote_rptr_sent(num_words);
            this->curr_packet_words_sent += num_words;
            uint32_t curr_packet_words_remaining = this->get_curr_packet_words_remaining();
            if (curr_packet_words_remaining == 0) {
                if (this->is_packetizer_input()) {
                    input_pad_words_skipped = this->cb_mode_rptr_sent_advance_page_align();
                }
                this->curr_packet_valid = false;
                this->advance_next_packet();
            }
        }
        return input_pad_words_skipped;
    }

    inline void input_queue_advance_words_cleared(uint32_t num_words) {
        if (num_words > 0) {
            this->advance_queue_local_rptr_cleared(num_words);
            this->advance_queue_remote_rptr_cleared(num_words);
            if (this->is_packetizer_input()) {
                this->packetizer_page_words_cleared += num_words;
                uint32_t remote_sem_inc = 0;
                while (this->packetizer_page_words_cleared >= this->cb_mode_page_size_words) {
                    remote_sem_inc++;
                    this->packetizer_page_words_cleared -= this->cb_mode_page_size_words;
                }
                this->cb_mode_inc_remote_sem_val(remote_sem_inc);
            }
        }
    }

    inline void input_queue_clear_all_words_sent() {
        uint32_t num_words = this->get_num_words_sent_not_cleared();
        if (num_words > 0) {
            this->input_queue_advance_words_cleared(num_words);
        }
    }

    void dprint_object() {
        DPRINT << "Input queue:" << ENDL();
        packet_queue_state_t::dprint_object();
        DPRINT << "  packet_valid: " << DEC() << static_cast<uint32_t>(this->curr_packet_valid) << ENDL();
        DPRINT << "  packet_tag: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_tag) << ENDL();
        DPRINT << "  packet_src: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_src) << ENDL();
        DPRINT << "  packet_dest: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_dest) << ENDL();
        DPRINT << "  packet_flags: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_flags) << ENDL();
        DPRINT << "  packet_size_words: " << DEC() << static_cast<uint32_t>(this->curr_packet_size_words) << ENDL();
        DPRINT << "  packet_words_sent: " << DEC() << static_cast<uint32_t>(this->curr_packet_words_sent) << ENDL();
    }

};


class packet_output_queue_state_t : public packet_queue_state_t {
public:
    fabric_sender_stream_state_t stream_state;

protected:
    uint32_t max_noc_send_words;
    uint32_t max_eth_send_words;

    uint32_t unpacketizer_page_words_sent;
    bool unpacketizer_remove_header;

    struct {

        packet_input_queue_state_t* input_queue_array;
        uint32_t input_queue_words_in_flight[2*MAX_SWITCH_FAN_IN];

        uint32_t* curr_input_queue_words_in_flight;
        uint32_t* prev_input_queue_words_in_flight;
        uint32_t curr_output_total_words_in_flight;
        uint32_t prev_output_total_words_in_flight;

        uint8_t num_input_queues;

        void init(packet_input_queue_state_t* input_queue_array, uint32_t num_input_queues) {
            this->num_input_queues = num_input_queues;
            this->input_queue_array = input_queue_array;
            this->curr_input_queue_words_in_flight = &(this->input_queue_words_in_flight[0]);
            this->prev_input_queue_words_in_flight = &(this->input_queue_words_in_flight[MAX_SWITCH_FAN_IN]);
            this->curr_output_total_words_in_flight = 0;
            this->prev_output_total_words_in_flight = 0;
            for (uint32_t i = 0; i < MAX_SWITCH_FAN_IN; i++) {
                this->curr_input_queue_words_in_flight[i] = 0;
                this->prev_input_queue_words_in_flight[i] = 0;
            }
        }

        inline uint32_t get_curr_output_total_words_in_flight() const {
            return this->curr_output_total_words_in_flight;
        }

        inline uint32_t get_prev_output_total_words_in_flight() const {
            return this->prev_output_total_words_in_flight;
        }

        // Not sure about the purpose of looping over *all* the inputs here
        inline uint32_t prev_words_in_flight_flush() {

            uint32_t words_flushed = this->prev_output_total_words_in_flight;
            if (words_flushed > 0) {
                for (uint32_t i = 0; i < num_input_queues; i++) {
                    this->input_queue_array[i].input_queue_advance_words_cleared(this->prev_input_queue_words_in_flight[i]);
                    this->prev_input_queue_words_in_flight[i] = 0;
                }
            }

            uint32_t* tmp = this->prev_input_queue_words_in_flight;
            this->prev_input_queue_words_in_flight = this->curr_input_queue_words_in_flight;
            this->curr_input_queue_words_in_flight = tmp;
            this->prev_output_total_words_in_flight = this->curr_output_total_words_in_flight;
            this->curr_output_total_words_in_flight = 0;

            return words_flushed;
        }

        inline void register_words_in_flight(uint32_t input_queue_id, uint32_t num_words) {
            uint32_t input_pad_words_skipped = this->input_queue_array[input_queue_id].input_queue_advance_words_sent(num_words);
            this->curr_input_queue_words_in_flight[input_queue_id] += (num_words + input_pad_words_skipped);
            this->curr_output_total_words_in_flight += num_words;
        }

        void dprint_object() {
            DPRINT << "  curr_output_total_words_in_flight: " << DEC() << this->curr_output_total_words_in_flight << ENDL();
            for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                DPRINT << "       from input queue id " << DEC() <<
                            this->input_queue_array[j].get_queue_id() << ": "
                            << DEC() << this->curr_input_queue_words_in_flight[j]
                            << ENDL();
            }
            DPRINT << "  prev_output_total_words_in_flight: " << DEC() << this->prev_output_total_words_in_flight << ENDL();
            for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                DPRINT << "       from input queue id " << DEC() <<
                            this->input_queue_array[j].get_queue_id() << ": "
                            << DEC() << this->prev_input_queue_words_in_flight[j]
                            << ENDL();
            }
        }

    } input_queue_status;

public:

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint32_t queue_size_words,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type,
              packet_input_queue_state_t* input_queue_array,
              uint8_t num_input_queues,
              bool unpacketizer_output = false,
              uint16_t unpacketizer_output_log_page_size = 0,
              uint8_t unpacketizer_output_sem_id = 0,
              uint8_t unpacketizer_output_remote_sem_id = 0,
              bool unpacketizer_output_remove_header = false) {

        packet_queue_state_t::init(queue_id, queue_start_addr_words, queue_size_words,
                                   remote_x, remote_y, remote_queue_id, remote_update_network_type,
                                   unpacketizer_output, unpacketizer_output_sem_id,
                                   unpacketizer_output_remote_sem_id,
                                   unpacketizer_output_log_page_size);

        this->unpacketizer_remove_header = unpacketizer_output_remove_header;
        this->unpacketizer_page_words_sent = 0;
        this->ptr_offset_mask = queue_size_words - 1;
        this->queue_size_mask = (queue_size_words << 1) - 1;
        this->max_noc_send_words = DEFAULT_MAX_NOC_SEND_WORDS;
        this->max_eth_send_words = DEFAULT_MAX_ETH_SEND_WORDS;
        this->input_queue_status.init(input_queue_array, num_input_queues);
        this->reset_queue_local_rptr_sent();
        this->reset_queue_local_rptr_cleared();
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }

    inline bool is_unpacketizer_output() const {
        return this->cb_mode;
    }

    inline void set_max_noc_send_words(uint32_t max_noc_send_words) {
        this->max_noc_send_words = max_noc_send_words;
    }

    inline void set_max_eth_send_words(uint32_t max_eth_send_words) {
        this->max_eth_send_words = max_eth_send_words;
    }

    inline uint32_t output_max_num_words_to_forward() const {
        return (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) ?
            this->max_eth_send_words : this->max_noc_send_words;
    }

    inline void send_data_to_remote(uint32_t src_base_addr, uint32_t src_buf_size, uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {
        if ((this->remote_update_network_type == DispatchRemoteNetworkType::NONE) ||
            (num_words == 0)) {
            return;
        } else if (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) {
            internal_::eth_send_packet(0, src_addr/PACKET_WORD_SIZE_BYTES, dest_addr/PACKET_WORD_SIZE_BYTES, num_words);
        } else if (this->remote_update_network_type== DispatchRemoteNetworkType::STREAM) {
            // Sending into a stream, from a remote_source_stream on this core, managed by this RISC processor.
            ASSERT(src_addr >= src_base_addr && (src_addr - src_base_addr) < src_buf_size);
            // DPRINT << "src_base_addr: " << HEX() << src_base_addr << "\n";
            // DPRINT << "src_buf_size: " << HEX() << src_buf_size << "\n";
            // DPRINT << "src_addr: " << HEX() << src_addr << "\n";
            stream_noc_write_from_mux(
                src_base_addr,
                src_buf_size,
                src_addr - src_base_addr,
                dest_addr,
                num_words*PACKET_WORD_SIZE_BYTES,
                this->remote_x,
                this->remote_y,
                this->stream_state);
        } else {
            uint64_t noc_dest_addr = get_noc_addr(this->remote_x, this->remote_y, dest_addr);
            noc_async_write(src_addr, noc_dest_addr, num_words*PACKET_WORD_SIZE_BYTES);
        }
    }

    inline void remote_wptr_update(uint32_t num_words) {
        if (this->remote_update_network_type != DispatchRemoteNetworkType::STREAM) {
            // only manually update the remote wrptr if *not* sending to a stream because
            // if sending to a stream, then the wrptrs will automatically be managed by the
            // stream hardware
            this->advance_queue_remote_wptr(num_words);
        }
    }

    inline uint32_t prev_words_in_flight_check_flush() {
        if (this->is_unpacketizer_output()) {
            uint32_t words_written_not_sent = get_num_words_written_not_sent();
            // We expect to never have streams on an unpacketize endpoint
            ASSERT (this->remote_update_network_type != DispatchRemoteNetworkType::STREAM);
            noc_async_writes_flushed();
            this->advance_queue_local_rptr_sent(words_written_not_sent);
            uint32_t words_flushed = this->input_queue_status.prev_words_in_flight_flush();
            this->cb_mode_local_sem_rptr_cleared_update();
            return words_flushed;
        }
        else if (this->get_num_words_written_not_sent() <= this->input_queue_status.get_curr_output_total_words_in_flight()) {
            return this->input_queue_status.prev_words_in_flight_flush();
        }
        else {
            return 0;
        }
    }

    bool output_barrier(uint32_t timeout_cycles = 0) {
        uint32_t start_timestamp = 0;
        if (timeout_cycles > 0) {
            start_timestamp = get_timestamp_32b();
        }
        if (this->is_unpacketizer_output()) {
           noc_async_writes_flushed();
        }
        while (this->get_queue_data_num_words_occupied() > 0) {
            if (this->is_unpacketizer_output()) {
                this->cb_mode_local_sem_rptr_cleared_update();
                if (this->cb_mode_local_sem_downstream_complete()) {
                    // There is no guaranteed that dispatch_h will increment semaphore for all commmands
                    // (specifically the final terminate command).
                    // So just clear whatever remains once the completion signal is received.
                    uint32_t words_occupied = this->get_queue_data_num_words_occupied();
                    this->advance_queue_local_rptr_cleared(words_occupied);
                }
            }
            if (timeout_cycles > 0) {
                uint32_t cycles_elapsed = get_timestamp_32b() - start_timestamp;
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

    /*
     * Given an input queue index, this function will return the size (in 4B? words) that we can send to the output stream
     */
    inline uint32_t get_num_words_to_send(uint32_t input_queue_index) {
        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);

        bool is_remote_sender_to_stream = this->remote_update_network_type == DispatchRemoteNetworkType::STREAM;
        bool is_remote_receiver_from_stream = input_queue_ptr->remote_update_network_type == DispatchRemoteNetworkType::STREAM;

        // if (is_remote_sender_to_stream) {
        //     ;
        //     uint32_t rdptr = (input_queue_ptr->queue_start_addr_words + input_queue_ptr->get_queue_rptr_sent_offset_words()) * PACKET_WORD_SIZE_BYTES;
        //     uint32_t num_words = *reinterpret_cast<volatile uint32_t*>(rdptr);
        //     DPRINT << "rdptr: " << HEX() << rdptr << ", num_words: " << DEC() << num_words << "\n";
        // }

        uint32_t num_words_available_in_input =
            is_remote_receiver_from_stream
                ? fw_managed_rx_stream_num_bytes_available(
                      input_queue_ptr->stream_state.local_stream_id, input_queue_ptr->stream_state) >>
                      4 :

                (is_remote_sender_to_stream ?
                    // Don't truncate at end of buffer and allow wraparound
                    // Additionally, unless we can safely overwrite the packet header, we can't
                    // batch send multiple packets with a single stream send, since the stream looks
                    // at the header in the payload for size information
                    *reinterpret_cast<volatile uint32_t*>((input_queue_ptr->queue_start_addr_words + input_queue_ptr->get_queue_rptr_sent_offset_words()) * PACKET_WORD_SIZE_BYTES) :
                    input_queue_ptr->input_queue_curr_packet_num_words_available_to_send());

        if (is_remote_sender_to_stream) {
            return num_words_available_in_input;
        }

        // Streams can properly handle wraparound so we should disable this check for sender_to_stream_mode
        uint32_t num_words_before_input_rptr_wrap = input_queue_ptr->get_queue_words_before_rptr_sent_wrap();
        // DPRINT << "num_words_available_in_nput 1: " << num_words_available_in_input << "\n";
        num_words_available_in_input = std::min(num_words_available_in_input, num_words_before_input_rptr_wrap);
        // DPRINT << "num_words_before_input_rptr_wrap: " << num_words_before_input_rptr_wrap << "\n";
        uint32_t num_words_free_in_output = this->get_queue_data_num_words_free();
        uint32_t num_words_to_forward = is_remote_sender_to_stream ? num_words_available_in_input : std::min(num_words_available_in_input, num_words_free_in_output);

        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t output_buf_words_before_wptr_wrap = this->get_queue_words_before_wptr_wrap();
        num_words_to_forward = std::min(num_words_to_forward, output_buf_words_before_wptr_wrap);
        num_words_to_forward = std::min(num_words_to_forward, this->output_max_num_words_to_forward());

        return num_words_to_forward;
    }

    // Snijjar: grok -
    /*
     * Initiates a packet send from the input queue at `input_queue_index` to `this` (the output queue).
     * This is a *non-blocking* function, so on completion of the call, the transfer can be in any one of 3 states:
     *  - not started
     *  - in progress
     *  - completed
     *
     *
     */
    inline uint32_t forward_data_from_input(uint32_t input_queue_index, bool& full_packet_sent, uint32_t end_of_cmd) {
        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);
        bool input_is_stream = input_queue_ptr->remote_update_network_type == DispatchRemoteNetworkType::STREAM;
        bool output_is_stream = this->remote_update_network_type == DispatchRemoteNetworkType::STREAM;
        uint32_t num_words_to_forward = this->get_num_words_to_send(input_queue_index);
        full_packet_sent = (num_words_to_forward == input_queue_ptr->get_curr_packet_words_remaining());
        if (num_words_to_forward == 0) {
            return 0;
        }
        // if (output_is_stream) {
        //     DPRINT << "\tnum_words_to_forward: " << num_words_to_forward << "\n";
        // }

        if (this->unpacketizer_remove_header && input_queue_ptr->curr_packet_start()) {
            num_words_to_forward--;
            this->input_queue_status.register_words_in_flight(input_queue_index, 1);
        }
        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t src_addr =
            (input_queue_ptr->queue_start_addr_words +
             input_queue_ptr->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES;
        uint32_t dest_addr =
            (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;

        if (output_is_stream && num_words_to_forward > 0) {
            full_packet_sent = true;
        //     DPRINT << "input_queue_index: " << DEC() << input_queue_index << ENDL();
        //     DPRINT << "src_addr: " << HEX() << src_addr << ENDL();
            DPRINT << "num_words_to_forward: " << DEC() << num_words_to_forward << ENDL();
        //     DPRINT << "dest_addr: " << HEX() << dest_addr << ENDL();
        }

        // Does the actual send
        this->send_data_to_remote(
            input_queue_ptr->queue_start_addr_words*PACKET_WORD_SIZE_BYTES,
            input_queue_ptr->queue_size_words*PACKET_WORD_SIZE_BYTES,
            src_addr, dest_addr, num_words_to_forward);

        this->input_queue_status.register_words_in_flight(input_queue_index, num_words_to_forward);
        this->advance_queue_local_wptr(num_words_to_forward);

        if (!this->is_unpacketizer_output()) {
            this->remote_wptr_update(num_words_to_forward);
        } else {
            ASSERT(this->remote_update_network_type != DispatchRemoteNetworkType::STREAM);
            this->unpacketizer_page_words_sent += num_words_to_forward;
            // packet header size is in bytes and includes the header contribution to send size
            // for now need to size relay stream buffer to 64k since that's max packet size
            //  -> max prefetcher command size
            // only set by dispatcher
            if (full_packet_sent && end_of_cmd) {
                uint32_t unpacketizer_page_words_sent_past_page_bound =
                    this->unpacketizer_page_words_sent & (this->cb_mode_page_size_words - 1);
                if (unpacketizer_page_words_sent_past_page_bound > 0) {
                    uint32_t pad_words = this->cb_mode_page_size_words - unpacketizer_page_words_sent_past_page_bound;
                    this->unpacketizer_page_words_sent += pad_words;
                    this->advance_queue_local_wptr(pad_words);
                }
            }
            uint32_t remote_sem_inc = 0; // page size semaphore... page => 4k
            while (this->unpacketizer_page_words_sent >= this->cb_mode_page_size_words) {
                this->unpacketizer_page_words_sent -= this->cb_mode_page_size_words;
                remote_sem_inc++;
            }
            this->cb_mode_inc_remote_sem_val(remote_sem_inc);
        }

        return num_words_to_forward;
    }

    void dprint_object() {
        DPRINT << "Output queue:" << ENDL();
        packet_queue_state_t::dprint_object();
        this->input_queue_status.dprint_object();
    }
};


/**
 *  Polling for ready signal from the remote peers of all input and output queues.
 *  Blocks until all are ready, but doesn't block polling on each individual queue.
 *  Returns false in case of timeout.
 */
bool wait_all_src_dest_ready(packet_input_queue_state_t* input_queue_array, uint32_t num_input_queues,
                             packet_output_queue_state_t* output_queue_array, uint32_t num_output_queues,
                             uint32_t timeout_cycles = 0) {

    bool all_src_dest_ready = false;
    bool src_ready[MAX_SWITCH_FAN_IN] = {false};
    for (std::size_t i = 0; i < MAX_SWITCH_FAN_IN; i++) {
        // For stream remote endpoints, the handshake is done by the stream hardware
        // instead so we don't explicitly check for readiness here
        src_ready[i] = input_queue_array[i].remote_update_network_type == DispatchRemoteNetworkType::STREAM;
    }
    bool dest_ready[MAX_SWITCH_FAN_OUT] = {false};
    for (std::size_t i = 0; i < MAX_SWITCH_FAN_OUT; i++) {
        // For stream remote endpoints, the handshake is done by the stream hardware
        // instead so we don't explicitly check for readiness here
        dest_ready[i] = output_queue_array[i].remote_update_network_type == DispatchRemoteNetworkType::STREAM;
    }

    uint32_t iters = 0;

    uint32_t start_timestamp = get_timestamp_32b();
    while (!all_src_dest_ready) {
        iters++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_start = get_timestamp_32b() - start_timestamp;
            if (cycles_since_start > timeout_cycles) {
                return false;
            }
        }
        all_src_dest_ready = true;
        for (uint32_t i = 0; i < num_input_queues; i++) {
            if (!src_ready[i]) {
                src_ready[i] = input_queue_array[i].is_packetizer_input() ||
                               input_queue_array[i].is_remote_ready();
                if (!src_ready[i]) {
                    input_queue_array[i].send_remote_ready_notification();
                    all_src_dest_ready = false;
                } else {
                    // handshake with src complete
                }
            }
        }
        for (uint32_t i = 0; i < num_output_queues; i++) {
            if (!dest_ready[i]) {
                dest_ready[i] = output_queue_array[i].is_remote_ready() ||
                                output_queue_array[i].is_unpacketizer_output();
                if (dest_ready[i]) {
                    output_queue_array[i].send_remote_ready_notification();
                } else {
                    all_src_dest_ready = false;
                }
            }
        }
#if defined(COMPILE_FOR_ERISC)
        if ((timeout_cycles == 0) && (iters & 0xFFF) == 0) {
            //if timeout is disabled, context switch every 4096 iterations.
            //this is necessary to allow ethernet routing layer to operate.
            //this core may have pending ethernet routing work.
            internal_::risc_context_switch();
        }
#endif
    }
    return true;
}
