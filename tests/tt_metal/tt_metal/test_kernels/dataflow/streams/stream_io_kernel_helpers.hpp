// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <tuple>

#include "dataflow_api.h"
#include "stream_interface.h"
#include "tt_metal/hw/inc/wormhole/noc/noc_overlay_parameters.h"


// THESE TWO FUNCTIONS WERE ONLY VALID FOR WORMHOLE_B0 AND MAY NOT WORK WITH BLACKHOLE!!!
// STREAM_RECEIVER_ENDPOINT_MULTI_TILE_CLEAR_REG_INDEX is aliased to STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX for
// whb0
inline bool is_stream_receiver_endpoint_tile_clearing_finished(uint32_t stream_id) {
    return (NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX) == 0);
}
inline void stream_receiver_endpoint_tiles_clear_b0(uint32_t stream_id, uint32_t num_tiles) {
    uint32_t clr_val = num_tiles;
    clr_val *= 2;
    clr_val = (~clr_val) + 1;
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX, clr_val);
}
//////////////////////////////////////////////////////////////////////////////////////////

struct phase_iterator_t {
    phase_iterator_t()=default;
    phase_iterator_t(uint32_t start_phase, uint32_t max_phase) :
        phase_id(start_phase), max_phase(max_phase), start_phase(start_phase) {}
    uint32_t phase_id;
    uint32_t max_phase;
    uint32_t start_phase;

    // FORCE_INLINE
    uint32_t get() const { return phase_id; }

    // FORCE_INLINE
    void increment() { phase_id = phase_id == max_phase ? start_phase : phase_id + 1; }
};

struct stream_state_t {
    const uint32_t local_data_buffer_base_address;
    const uint32_t local_msg_info_ptr_base_address;

    uint32_t local_stream_id;
    uint32_t remote_stream_id;

    const uint32_t local_start_phase_id;
    uint32_t local_phase_id;
    uint32_t messages_per_phase;
    uint32_t msg_info_wrptr_addr;

    uint32_t num_tiles_sent;
    uint32_t tile_header_num_msgs;

    uint32_t local_buffer_base_addr;
    uint32_t local_buffer_size;
    uint32_t local_msg_info_ptr;
    uint32_t local_buffer_read_offset;

    uint32_t remote_buffer_base_addr;
    uint32_t remote_buffer_size;
    uint32_t remote_msg_info_ptr;
    uint32_t remote_buffer_write_offset;

    uint32_t remote_phase_id;

    // FORCE_INLINE
    uint32_t get_current_local_buffer_address() const {
        return local_data_buffer_base_address + local_buffer_read_offset;
    }
};


struct stream_remote_sender_kernel_args_t {
    // FORCE_INLINE
    int init_from_rt_args(uint32_t arg_idx) {
        // this->num_messages_to_forward = get_arg_val<uint32_t>(arg_idx++);
        this->local_stream_id = get_arg_val<uint32_t>(arg_idx++);
        this->local_stream_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
        this->messages_per_phase = get_arg_val<uint32_t>(arg_idx++);
        this->remote_dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
        this->remote_dest_noc_y = get_arg_val<uint32_t>(arg_idx++);
        this->remote_dest_noc_stream_id = get_arg_val<uint32_t>(arg_idx++);
        this->remote_dest_noc_id = get_arg_val<uint32_t>(arg_idx++);
        this->remote_buffer_base_addr = get_arg_val<uint32_t>(arg_idx++);
        this->remote_buffer_size_4B_words = get_arg_val<uint32_t>(arg_idx++);
        this->remote_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
        this->relay_done_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
        this->other_relay_core_to_signal_x = get_arg_val<uint32_t>(arg_idx++);
        this->other_relay_core_to_signal_y = get_arg_val<uint32_t>(arg_idx++);
        this->other_relay_done_semaphore = get_arg_val<uint32_t>(arg_idx++);
        this->wait_receiver_semaphore = get_arg_val<uint32_t>(arg_idx++);
        *reinterpret_cast<volatile uint32_t *>(wait_receiver_semaphore) = 0;
        this->first_relay_remote_src_start_phase_addr = get_arg_val<uint32_t>(arg_idx++);

        return arg_idx;
    }

    // uint32_t num_messages_to_forward;
    uint32_t local_stream_id;
    uint32_t local_stream_tile_header_buffer_addr;
    uint32_t messages_per_phase;
    uint32_t remote_dest_noc_x;
    uint32_t remote_dest_noc_y;
    uint32_t remote_dest_noc_stream_id;
    uint32_t remote_dest_noc_id;
    uint32_t remote_buffer_base_addr;
    uint32_t remote_buffer_size_4B_words;
    uint32_t remote_tile_header_buffer_addr;
    uint32_t relay_done_semaphore_addr;
    uint32_t other_relay_core_to_signal_x;
    uint32_t other_relay_core_to_signal_y;
    uint32_t other_relay_done_semaphore;
    uint32_t wait_receiver_semaphore;
    uint32_t first_relay_remote_src_start_phase_addr;
};

struct fabric_sender_stream_state_t {
    // All possible helpers should be hidden behind this struct because it will
    // simplify the enablement of optimization
    uint32_t local_msg_info_ptr_base_address;
    uint32_t local_stream_id;
    uint32_t remote_stream_id;
    uint32_t local_start_phase_id;
    uint32_t local_phase_id;
    uint32_t messages_per_phase;
    uint32_t msg_info_wrptr_addr;
    uint32_t num_tiles_sent;
    uint32_t tile_header_num_msgs;
    uint32_t local_msg_info_ptr;
    uint32_t remote_buffer_base_addr;
    uint32_t remote_buffer_size;
    uint32_t remote_msg_info_ptr;
    uint32_t remote_buffer_write_offset;
    uint32_t remote_phase_id;
    uint32_t data_noc_id;

    fabric_sender_stream_state_t() {
        // local_msg_info_ptr_base_address = std::numeric_limits<uint32_t>::max();
        // local_stream_id = std::numeric_limits<uint32_t>::max();
        // remote_stream_id = std::numeric_limits<uint32_t>::max();
        // local_start_phase_id = std::numeric_limits<uint32_t>::max();
        // local_phase_id = std::numeric_limits<uint32_t>::max();
        // messages_per_phase = std::numeric_limits<uint32_t>::max();
        // msg_info_wrptr_addr = std::numeric_limits<uint32_t>::max();
        // num_tiles_sent = std::numeric_limits<uint32_t>::max();
        // tile_header_num_msgs = std::numeric_limits<uint32_t>::max();
        // local_msg_info_ptr = std::numeric_limits<uint32_t>::max();
        // remote_buffer_base_addr = std::numeric_limits<uint32_t>::max();
        // remote_buffer_size = std::numeric_limits<uint32_t>::max();
        // remote_msg_info_ptr = std::numeric_limits<uint32_t>::max();
        // remote_buffer_write_offset = std::numeric_limits<uint32_t>::max();
        // remote_phase_id = std::numeric_limits<uint32_t>::max();
    }

    // bool is_initialized() const {
    //     return local_msg_info_ptr_base_address != std::numeric_limits<uint32_t>::max() &&
    //            local_stream_id != std::numeric_limits<uint32_t>::max() &&
    //            remote_stream_id != std::numeric_limits<uint32_t>::max() &&
    //            local_start_phase_id != std::numeric_limits<uint32_t>::max() &&
    //            local_phase_id != std::numeric_limits<uint32_t>::max() &&
    //            messages_per_phase != std::numeric_limits<uint32_t>::max() &&
    //            msg_info_wrptr_addr != std::numeric_limits<uint32_t>::max() &&
    //            num_tiles_sent != std::numeric_limits<uint32_t>::max() &&
    //            tile_header_num_msgs != std::numeric_limits<uint32_t>::max() &&
    //            local_msg_info_ptr != std::numeric_limits<uint32_t>::max() &&
    //            remote_buffer_base_addr != std::numeric_limits<uint32_t>::max() &&
    //            remote_buffer_size != std::numeric_limits<uint32_t>::max() &&
    //            remote_msg_info_ptr != std::numeric_limits<uint32_t>::max() &&
    //            remote_buffer_write_offset != std::numeric_limits<uint32_t>::max() &&
    //            remote_phase_id != std::numeric_limits<uint32_t>::max();
    // }

    // FORCE_INLINE
    void init_from_runtime_args(stream_remote_sender_kernel_args_t const& args, uint32_t local_starting_phase_id) {
        this->local_msg_info_ptr_base_address = args.local_stream_tile_header_buffer_addr;
        this->local_stream_id = args.local_stream_id;
        this->remote_stream_id = args.remote_dest_noc_stream_id;
        this->local_start_phase_id = local_starting_phase_id;
        this->local_phase_id = local_starting_phase_id;
        this->messages_per_phase = args.messages_per_phase;
        this->msg_info_wrptr_addr = args.local_stream_tile_header_buffer_addr;
        this->num_tiles_sent = 0;
        this->tile_header_num_msgs = args.messages_per_phase;
        this->local_msg_info_ptr = args.local_stream_tile_header_buffer_addr;
        this->remote_buffer_base_addr = args.remote_buffer_base_addr;
        this->remote_buffer_size = args.remote_buffer_size_4B_words;
        this->remote_msg_info_ptr = args.remote_tile_header_buffer_addr;
        this->remote_buffer_write_offset = 0;
        // Doesn't need to match the real consumer phase. Just for internal book-keeping
        this->remote_phase_id = 1;
        this->data_noc_id = args.remote_dest_noc_id;
    }

    // uint32_t get_current_local_buffer_address() const {
    //     return local_data_buffer_base_address + local_buffer_read_offset;
    // }
};

// struct noc_endpoint_info_t {
//     uint32_t data_noc_id;
//     uint32_t update_noc_id;
//     uint32_t noc_x;
//     uint32_t noc_y;
// };





struct noc_endpoint_info_t {
    uint32_t data_noc_id;
    uint32_t update_noc_id;
    uint32_t noc_x;
    uint32_t noc_y;
};

#define STREAM_CFG(field, val) ((val) << (field))

#define AUTO_CFG_HEADER(next_phase_num_cfg_reg_writes, curr_phase_num_msgs, phase_num_incr) \
    ((uint32_t)(((next_phase_num_cfg_reg_writes) << 24) | ((curr_phase_num_msgs) << 12) | (phase_num_incr)))

#define STREAM_REMOTE_DEST(dest_x, dest_y, dest_stream_id)                     \
    (((dest_x) << STREAM_REMOTE_DEST_X) | ((dest_y) << STREAM_REMOTE_DEST_Y) | \
     ((dest_stream_id) << STREAM_REMOTE_DEST_STREAM_ID))

#define STREAM_REMOTE_SRC(src_x, src_y, src_stream_id) \
    (((src_x) << STREAM_REMOTE_SRC_X) | ((src_y) << STREAM_REMOTE_SRC_Y) | ((src_stream_id) << REMOTE_SRC_STREAM_ID))

// FORCE_INLINE
uint32_t
blob_header_dw(uint32_t next_phase_num_cfg_reg_writes, uint32_t curr_phase_num_msgs, uint32_t phase_num_incr) {
    return (next_phase_num_cfg_reg_writes << 24) | (curr_phase_num_msgs << 12) | phase_num_incr;
}

// FORCE_INLINE
void stream_phase_blob_run(
    uint32_t stream_id, volatile uint32_t *blob_start_addr, uint32_t start_phase_num_cfg_regs) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, reinterpret_cast<uint32_t>(blob_start_addr));
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, start_phase_num_cfg_regs << NEXT_PHASE_NUM_CFG_REG_WRITES);
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_MISC_CFG_REG_INDEX,
        (0x1 << PHASE_AUTO_CONFIG) | (1 << NEXT_PHASE_SRC_CHANGE) | (1 << NEXT_PHASE_DEST_CHANGE));
}
// FORCE_INLINE
void stream_phase_blob_run(
    uint32_t stream_id,
    volatile uint32_t *blob_start_addr,
    uint32_t num_messages_per_phase,
    uint32_t start_phase_num_cfg_regs) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, reinterpret_cast<uint32_t>(blob_start_addr));

    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX,
        blob_header_dw(start_phase_num_cfg_regs, num_messages_per_phase, 1));
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_MISC_CFG_REG_INDEX,
        (0x1 << PHASE_AUTO_ADVANCE) | (0x1 << PHASE_AUTO_CONFIG) | (1 << NEXT_PHASE_SRC_CHANGE) |
            (1 << NEXT_PHASE_DEST_CHANGE));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 1);
}

// FORCE_INLINE
uint32_t blob_cfg_dw(uint32_t reg_index, uint32_t reg_val) { return (reg_val << 8) | reg_index; }

// FORCE_INLINE
uint32_t set_blob_reg_field(uint32_t blob_dw, uint32_t field_width, uint32_t field_offset, uint32_t val) {
    uint32_t mask = ((1 << field_width) - 1) << field_offset;
    return (blob_dw & ~mask) | ((val << field_offset) & mask);
}

// FORCE_INLINE
uint32_t get_first_available_phase_out_of_reset(uint32_t stream_id) {
    uint32_t stream_phase_coming_out_of_reset = stream_get_curr_phase(stream_id);
    return (
        stream_phase_coming_out_of_reset < 4096   ? 4096 : 1);
}

// FORCE_INLINE
uint32_t notify_remote_receiver_of_starting_phase(
    uint32_t stream_id, uint32_t local_buffer_addr, uint64_t remote_receiver_noc_addr) {
    uint32_t starting_phase = get_first_available_phase_out_of_reset(stream_id);
    ASSERT(starting_phase > 0);
    *reinterpret_cast<volatile uint32_t *>(local_buffer_addr) = starting_phase;
    noc_async_write(local_buffer_addr, remote_receiver_noc_addr, sizeof(uint32_t));
    // noc_semaphore_set_remote(local_buffer_addr, remote_receiver_noc_addr);
    noc_async_writes_flushed();
    return starting_phase;
}

// FORCE_INLINE
uint32_t wait_for_remote_source_starting_phase(volatile uint32_t *addr) {
    while (*addr == 0) {
        asm volatile("nop");
    }
    return *addr;
}

////////////////////////////////////////////////
///  Remote Sender Helpers
////////////////////////////////////////////////
// FORCE_INLINE
uint32_t get_sender_stream_config_reg(uint32_t tx_noc_id, uint32_t rx_src_update_noc, bool drain_after_phase_send) {
    uint32_t stream_cfg_reg = 0;
    bool next_phase_src_dest_change = drain_after_phase_send ? 1 : 0;
    stream_cfg_reg |= STREAM_CFG(OUTGOING_DATA_NOC, tx_noc_id) | STREAM_CFG(REMOTE_SRC_UPDATE_NOC, rx_src_update_noc) |
                      STREAM_CFG(SOURCE_ENDPOINT, 1) | STREAM_CFG(REMOTE_RECEIVER, 1) |
                      STREAM_CFG(NEXT_PHASE_SRC_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(NEXT_PHASE_DEST_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(PHASE_AUTO_ADVANCE, 0) | STREAM_CFG(DATA_AUTO_SEND, 0) |
                      STREAM_CFG(REG_UPDATE_VC_REG, 1);

    return stream_cfg_reg;
}


// FORCE_INLINE
void write_message_size_to_message_info_buffer(uint32_t local_msg_info_base_addr, uint32_t message_size_noc_words) {
    *reinterpret_cast<volatile uint32_t *>(local_msg_info_base_addr) = message_size_noc_words;
}

// FORCE_INLINE
void reset_stream_message_info_buffer_rdptr(stream_state_t &stream_state, uint32_t stream_id) {
    stream_state.local_msg_info_ptr = stream_state.local_msg_info_ptr_base_address;
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MSG_INFO_PTR_REG_INDEX, ((uint32_t)(stream_state.local_msg_info_ptr_base_address >> 4)));
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, (((uint32_t)stream_state.local_msg_info_ptr_base_address >> 4)));
}

// FORCE_INLINE
void reset_stream_message_info_buffer_rdptr(fabric_sender_stream_state_t &stream_state, uint32_t stream_id) {
    stream_state.local_msg_info_ptr = stream_state.local_msg_info_ptr_base_address;
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MSG_INFO_PTR_REG_INDEX, ((uint32_t)(stream_state.local_msg_info_ptr_base_address >> 4)));
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, (((uint32_t)stream_state.local_msg_info_ptr_base_address >> 4)));
}

// FORCE_INLINE
void advance_stream_message_info_buffer_wrptr(
    stream_state_t &stream_state, uint32_t stream_id, uint32_t message_size) {
    stream_state.local_msg_info_ptr += (1 << 4);
    stream_state.local_buffer_read_offset += message_size;
    if (stream_state.local_buffer_read_offset >= stream_state.local_buffer_size) {
        stream_state.local_buffer_read_offset -= stream_state.local_buffer_size;
    }
}

// FORCE_INLINE
void wait_for_stream_write_complete(uint32_t sender_stream_id) {
    while (!stream_phase_advance_wait(sender_stream_id)) {
        asm volatile("nop");
    }
}

// FORCE_INLINE
void copy_from_cb_to_stream_buffer(
    stream_state_t &stream_state, uint32_t message_base, uint32_t message_size_noc_words) {
    ASSERT((message_size_noc_words << 4) <= stream_state.local_buffer_size);
    if (!((message_size_noc_words << 4) <= stream_state.local_buffer_size)) {
        DPRINT << "YIKES2\n";
    }
    uint32_t message_size_size_in_bytes = message_size_noc_words << 4;
    uint32_t bytes_to_copy =
        std::min(stream_state.local_buffer_size - stream_state.local_buffer_read_offset, message_size_size_in_bytes);
    noc_async_write(message_base, get_noc_addr(stream_state.get_current_local_buffer_address()), bytes_to_copy);
    ASSERT(stream_state.local_buffer_size + stream_state.local_buffer_read_offset >= bytes_to_copy);
    if (!(stream_state.local_buffer_size + stream_state.local_buffer_read_offset >= bytes_to_copy)) {
        DPRINT << "YIKES3\n";
    }

    if (bytes_to_copy < message_size_size_in_bytes) {
        uint32_t second_bytes_to_copy = message_size_size_in_bytes - bytes_to_copy;
        noc_async_write(
            message_base + bytes_to_copy, get_noc_addr(stream_state.local_buffer_base_addr), second_bytes_to_copy);
    }
    noc_async_write_barrier();
}

// This function is heavily coupled with the autonomous looping stream setup. It's *NOT*
// recommended to use this as a generic function for talking to streams unless they are
// setup in this specific looping configuration.
// FORCE_INLINE
void stream_noc_write(
    uint32_t src_addr,
    uint32_t dest_addr,
    uint32_t size_bytes,
    uint32_t remote_noc_x,
    uint32_t remote_noc_y,
    uint32_t dest_noc_id,
    stream_state_t &stream_state) {

    // This was taken from the autonomous stream test-bench which already correctly stores
    // the message size in 16B words, in the packet header. However, for packet_(mux|demux),
    // the message size is stored in bytes, so we need to override it to be in 16B words

    uint32_t message_size_noc_words = *reinterpret_cast<volatile uint32_t *>(src_addr);
    ASSERT(size_bytes == message_size_noc_words);
    // Convert message size header field from size bytes to size noc words
    message_size_noc_words = message_size_noc_words >> 4;
    *reinterpret_cast<volatile uint32_t *>(src_addr) = message_size_noc_words;


    uint32_t dest_noc_reg = 0;
    uint32_t num_tiles = stream_state.num_tiles_sent;
    const bool send_last_message_and_drain = num_tiles == (stream_state.tile_header_num_msgs - 1);

    bool first_message = num_tiles == 0;

    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_CURR_PHASE_BASE_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_CURR_PHASE_REG_INDEX, stream_state.local_phase_id);

    if (first_message) {
        reset_stream_message_info_buffer_rdptr(stream_state, stream_state.local_stream_id);
        stream_state.local_buffer_read_offset = 0;
    }
    copy_from_cb_to_stream_buffer(stream_state, src_addr, message_size_noc_words);
    uint32_t rx_src_update_noc = 1 - dest_noc_id;
    if (send_last_message_and_drain) {
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_MISC_CFG_REG_INDEX,
            get_sender_stream_config_reg(dest_noc_id, rx_src_update_noc, true));

    } else if (first_message) {

        uint32_t rx_src_update_noc = 1 - dest_noc_id;
        uint32_t translated_remote_noc_x = dest_noc_id == 0 ? remote_noc_x : noc_size_x - 1 - remote_noc_x;
        uint32_t translated_remote_noc_y = dest_noc_id == 0 ? remote_noc_y : noc_size_y - 1 - remote_noc_y;
        uint32_t dest_stream_id = stream_state.remote_stream_id;
        NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_BUF_START_REG_INDEX, src_addr >> 4);
        NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_BUF_SIZE_REG_INDEX, message_size_noc_words);

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_REG_INDEX,
            STREAM_REMOTE_DEST(translated_remote_noc_x, translated_remote_noc_y, dest_stream_id));
        NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_HI_REG_INDEX, 0);
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX, stream_state.remote_msg_info_ptr >> 4);

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX, stream_state.remote_buffer_base_addr >> 4);
        // Inserting an assert here causes test to pass
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_BUF_START_HI_REG_INDEX,
            (stream_state.remote_buffer_base_addr / MEM_WORD_WIDTH) >> MEM_WORD_ADDR_WIDTH);
        NOC_STREAM_WRITE_REG_FIELD(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX,
            REMOTE_DEST_BUF_SIZE_WORDS,
            stream_state.remote_buffer_size >> 4);

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_MISC_CFG_REG_INDEX,
            get_sender_stream_config_reg(dest_noc_id, rx_src_update_noc, false));
    }

    // Remove this if we want to remove the sync at the end of this function (and instead want to support syncing later)
    ASSERT(
        dest_addr == (NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX) +
                      NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_WR_PTR_REG_INDEX))
                         << 4);

    write_message_size_to_message_info_buffer(stream_state.local_msg_info_ptr, message_size_noc_words);
    advance_stream_message_info_buffer_wrptr(stream_state, stream_state.local_stream_id, message_size_noc_words << 4);


    NOC_STREAM_WRITE_REG(
        stream_state.local_stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, AUTO_CFG_HEADER(0, 1 /*tiles_per_phase*/, 1));
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 0x1);


    if (first_message) {
        // wait for handshake to complete
        while (!stream_phase_is_active(stream_state.local_stream_id)) {
            asm volatile("");
        }
    }

    if (send_last_message_and_drain) {
        // We only wrap around to 0 when the remote receiver relay stream has finished its second phase. We need to do
        // this to avoid any handshake bugs we might hit if the second phase of relay must sync with phase 1 of the
        // producer (this) since the relay will handshake with phase 1 of the producer (this) stream for relay stream's
        // first phase too
        num_tiles = 0;
        stream_state.remote_phase_id = 3 - stream_state.remote_phase_id;  // will alternate between 1 and 2
        // Remote phase was already updated so the condition is inverted
        stream_state.local_phase_id =
            (stream_state.remote_phase_id == 1) ? stream_state.local_start_phase_id : stream_state.local_phase_id + 1;
    } else {
        num_tiles++;
        stream_state.local_phase_id++;
    }

    stream_relay_tiles(stream_state.local_stream_id, 1, message_size_noc_words);
    wait_for_stream_write_complete(stream_state.local_stream_id);

    stream_state.num_tiles_sent = num_tiles;
}

/*// FORCE_INLINE*/ void stream_noc_write_from_mux(
    uint32_t src_addr,
    uint32_t dest_addr,
    uint32_t size_bytes,
    uint32_t remote_noc_x,
    uint32_t remote_noc_y,
    fabric_sender_stream_state_t &stream_state) {
    uint32_t dest_noc_id = stream_state.data_noc_id;

    // This was taken from the autonomous stream test-bench which already correctly stores
    // the message size in 16B words, in the packet header. However, for packet_(mux|demux),
    // the message size is stored in bytes, so we need to override it to be in 16B words
    uint32_t message_size_noc_words = *reinterpret_cast<volatile uint32_t *>(src_addr);
    if (!(size_bytes == message_size_noc_words)) {
        DPRINT << "src_addr: " << src_addr << "\n";
        DPRINT << "size_bytes = " << size_bytes << " message_size_noc_words = " << message_size_noc_words << "\n";
    }
    ASSERT(size_bytes == message_size_noc_words);
    message_size_noc_words = message_size_noc_words >> 4;
    *reinterpret_cast<volatile uint32_t *>(src_addr) = message_size_noc_words;

    uint32_t dest_noc_reg = 0;
    uint32_t num_tiles = stream_state.num_tiles_sent;
    const bool send_last_message_and_drain = num_tiles == (stream_state.tile_header_num_msgs - 1);

    bool first_message = num_tiles == 0;

    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_CURR_PHASE_BASE_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_CURR_PHASE_REG_INDEX, stream_state.local_phase_id);

    if (first_message) {
        DPRINT << "Stream Noc Write: reset_stream_message_info_buffer_rdptr \n";
        reset_stream_message_info_buffer_rdptr(stream_state, stream_state.local_stream_id);
    }
    // copy_from_cb_to_stream_buffer(stream_state, src_addr, message_size_noc_words);

    // For mux stream send, we must override the buffer start and size per message because we could be forwarding
    // from any of the mux input buffers
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_BUF_START_REG_INDEX, src_addr >> 4);
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_BUF_SIZE_REG_INDEX, message_size_noc_words);

    uint32_t rx_src_update_noc = 1 - dest_noc_id;
    if (send_last_message_and_drain) {
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_MISC_CFG_REG_INDEX,
            get_sender_stream_config_reg(dest_noc_id, rx_src_update_noc, true));

    } else if (first_message) {
        uint32_t rx_src_update_noc = 1 - dest_noc_id;
        uint32_t translated_remote_noc_x = dest_noc_id == 0 ? remote_noc_x : noc_size_x - 1 - remote_noc_x;
        uint32_t translated_remote_noc_y = dest_noc_id == 0 ? remote_noc_y : noc_size_y - 1 - remote_noc_y;
        uint32_t dest_stream_id = stream_state.remote_stream_id;

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_REG_INDEX,
            STREAM_REMOTE_DEST(translated_remote_noc_x, translated_remote_noc_y, dest_stream_id));
        NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_HI_REG_INDEX, 0);
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX, stream_state.remote_msg_info_ptr >> 4);

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX, stream_state.remote_buffer_base_addr >> 4);
        // Inserting an assert here causes test to pass
        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_BUF_START_HI_REG_INDEX,
            (stream_state.remote_buffer_base_addr / MEM_WORD_WIDTH) >> MEM_WORD_ADDR_WIDTH);
        NOC_STREAM_WRITE_REG_FIELD(
            stream_state.local_stream_id,
            STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX,
            REMOTE_DEST_BUF_SIZE_WORDS,
            stream_state.remote_buffer_size >> 4);

        NOC_STREAM_WRITE_REG(
            stream_state.local_stream_id,
            STREAM_MISC_CFG_REG_INDEX,
            get_sender_stream_config_reg(dest_noc_id, rx_src_update_noc, false));
    }

    // Remove this if we want to remove the sync at the end of this function (and instead want to support syncing later)
    ASSERT(
        dest_addr == (NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX) +
                      NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_REMOTE_DEST_WR_PTR_REG_INDEX))
                         << 4);

    write_message_size_to_message_info_buffer(stream_state.local_msg_info_ptr, message_size_noc_words);
    stream_state.local_msg_info_ptr += (1 << 4);
    // advance_stream_message_info_buffer_wrptr(stream_state, stream_state.local_stream_id, message_size_noc_words << 4);

    NOC_STREAM_WRITE_REG(
        stream_state.local_stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, AUTO_CFG_HEADER(0, 1 /*tiles_per_phase*/, 1));
    NOC_STREAM_WRITE_REG(stream_state.local_stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 0x1);

    if (first_message) {
        DPRINT << "Stream Noc Write: Handshaking...\n";
        // wait for handshake to complete
        while (!stream_phase_is_active(stream_state.local_stream_id)) {
            asm volatile("");
        }

        DPRINT << "Stream Noc Write: ... Done\n";
    }

    if (send_last_message_and_drain) {
        // We only wrap around to 0 when the remote receiver relay stream has finished its second phase. We need to do
        // this to avoid any handshake bugs we might hit if the second phase of relay must sync with phase 1 of the
        // producer (this) since the relay will handshake with phase 1 of the producer (this) stream for relay stream's
        // first phase too
        num_tiles = 0;
        stream_state.remote_phase_id = 3 - stream_state.remote_phase_id;  // will alternate between 1 and 2
        // Remote phase was already updated so the condition is inverted
        stream_state.local_phase_id =
            (stream_state.remote_phase_id == 1) ? stream_state.local_start_phase_id : stream_state.local_phase_id + 1;
    } else {
        num_tiles++;
        stream_state.local_phase_id++;
    }

    // DPRINT << "Stream Noc Write: Sending message\n";
    stream_relay_tiles(stream_state.local_stream_id, 1, message_size_noc_words);
    // DPRINT << "Stream Noc Write: Waiting for flush\n";
    wait_for_stream_write_complete(stream_state.local_stream_id);
    DPRINT << "Stream Noc Write: Message sent!\n";

    stream_state.num_tiles_sent = num_tiles;
}



/////////////////////////////////////////////////
///  REMOTE RECEIVER HELPERS
/////////////////////////////////////////////////


struct stream_remote_receiver_kernel_args_t {
    // FORCE_INLINE
    int init_from_rt_args(uint32_t arg_idx) {
        this->local_stream_id = get_arg_val<uint32_t>(arg_idx++);
        this->local_stream_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
        this->local_stream_buffer_size = get_arg_val<uint32_t>(arg_idx++);
        this->local_stream_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
        this->num_message_per_phase = get_arg_val<uint32_t>(arg_idx++);
        this->remote_src_noc_x = get_arg_val<uint32_t>(arg_idx++);
        this->remote_src_noc_y = get_arg_val<uint32_t>(arg_idx++);
        this->remote_src_noc_stream_id = get_arg_val<uint32_t>(arg_idx++);
        this->remote_src_data_noc_id = get_arg_val<uint32_t>(arg_idx++);
        this->relay_done_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
        this->other_relay_core_to_signal_x = get_arg_val<uint32_t>(arg_idx++);
        this->other_relay_core_to_signal_y = get_arg_val<uint32_t>(arg_idx++);
        this->other_relay_done_semaphore = get_arg_val<uint32_t>(arg_idx++);
        this->fw_sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
        this->fw_sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
        this->sender_wait_finish_semaphore = get_arg_val<uint32_t>(arg_idx++);
        this->remote_src_start_phase_addr = get_arg_val<uint32_t>(arg_idx++);

        this->remote_src_update_noc_id = 1 - this->remote_src_data_noc_id;

        return arg_idx;
    }

    bool is_initialized() const {
        return
            this->local_stream_id != std::numeric_limits<uint32_t>::max() &&
            this->local_stream_buffer_addr != std::numeric_limits<uint32_t>::max() &&
            this->local_stream_buffer_size != std::numeric_limits<uint32_t>::max() &&
            this->local_stream_tile_header_buffer_addr != std::numeric_limits<uint32_t>::max() &&
            this->num_message_per_phase != std::numeric_limits<uint32_t>::max() &&
            this->remote_src_noc_x != std::numeric_limits<uint32_t>::max() &&
            this->remote_src_noc_y != std::numeric_limits<uint32_t>::max() &&
            this->remote_src_noc_stream_id != std::numeric_limits<uint32_t>::max() &&
            this->remote_src_data_noc_id != std::numeric_limits<uint32_t>::max() &&
            this->relay_done_semaphore_addr != std::numeric_limits<uint32_t>::max() &&
            this->other_relay_core_to_signal_x != std::numeric_limits<uint32_t>::max() &&
            this->other_relay_core_to_signal_y != std::numeric_limits<uint32_t>::max() &&
            this->other_relay_done_semaphore != std::numeric_limits<uint32_t>::max() &&
            this->fw_sender_noc_x != std::numeric_limits<uint32_t>::max() &&
            this->fw_sender_noc_y != std::numeric_limits<uint32_t>::max() &&
            this->sender_wait_finish_semaphore != std::numeric_limits<uint32_t>::max() &&
            this->remote_src_start_phase_addr != std::numeric_limits<uint32_t>::max() &&
            this->remote_src_update_noc_id != std::numeric_limits<uint32_t>::max();
    }

    uint32_t local_stream_id = std::numeric_limits<uint32_t>::max();
    uint32_t local_stream_buffer_addr = std::numeric_limits<uint32_t>::max();
    uint32_t local_stream_buffer_size = std::numeric_limits<uint32_t>::max();
    uint32_t local_stream_tile_header_buffer_addr = std::numeric_limits<uint32_t>::max();
    uint32_t num_message_per_phase = std::numeric_limits<uint32_t>::max();
    uint32_t remote_src_noc_x = std::numeric_limits<uint32_t>::max();
    uint32_t remote_src_noc_y = std::numeric_limits<uint32_t>::max();
    uint32_t remote_src_noc_stream_id = std::numeric_limits<uint32_t>::max();
    uint32_t remote_src_data_noc_id = std::numeric_limits<uint32_t>::max();
    uint32_t relay_done_semaphore_addr = std::numeric_limits<uint32_t>::max();
    uint32_t other_relay_core_to_signal_x = std::numeric_limits<uint32_t>::max();
    uint32_t other_relay_core_to_signal_y = std::numeric_limits<uint32_t>::max();
    uint32_t other_relay_done_semaphore = std::numeric_limits<uint32_t>::max();
    // The remote src of the first tunneler stream
    uint32_t fw_sender_noc_x = std::numeric_limits<uint32_t>::max();
    uint32_t fw_sender_noc_y = std::numeric_limits<uint32_t>::max();
    uint32_t sender_wait_finish_semaphore = std::numeric_limits<uint32_t>::max();
    uint32_t remote_src_start_phase_addr = std::numeric_limits<uint32_t>::max();
    uint32_t remote_src_update_noc_id = std::numeric_limits<uint32_t>::max();
};
struct fabric_receiver_stream_state_t {
    fabric_receiver_stream_state_t() = default;
    uint32_t local_data_buffer_base_address;
    uint32_t local_msg_info_ptr_base_address;

    uint32_t local_stream_id;
    uint32_t remote_stream_id;

    uint32_t local_start_phase_id;
    uint32_t local_phase_id;
    uint32_t messages_per_phase;
    uint32_t msg_info_wrptr_addr;

    uint32_t num_tiles_sent;
    uint32_t tile_header_num_msgs;

    uint32_t local_buffer_base_addr;
    uint32_t local_buffer_size;
    uint32_t local_msg_info_ptr;
    uint32_t local_buffer_read_offset;

    uint32_t remote_phase_id;

    uint32_t remote_src_noc_x;
    uint32_t remote_src_noc_y;
    uint32_t remote_src_data_noc_id;
    uint32_t remote_src_update_noc_id;


    phase_iterator_t local_phase_iterator;
    phase_iterator_t remote_phase_iterator;

    // FORCE_INLINE
    uint32_t get_current_local_buffer_address() const {
        return local_data_buffer_base_address + local_buffer_read_offset;
    }

    // FORCE_INLINE
    void init_from_runtime_args(
        stream_remote_receiver_kernel_args_t const& args,
        phase_iterator_t const& local_phase_iterator,
        phase_iterator_t const& remote_phase_iterator) {
        this->local_data_buffer_base_address = args.local_stream_buffer_addr;
        this->local_msg_info_ptr_base_address = args.local_stream_tile_header_buffer_addr;
        this->local_stream_id = args.local_stream_id;
        this->remote_stream_id = args.remote_src_noc_stream_id;
        this->local_start_phase_id = local_phase_iterator.get();
        this->local_phase_id = this->local_start_phase_id;
        this->messages_per_phase = args.num_message_per_phase;
        this->msg_info_wrptr_addr = this->local_msg_info_ptr_base_address;
        this->num_tiles_sent = 0;
        this->tile_header_num_msgs = this->messages_per_phase;
        this->local_buffer_base_addr = args.local_stream_buffer_addr;
        this->local_buffer_size = args.local_stream_buffer_size;
        this->local_msg_info_ptr = this->local_msg_info_ptr_base_address;
        this->local_buffer_read_offset = 0;
        this->remote_phase_id = remote_phase_iterator.get();

        this->remote_src_noc_x = args.remote_src_noc_x;
        this->remote_src_noc_y = args.remote_src_noc_y;
        this->remote_src_data_noc_id = args.remote_src_data_noc_id;
        this->remote_src_update_noc_id = args.remote_src_update_noc_id;

        this->local_phase_iterator = local_phase_iterator;
        this->remote_phase_iterator = remote_phase_iterator;
    }
};

// Taken from `advance_stream_state_struct`
FORCE_INLINE void advance_remote_receiver_stream_state_struct(
    fabric_receiver_stream_state_t &stream_state, uint32_t msg_size_bytes) {
    uint32_t next_offset = stream_state.local_buffer_read_offset + msg_size_bytes;
    if (next_offset >= stream_state.local_buffer_size) {
        next_offset -= stream_state.local_buffer_size;
    }
    stream_state.local_buffer_read_offset = next_offset;
    stream_state.local_msg_info_ptr += (1 << 4);
}

uint32_t get_receiver_stream_config_reg_val_for_looping_mode(uint32_t data_noc_id, uint32_t update_noc, bool drain_after_phase_send) {
    uint32_t stream_cfg_reg = 0;
    bool next_phase_src_dest_change = drain_after_phase_send ? 1 : 0;
    stream_cfg_reg |= STREAM_CFG(INCOMING_DATA_NOC, data_noc_id) | STREAM_CFG(REMOTE_SRC_UPDATE_NOC, update_noc) |
                      STREAM_CFG(RECEIVER_ENDPOINT, 1) | STREAM_CFG(REMOTE_SOURCE, 1) |
                      STREAM_CFG(NEXT_PHASE_SRC_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(NEXT_PHASE_DEST_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(PHASE_AUTO_ADVANCE, 0) | STREAM_CFG(DATA_AUTO_SEND, 0) |
                      STREAM_CFG(REG_UPDATE_VC_REG, 1);

    return stream_cfg_reg;
}


FORCE_INLINE void advance_remote_receiver_phase(
    // noc_endpoint_info_t const &remote_endpoint_info,
    fabric_receiver_stream_state_t &state//,
    // uint32_t stream_id
    ) {
    // This is remote receiver, so it sends messages (updates) to remote source, NOT data, so it uses
    // the update noc to communicate to remote src instead of the data noc. Therefore, we need to set remote
    // src x/y based on the update noc.
    uint32_t translated_remote_noc_x = state.remote_src_update_noc_id == 0
                                           ? state.remote_src_noc_x
                                           : noc_size_x - 1 - state.remote_src_noc_x;
    uint32_t translated_remote_noc_y = state.remote_src_update_noc_id == 0
                                           ? state.remote_src_noc_y
                                           : noc_size_y - 1 - state.remote_src_noc_y;

    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_CURR_PHASE_BASE_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_CURR_PHASE_REG_INDEX, ((uint32_t)state.local_phase_id));
    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_BUF_START_REG_INDEX, ((uint32_t)state.local_buffer_base_addr) >> 4);
    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_BUF_SIZE_REG_INDEX, state.local_buffer_size >> 4);
    NOC_STREAM_WRITE_REG(
        state.local_stream_id,
        STREAM_REMOTE_SRC_REG_INDEX,
        STREAM_REMOTE_SRC(translated_remote_noc_x, translated_remote_noc_y, state.local_stream_id));
    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX, ((uint32_t)state.remote_phase_id));

    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_MSG_INFO_PTR_REG_INDEX, ((uint32_t)state.local_msg_info_ptr) >> 4);
    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, ((uint32_t)state.local_msg_info_ptr) >> 4);

    NOC_STREAM_WRITE_REG(
        state.local_stream_id,
        STREAM_MISC_CFG_REG_INDEX,
        get_receiver_stream_config_reg_val_for_looping_mode(state.remote_src_data_noc_id, state.remote_src_update_noc_id, true));

    NOC_STREAM_WRITE_REG(
        state.local_stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, AUTO_CFG_HEADER(0, state.messages_per_phase, 0));
    NOC_STREAM_WRITE_REG(state.local_stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 0x1);
}


// Taken from `flush_message_from_stream_buffer`
FORCE_INLINE void flush_message_from_remote_receiver_stream_buffer(
    fabric_receiver_stream_state_t &stream_state, uint32_t msg_size_bytes) {
    stream_receiver_endpoint_tiles_clear_b0(stream_state.local_stream_id, 1);
    while (!is_stream_receiver_endpoint_tile_clearing_finished(stream_state.local_stream_id)) {
        asm volatile("");
    }
}

// Taken from `advance_stream_to_next_message`
FORCE_INLINE void advance_remote_receiver_stream_to_next_message(
    // noc_endpoint_info_t const &remote_endpoint_info,
    fabric_receiver_stream_state_t &state,
    // uint32_t stream_id,
    uint32_t msg_size_bytes) {
    advance_remote_receiver_stream_state_struct(state, msg_size_bytes);
    flush_message_from_remote_receiver_stream_buffer(state, msg_size_bytes);

    if (state.num_tiles_sent == state.tile_header_num_msgs - 1) {
        state.remote_phase_iterator.increment();
        state.remote_phase_id = state.remote_phase_iterator.get();
        state.local_phase_iterator.increment();
        state.local_phase_id = state.local_phase_iterator.get();
        state.num_tiles_sent = 0;
        state.local_msg_info_ptr = state.local_msg_info_ptr_base_address;

        advance_remote_receiver_phase(state);
        state.local_buffer_read_offset = 0;
    } else {
        state.num_tiles_sent++;
    }
}

FORCE_INLINE uint32_t get_next_available_stream_message_size_in_bytes(fabric_receiver_stream_state_t &stream_state) {
    uint32_t msg_info_byte_ptr = stream_state.local_msg_info_ptr;
    uint32_t msg_size_bytes = *reinterpret_cast<volatile uint32_t *>(msg_info_byte_ptr) << 4;
    ASSERT(msg_size_bytes > 0);
    return msg_size_bytes;
}


FORCE_INLINE bool messages_are_available(fabric_receiver_stream_state_t &stream_state) {
    uint32_t wrptr = NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX);
    uint32_t rdptr = NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_MSG_INFO_PTR_REG_INDEX);
    uint32_t internal_rdptr = stream_state.local_msg_info_ptr >> 4;
    bool messages_available = internal_rdptr < wrptr;
    return messages_available;
}

FORCE_INLINE std::tuple<uint32_t, uint32_t> get_next_message_info(fabric_receiver_stream_state_t &stream_state) {
    uint32_t rdptr_offset = NOC_STREAM_READ_REG(stream_state.local_stream_id, STREAM_RD_PTR_REG_INDEX) << 4;
    uint32_t addr = rdptr_offset + stream_state.local_data_buffer_base_address;
    ASSERT((rdptr_offset & 0xF) == 0);
    ASSERT((addr & 0xF) == 0);
    return std::make_tuple(addr, get_next_available_stream_message_size_in_bytes(stream_state));
}

// FORCE_INLINE
bool fw_managed_rx_stream_num_bytes_available_impl(uint32_t stream_id, uint32_t local_msg_info_ptr) {
    uint32_t wrptr = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX);
    uint32_t n_16B_words_available = 0;
    for (uint32_t internal_rdptr = local_msg_info_ptr >> 4; internal_rdptr < wrptr; internal_rdptr++) {
        volatile uint32_t *msg_hdr_ptr = reinterpret_cast<volatile uint32_t*>(internal_rdptr << 4);
        n_16B_words_available += *msg_hdr_ptr;
    }

    return n_16B_words_available << 4;
}

// FORCE_INLINE
bool fw_managed_rx_stream_num_bytes_available(uint32_t stream_id, stream_state_t const& stream_state) {
    return fw_managed_rx_stream_num_bytes_available_impl(stream_id, stream_state.local_msg_info_ptr);
}

// FORCE_INLINE
bool fw_managed_rx_stream_num_bytes_available(uint32_t stream_id, fabric_receiver_stream_state_t const& stream_state) {
    return fw_managed_rx_stream_num_bytes_available_impl(stream_id, stream_state.local_msg_info_ptr);
}
