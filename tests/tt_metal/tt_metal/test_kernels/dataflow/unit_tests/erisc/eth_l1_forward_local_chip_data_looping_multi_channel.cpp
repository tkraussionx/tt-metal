// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

#include "dataflow_api.h"
// #include "debug_dprint.h"

#define ENABLE_L1_BUFFER_OVERLAP 0
// #define ENABLE_L1_BUFFER_OVERLAP 1
// #define EMULATE_DRAM_READ_CYCLES 1
#define EMULATE_DRAM_READ_CYCLES 0
// #define DONT_STRIDE_IN_ETH_BUFFER 1
#define DONT_STRIDE_IN_ETH_BUFFER 0

void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();

        // eth_wait_for_bytes(16);
        // eth_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_done();

        // eth_send_bytes(handshake_register_address,handshake_register_address, 16);
        // eth_wait_for_receiver_done();
    }
}

template <bool src_is_dram>
void read_chunk(
    uint32_t eth_l1_buffer_address_base,
    uint32_t num_pages,
    uint32_t num_pages_per_l1_buffer,
    uint32_t page_size,
    uint32_t &page_index,
    const InterleavedAddrGen<src_is_dram> &source_address_generator
)
{
    uint32_t local_eth_l1_curr_src_addr = eth_l1_buffer_address_base;
    uint32_t end_page_index = std::min(page_index + num_pages_per_l1_buffer, num_pages);
    for (; page_index < end_page_index; ++page_index) {
        // read source address
        uint64_t src_noc_addr = get_noc_addr(page_index, source_address_generator);
        noc_async_read(src_noc_addr, local_eth_l1_curr_src_addr, page_size);
        // read dest addr
        #if DONT_STRIDE_IN_ETH_BUFFER == 0
        local_eth_l1_curr_src_addr += page_size;
        #endif
    }
}

#if EMULATE_DRAM_READ_CYCLES == 1
static uint32_t timestamp_H = 0;
static uint32_t timestamp_L = 0;
static uint32_t CYCLES_PER_DRAM_READ = 4000;

FORCE_INLINE void get_timestamp() {
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
    timestamp_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    timestamp_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
    timestamp_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    timestamp_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif
}

FORCE_INLINE uint32_t cycles_since_last_timestamp() {
#if defined(COMPILE_FOR_NCRISC) | defined(COMPILE_FOR_BRISC) | defined(COMPILE_FOR_ERISC)
    uint32_t t_L = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t t_H = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#else
    uint32_t t_L = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t t_H = ckernel::reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
#endif
    // Assume t_H didn't increment twice
    return t_H == timestamp_H ? t_L - timestamp_L : ((std::numeric_limits<uint32_t>::max() - timestamp_L) + t_L);
}

FORCE_INLINE void issue_read_chunk() { get_timestamp(); }

FORCE_INLINE bool emulated_dram_read_cycles_finished() { return cycles_since_last_timestamp() > CYCLES_PER_DRAM_READ; }

#endif


template <uint8_t MAX_NUM_CHANNELS>
FORCE_INLINE  bool buffer_pool_full(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    const QueueIndexPointer<uint8_t> eth_sender_rdptr,
    const QueueIndexPointer<uint8_t> eth_sender_ackptr) {
    return QueueIndexPointer<uint8_t>::full(noc_reader_buffer_wrptr, eth_sender_ackptr);
}

FORCE_INLINE  bool is_noc_read_in_progress(const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr, const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr) {
    return noc_reader_buffer_wrptr != noc_reader_buffer_ackptr;
}

FORCE_INLINE  bool buffer_pool_empty(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    const QueueIndexPointer<uint8_t> eth_sender_rdptr,
    const QueueIndexPointer<uint8_t> eth_sender_ackptr) {
    return QueueIndexPointer<uint8_t>::empty(eth_sender_rdptr, noc_reader_buffer_wrptr);
}

FORCE_INLINE  bool buffer_available_for_eth_send(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    const QueueIndexPointer<uint8_t> eth_sender_rdptr,
    const QueueIndexPointer<uint8_t> eth_sender_ackptr) {
    return eth_sender_rdptr != noc_reader_buffer_ackptr;
}

// 2 main main concurrent sequences of execution (but time multiplexed on the erisc)
//
// 1. The local chip data reader sequence
//    Reads data from the local chip (L1 or DRAM, doesn't really matter) into a
//    local L1 buffer
// 2. Ethernet data forwarding sequence
//    when data is available in one of the buffers in the queue, it will send
//    it over ethernet, assuming the receiver is able to accept it
//    -> Receiver may not have acknowledged the previous send to this buffer
//       so we'd have to wait in that case

template <int MAX_NUM_CHANNELS>
FORCE_INLINE  bool eth_send_data_sequence(
    std::array<uint32_t, MAX_NUM_CHANNELS> &transaction_channel_sender_buffer_addresses,
    std::array<uint32_t, MAX_NUM_CHANNELS> &transaction_channel_receiver_buffer_addresses,
    uint32_t local_eth_l1_src_addr,
    uint32_t remote_eth_l1_dst_addr,
    uint32_t num_bytes,
    uint32_t num_bytes_per_send,
    uint32_t num_bytes_per_send_word_size,
    QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    QueueIndexPointer<uint8_t> &eth_sender_rdptr,
    QueueIndexPointer<uint8_t> &eth_sender_ackptr) {
    bool did_something = false;
    bool data_ready_for_send = buffer_available_for_eth_send(
        noc_reader_buffer_wrptr, noc_reader_buffer_ackptr, eth_sender_rdptr, eth_sender_ackptr);
    if (data_ready_for_send) {
        bool consumer_ready_to_accept = eth_is_receiver_channel_send_done(eth_sender_rdptr.index());
        if (consumer_ready_to_accept) {
            // kernel_profiler::mark_time(14);
            // Queue up another send
            uint32_t sender_buffer_address = transaction_channel_sender_buffer_addresses[eth_sender_rdptr.index()];
            uint32_t receiver_buffer_address = transaction_channel_receiver_buffer_addresses[eth_sender_rdptr.index()];

            // // DPRINT << "tx: sending data on channel " << (uint32_t)eth_sender_rdptr << "\n";
            eth_send_bytes_over_channel(
                sender_buffer_address,
                receiver_buffer_address,
                num_bytes,
                eth_sender_rdptr.index(),
                num_bytes_per_send,
                num_bytes_per_send_word_size);
            eth_sender_rdptr.increment();
            did_something = true;
        }
    }

    return did_something;
}

template <uint8_t MAX_NUM_CHANNELS>
FORCE_INLINE  bool eth_check_receiver_ack_sequence(
    const QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr,
    const QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr,
    QueueIndexPointer<uint8_t> &eth_sender_rdptr,
    QueueIndexPointer<uint8_t> &eth_sender_ackptr,
    uint32_t &num_eth_sends_acked) {
    bool did_something = false;
    bool eth_sends_unacknowledged = QueueIndexPointer<uint8_t>::distance(eth_sender_rdptr, eth_sender_ackptr) > 0;
    if (eth_sends_unacknowledged) {
        bool transimission_acked_by_receiver = eth_is_receiver_channel_send_acked(eth_sender_ackptr.index()) || eth_is_receiver_channel_send_done(eth_sender_ackptr.index());
        if (transimission_acked_by_receiver) {
            // kernel_profiler::mark_time(15);
            // // DPRINT << "tx: got receiver ack on channel " << (uint32_t)eth_sender_ackptr << "\n";
            eth_clear_sender_channel_ack(eth_sender_ackptr.index());
            num_eth_sends_acked++;
            eth_sender_ackptr.increment();

            did_something = true;
        }
    }

    return did_something;
}

template <uint8_t MAX_NUM_CHANNELS>
FORCE_INLINE  bool noc_read_ack_check_sequence(
    QueueIndexPointer<uint8_t> &noc_reader_buffer_wrptr,
    QueueIndexPointer<uint8_t> &noc_reader_buffer_ackptr,
    const uint8_t noc_index) {
    bool did_something = false;

    bool noc_read_is_in_progress =
        is_noc_read_in_progress(noc_reader_buffer_wrptr, noc_reader_buffer_ackptr);
    if (noc_read_is_in_progress) {
#if EMULATE_DRAM_READ_CYCLES == 1
        bool read_finished = emulated_dram_read_cycles_finished();
#else
        bool read_finished = ncrisc_noc_reads_flushed(noc_index);
#endif
        if (read_finished) {
            // kernel_profiler::mark_time(13);
            // // DPRINT << "tx: read completed on channel " << (uint32_t)noc_reader_buffer_ackptr << "\n";
            noc_reader_buffer_ackptr.increment();
            did_something = true;
        }
    }

    return did_something;
}

template <uint8_t MAX_NUM_CHANNELS, bool src_is_dram>
FORCE_INLINE  bool noc_read_data_sequence(
    std::array<uint32_t, MAX_NUM_CHANNELS> &transaction_channel_sender_buffer_addresses,
    uint32_t num_bytes_per_send,
    QueueIndexPointer<uint8_t> &noc_reader_buffer_wrptr,
    QueueIndexPointer<uint8_t> &noc_reader_buffer_ackptr,
    const QueueIndexPointer<uint8_t> eth_sender_rdptr,
    const QueueIndexPointer<uint8_t> eth_sender_ackptr,
    const uint8_t noc_index,
    const InterleavedAddrGen<src_is_dram> &source_address_generator,
    const uint32_t page_size,
    const uint32_t num_pages_per_l1_buffer,
    const uint32_t num_pages,
    uint32_t &page_index
    ) {
    bool did_something = false;

    bool noc_read_is_in_progress =
        is_noc_read_in_progress(noc_reader_buffer_wrptr, noc_reader_buffer_ackptr);
    bool more_data_to_read = page_index < num_pages;
    if (!noc_read_is_in_progress && more_data_to_read) {
        // We can only If a noc read is in progress, we can't issue another noc read

        bool next_buffer_available = !buffer_pool_full<MAX_NUM_CHANNELS>(
            noc_reader_buffer_wrptr, noc_reader_buffer_ackptr, eth_sender_rdptr, eth_sender_ackptr);
        // Really we should be able to assert on this second condition but I don't yet know how to
        // propagate that info to host (especially on erisc... TODO(snijjar))
        // next_buffer_available = next_buffer_available && channels_active[noc_reader_buffer_wrptr] == 0;
        if (next_buffer_available) {
            // Queue up another read
            // non blocking - issues noc_async_read
            // issue_read_chunk(noc_reader_buffer_wrptr, ...);
            // kernel_profiler::mark_time(12);
            #if EMULATE_DRAM_READ_CYCLES == 1
            issue_read_chunk();
            #else

            // // DPRINT << "tx: reading data into L1 buffer on channel " << (uint32_t)noc_reader_buffer_wrptr << "\n";
            read_chunk<src_is_dram>(
                transaction_channel_sender_buffer_addresses[noc_reader_buffer_wrptr.index()], // eth_l1_buffer_address_base
                num_pages,
                num_pages_per_l1_buffer,
                page_size,
                page_index,
                source_address_generator
            );
            #endif
            noc_reader_buffer_wrptr.increment();

            did_something = true;
        }
    }

    return did_something;
}


template <int8_t MAX_CONCURRENT_TRANSACTIONS>
void initialize_transaction_buffer_addresses(
    uint32_t sender_buffer_base_address,
    uint32_t receiver_buffer_base_address,
    uint32_t num_bytes_per_send,
    std::array<uint32_t, MAX_CONCURRENT_TRANSACTIONS> &transaction_channel_sender_buffer_addresses,
    std::array<uint32_t, MAX_CONCURRENT_TRANSACTIONS> &transaction_channel_receiver_buffer_addresses) {
    uint32_t sender_buffer_address = sender_buffer_base_address;
    uint32_t receiver_buffer_address = receiver_buffer_base_address;
    for (uint32_t i = 0; i < MAX_CONCURRENT_TRANSACTIONS; i++) {
        transaction_channel_sender_buffer_addresses[i] = sender_buffer_address;
        transaction_channel_receiver_buffer_addresses[i] = receiver_buffer_address;
        #if ENABLE_L1_BUFFER_OVERLAP == 0
        sender_buffer_address += num_bytes_per_send;
        receiver_buffer_address += num_bytes_per_send;
        #endif
    }
}



void kernel_main() {
    // COMPILE TIME ARGS
    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);
    constexpr std::uint32_t total_num_message_sends = get_compile_time_arg_val(2);
    constexpr std::uint32_t NUM_TRANSACTION_BUFFERS = get_compile_time_arg_val(3);
    constexpr bool src_is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t MAX_NUM_CHANNELS = NUM_TRANSACTION_BUFFERS;

    // COMPILE TIME ARG VALIDATION
    static_assert(MAX_NUM_CHANNELS > 1, "Implementation currently doesn't support single buffering");

    // RUNTIME ARGS
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);

    std::uint32_t src_addr = get_arg_val<uint32_t>(2);
    std::uint32_t page_size = get_arg_val<uint32_t>(3);
    std::uint32_t num_pages = get_arg_val<uint32_t>(4);

    QueueIndexPointer<uint8_t> noc_reader_buffer_ackptr(MAX_NUM_CHANNELS);
    QueueIndexPointer<uint8_t> noc_reader_buffer_wrptr(MAX_NUM_CHANNELS);
    QueueIndexPointer<uint8_t> eth_sender_rdptr(MAX_NUM_CHANNELS);
    QueueIndexPointer<uint8_t> eth_sender_ackptr(MAX_NUM_CHANNELS);

    // Handshake with the other erisc first so we don't include dispatch time
    // in our measurements
    eth_setup_handshake(remote_eth_l1_dst_addr, true);

    // const InterleavedAddrGenFast<src_is_dram> s = {
    //     .bank_base_address = src_addr, .page_size = page_size, .data_format = df};

    const InterleavedAddrGen<src_is_dram> source_address_generator = {
        .bank_base_address = src_addr, .page_size = page_size};

    kernel_profiler::mark_time(10);

    // SETUP DATASTRUCTURES
    std::array<uint32_t, MAX_NUM_CHANNELS> transaction_channel_sender_buffer_addresses;
    std::array<uint32_t, MAX_NUM_CHANNELS> transaction_channel_receiver_buffer_addresses;
    initialize_transaction_buffer_addresses<MAX_NUM_CHANNELS>(
        local_eth_l1_src_addr,
        remote_eth_l1_dst_addr,
        num_bytes_per_send,
        transaction_channel_sender_buffer_addresses,
        transaction_channel_receiver_buffer_addresses);

    uint32_t eth_sends_completed = 0;

    kernel_profiler::mark_time(11);
    constexpr uint32_t SWITCH_INTERVAL = 100000;
    uint32_t count = 0;
    uint32_t page_index = 0;
    uint32_t num_pages_per_l1_buffer = num_bytes_per_send / page_size;
    uint32_t num_context_switches = 0;
    uint32_t max_num_context_switches = 10000;
    bool printed_hang = false;
    uint32_t total_eth_sends = 0;
    while (eth_sends_completed < total_num_message_sends) {
        bool did_something = false;

        did_something = noc_read_ack_check_sequence<MAX_NUM_CHANNELS>(
                            noc_reader_buffer_wrptr,
                            noc_reader_buffer_ackptr,
                            noc_index) || did_something;

        did_something = noc_read_data_sequence<MAX_NUM_CHANNELS,src_is_dram>(
                            transaction_channel_sender_buffer_addresses,
                            num_bytes_per_send,
                            noc_reader_buffer_wrptr,
                            noc_reader_buffer_ackptr,
                            eth_sender_rdptr,
                            eth_sender_ackptr,
                            noc_index,
                            source_address_generator,
                            page_size,
                            num_pages_per_l1_buffer,
                            num_pages,
                            page_index) || did_something;

        bool sent_eth_data = eth_send_data_sequence<MAX_NUM_CHANNELS>(
                            transaction_channel_sender_buffer_addresses,
                            transaction_channel_receiver_buffer_addresses,
                            local_eth_l1_src_addr,
                            remote_eth_l1_dst_addr,
                            num_bytes_per_send,  // bytes to send from this buffer over eth link
                            num_bytes_per_send,  // break the end-to-end send into messages of this size
                            num_bytes_per_send_word_size,
                            noc_reader_buffer_wrptr,
                            noc_reader_buffer_ackptr,
                            eth_sender_rdptr,
                            eth_sender_ackptr);
        total_eth_sends = sent_eth_data ? total_eth_sends + 1 : total_eth_sends;
        did_something = sent_eth_data || did_something;


        did_something = eth_check_receiver_ack_sequence<MAX_NUM_CHANNELS>(
                            noc_reader_buffer_wrptr,
                            noc_reader_buffer_ackptr,
                            eth_sender_rdptr,
                            eth_sender_ackptr,
                            eth_sends_completed) ||
                        did_something;

        if (!did_something) {
            if (count++ > SWITCH_INTERVAL) {
                count = 0;
                // kernel_profiler::mark_time(15);
                run_routing();
                num_context_switches++;
                if (num_context_switches > max_num_context_switches) {
                    if (!printed_hang) {
                        // DPRINT << "tx: HANG\n";
                        // DPRINT << "tx: HANG eth_sends_completed " << eth_sends_completed << "\n";
                        // DPRINT << "tx: HANG noc_reader_buffer_wrptr " << (uint32_t)noc_reader_buffer_ackptr.index() << "\n";
                        // DPRINT << "tx: HANG (raw) noc_reader_buffer_wrptr " << (uint32_t)noc_reader_buffer_ackptr.raw_index() << "\n";
                        // DPRINT << "tx: HANG noc_reader_buffer_ackptr " << (uint32_t)noc_reader_buffer_wrptr.index() << "\n";
                        // DPRINT << "tx: HANG (raw) noc_reader_buffer_ackptr " << (uint32_t)noc_reader_buffer_wrptr.raw_index() << "\n";
                        // DPRINT << "tx: HANG eth_sender_rdptr " << (uint32_t)eth_sender_rdptr.index() << "\n";
                        // DPRINT << "tx: HANG (raw) eth_sender_rdptr " << (uint32_t)eth_sender_rdptr.raw_index() << "\n";
                        // DPRINT << "tx: HANG eth_sender_ackptr " << (uint32_t)eth_sender_ackptr.index() << "\n";
                        // DPRINT << "tx: HANG (raw) eth_sender_ackptr " << (uint32_t)eth_sender_ackptr.raw_index() << "\n";
                        // DPRINT << "tx: HANG total_eth_sends " << (uint32_t)total_eth_sends << "\n";
                        for (uint32_t i = 0; i < MAX_NUM_CHANNELS; i++) {
                            // DPRINT << "tx: HANG channel [" << i << "] bytes_sent " << erisc_info->per_channel_user_bytes_send[0].bytes_sent << "\n";
                            // DPRINT << "tx: HANG channel [" << i << "] bytes_receiver_ack " << erisc_info->per_channel_user_bytes_send[0].receiver_ack << "\n";
                            // DPRINT << "tx: HANG eth_is_receiver_channel_send_acked (" << i << ") " << (eth_is_receiver_channel_send_acked(i) ? "true" : "false") << "\n";
                            // DPRINT << "tx: HANG eth_is_receiver_channel_send_done(" << i << ") " << (eth_is_receiver_channel_send_done(i) ? "true" : "false") << "\n";
                        }
                        num_context_switches = 0;
                        printed_hang = true;
                    }
                }
            } else {
                count++;
            }
        } else {
            num_context_switches = 0;
        }
    }


    // DPRINT << "tx: DONE\n";
    // DPRINT << "tx: DONE eth_sends_completed " << (uint32_t)eth_sends_completed << "\n";
    // DPRINT << "tx: DONE total_num_message_sends " << (uint32_t)total_num_message_sends << "\n";
    kernel_profiler::mark_time(16);
}
