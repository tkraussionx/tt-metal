template <uint32_t consumer_cmd_base_addr, uint32_t consumer_data_buffer_size, PullAndRelayType src_type, PullAndRelayType dst_type>
void pull_and_relay(
    uint32_t producer_cb_size,
    uint32_t producer_cb_num_pages,
    uint64_t consumer_noc_encoding,
    uint32_t producer_consumer_transfer_num_pages,
    bool db_buf_switch,
    db_cb_config_t* db_cb_config,
    const db_cb_config_t* remote_db_cb_config) {

    uint32_t consumer_cb_size = (db_cb_config->total_size_16B << 4);
    uint32_t consumer_cb_num_pages = db_cb_config->num_pages;

    uint32_t l1_consumer_fifo_limit_16B =
        (get_db_buf_addr<consumer_cmd_base_addr, consumer_data_buffer_size>(db_buf_switch) + consumer_cb_size) >> 4;

    Buffer buffer;
    buffer.init((BufferType)src_buf_type, bank_base_address, page_size);

    while (num_writes_completed != num_pages) {
        if (cb_producer_space_available(num_to_read) and num_reads_issued < num_pages) {
            uint32_t l1_write_ptr = get_write_ptr(0);
            if constexpr (src_type == PullAndRelayType::BUFFER) {
                /*
                    In this case, we are pulling from a buffer. We pull from
                    buffers when our src is in system memory, or we are pulling in
                    data from local chip SRAM/DRAM.
                */
                buffer.noc_async_read_buffer(l1_write_ptr, src_page_id, num_to_read);
            } else if constexpr (src_type == PullAndRelayType::CIRCULAR_BUFFER) {
                /*
                    In this case, we are pulling from a circular buffer. We pull from
                    circular buffers typically when our src is an erisc core.
                */
            } else {
                static_assert(false);
            }

            cb_push_back(0, num_to_read);
            num_reads_issued += num_to_read;
            src_page_id += num_to_read;

            uint32_t num_pages_left = num_pages - num_reads_issued;
            num_to_read = min(num_pages_left, fraction_of_producer_cb_num_pages);
        }

        if (num_reads_issued > num_writes_completed and cb_consumer_space_available(db_cb_config, num_to_write)) {
            if (num_writes_completed == num_reads_completed) {
                noc_async_read_barrier();
                num_reads_completed = num_reads_issued;
            }

            uint32_t dst_addr = (db_cb_config->wr_ptr_16B << 4);
            uint64_t dst_noc_addr = consumer_noc_encoding | dst_addr;
            if constexpr (dst_type == PullAndRelayType::CIRCULAR_BUFFER) {
                /*
                    In this case, we are writing to a circular buffer. This is the most
                    common case in which we are writing to a dispatch core.
                */
                noc_async_write(get_read_ptr(0), dst_noc_addr, page_size * num_to_write);
                multicore_cb_push_back(
                    db_cb_config, remote_db_cb_config, consumer_noc_encoding, l1_consumer_fifo_limit_16B, num_to_write);
                noc_async_write_barrier();
            } else if constexpr (dst_type == PullAndRelayType::BUFFER) {
                /*
                    In this case, we are writing to a buffer. This is only the case
                    for the completion queue writer, which relays data into the completion
                    queue from remote chips.
                */
            } else {
                static_assert(false);
            }
            cb_pop_front(0, num_to_write);
            num_writes_completed += num_to_write;
            num_to_write = min(num_pages - num_writes_completed, producer_consumer_transfer_num_pages);
        }
    }
}
