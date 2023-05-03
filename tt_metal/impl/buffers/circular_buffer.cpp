#include "tt_metal/impl/buffers/circular_buffer.hpp"

#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

CircularBuffer::CircularBuffer(
    Device *device,
    const tt_xy_pair &logical_core,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    DataFormat data_format) :
    buffer_index_(buffer_index), num_tiles_(num_tiles), data_format_(data_format), L1Buffer(device, logical_core, size_in_bytes) {
}

CircularBuffer::CircularBuffer(
    Device *device,
    const tt_xy_pair &logical_core,
    uint32_t buffer_index,
    uint32_t num_tiles,
    uint32_t size_in_bytes,
    uint32_t address,
    DataFormat data_format) :
    buffer_index_(buffer_index), num_tiles_(num_tiles), data_format_(data_format), L1Buffer(device, logical_core, size_in_bytes, address) {
}

Buffer *CircularBuffer::clone() {
    return new CircularBuffer(
        this->device_, this->logical_core_, this->buffer_index_, this->num_tiles_, this->size_in_bytes_, this->data_format_);
}

CircularBuffer::~CircularBuffer() {
    this->free();
}

TableCircularBuffer::TableCircularBuffer(Device *device, const uint32_t num_tables, const uint32_t table_size_in_bytes) : device_(device), num_tables_(num_tables), table_size_in_bytes_(table_size_in_bytes), size_in_bytes_(num_tables * table_size_in_bytes_) {
    // For now, hard-coding to 0... eventually, need to make configurable for both dispatch cores
    uint32_t dispatch_core_id = 0;

    tt_cluster* cluster = device->cluster();

    // Setup CB write interface
    this->write_ptr_ = 0;
    uint32_t fifo_addr = this->write_ptr_;
    uint32_t fifo_size = this->size_in_bytes_;
    uint32_t fifo_size_tables = this->num_tables_;

    uint32_t fifo_limit = fifo_addr + fifo_size - 1;
    uint32_t fifo_wr_ptr = fifo_addr;

    // Writing the write interface to system memory
    vector<uint32_t> cb_write_interface = {fifo_limit, fifo_wr_ptr, fifo_size, fifo_size_tables};
    cluster->write_sysmem_vec(cb_write_interface, 0, 0);

    // Setup CB read interface
    tt_xy_pair dispatch_core = device->worker_core_from_logical_core({9, 0});
    vector<uint32_t> cb_read_interface;
    uint32_t table_cb_read_interface_addr;
    llrt::write_hex_vec_to_core(cluster, 0, dispatch_core, cb_read_interface, table_cb_read_interface_addr);
}

void TableCircularBuffer::cb_reserve_back() {

}

void TableCircularBuffer::cb_push_back() {

}


}  // namespace tt_metal

}  // namespace tt
