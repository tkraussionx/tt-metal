#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/tt_xy_pair.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

class CircularBuffer : public L1Buffer {
   public:
    CircularBuffer(
        Device *device,
        const tt_xy_pair &logical_core,
        uint32_t buffer_index,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format);

    CircularBuffer(
        Device *device,
        const tt_xy_pair &logical_core,
        uint32_t buffer_index,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t address,
        DataFormat data_format);

    ~CircularBuffer();

    Buffer *clone();

    uint32_t buffer_index() const { return buffer_index_; }

    uint32_t num_tiles() const { return num_tiles_; }

    DataFormat data_format() const { return data_format_; }

   private:
    uint32_t buffer_index_;               // A buffer ID unique within a Tensix core (0 to 32)
    uint32_t num_tiles_;                  // Size in tiles
    DataFormat data_format_;              // e.g. fp16, bfp8
};

class TableCircularBuffer {
    public:
     TableCircularBuffer(Device *device, const uint32_t num_tables, const uint32_t table_size_in_bytes) : num_tables_(num_tables), table_size_in_bytes_(table_size_in_bytes), size_in_bytes_(num_tables * table_size_in_bytes) {}
     ~TableCircularBuffer() {}

     void cb_reserve_back();

     void cb_push_back();

    private:
     Device *device_;
     const uint32_t size_in_bytes_;
     const uint32_t num_tables_;
     const uint32_t table_size_in_bytes_;
     uint32_t address_;

     uint32_t write_ptr_; // No read ptr, since device will update that
};


}  // namespace tt_metal

}  // namespace tt
