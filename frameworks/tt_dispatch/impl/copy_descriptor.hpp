#include <cstdint>
#include <array>

template <size_t T>
struct CopyDescriptor {

    static_assert((T % 2) == 0, "Size of copy descriptor must be divisible by 2");

    // These two attributes are only used by host to know where
    // to write this copy descriptor
    uint32_t l1_addr;
    const uint32_t read_base = 0;
    const uint32_t write_base = T / 2;
    uint32_t read_ptr = read_base + 1;
    uint32_t write_ptr = write_base + 1;

    std::array<uint32_t, T> data;

    void add_read(uint64_t src, uint32_t dst, uint32_t size) {
        this->data[read_base]++; // Increment num_reads

        uint32_t upper = src >> 32;
        uint32_t lower = src & 0xffffffff;
        this->data[this->read_ptr++] = lower;
        this->data[this->read_ptr++] = upper;
        this->data[this->read_ptr++] = dst;
        this->data[this->read_ptr++] = size;
    }

    void add_write(uint32_t src, uint64_t dst, uint32_t size) {
        this->data[write_base]++; // Increment num_writes

        this->data[this->write_ptr++] = src;
        uint32_t upper = dst >> 32;
        uint32_t lower = dst & 0xffffffff;
        this->data[this->write_ptr++] = lower;
        this->data[this->write_ptr++] = upper;
        this->data[this->write_ptr++] = size;
    }

    void clear() {
        this->data.fill(0);
        this->read_ptr = this->read_base;
        this->write_ptr = this->write_base;
    }
};
