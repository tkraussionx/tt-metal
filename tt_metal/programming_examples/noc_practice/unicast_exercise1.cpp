#include <cstring>
#include <string>
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

uint16_t round_to_nearest_even(float val) {
    uint _val = reinterpret_cast<uint &>(val);
    return static_cast<ushort>((_val + ((_val >> 16) & 1) + ((uint)0x7FFF)) >> 16);
}

float bf16_to_float(uint16_t bf16) {
    union {
        uint32_t u;
        float f;
    } tmp;
    tmp.u = bf16 << 16;
    return tmp.f;
}

void print_buffer_info(std::shared_ptr<const Buffer> buffer, std::string name) {
    log_info(tt::LogTest, "{} info", name);
    log_info(tt::LogTest, "\taddress: {}", buffer->address());
    log_info(tt::LogTest, "\ttotal size : {}", buffer->size());
    log_info(tt::LogTest, "\tpage size: {}", buffer->page_size());
    log_info(tt::LogTest, "\tnoc xy: {}", buffer->noc_coordinates());
}

int main() {
    // Create device object and get command queue.
    constexpr int device_id = 0;
    tt::tt_metal::Device *device = tt::tt_metal::CreateDevice(device_id);
    tt::tt_metal::CommandQueue &cq = device->command_queue();

    //////////////////////////////////////////////////////////////////////////////////
    // Allocate host buffer0.
    //////////////////////////////////////////////////////////////////////////////////
    tt::DataFormat buffer_format = tt::DataFormat::Float16_b;
    uint32_t tile_size = tt::tt_metal::detail::TileSize(buffer_format);
    uint32_t num_tiles = 12;
    uint32_t host_buffer_size = num_tiles * tile_size;
    auto host_buffer0 = std::shared_ptr<void>(malloc(host_buffer_size), free);

    /////////////////////////////////////////////////////////////////////////////////
    // Allocate DRAM Buffer
    /////////////////////////////////////////////////////////////////////////////////
    uint32_t dram_buffer_size = host_buffer_size;
    auto dram_config = tt::tt_metal::InterleavedBufferConfig{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> input_dram_buffer = tt::tt_metal::CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = tt::tt_metal::CreateBuffer(dram_config);

    // Get Device Buffer 0 Info
    print_buffer_info(input_dram_buffer, "input_dram_buffer");
    print_buffer_info(output_dram_buffer, "output_dram_buffer");

    /////////////////////////////////////////////////////////////////////////////////
    // Copy host buffer to dram buffer
    /////////////////////////////////////////////////////////////////////////////////
    // Clear dram buffers
    std::memset(host_buffer0.get(), 0, host_buffer_size);
    tt::tt_metal::EnqueueWriteBuffer(cq, input_dram_buffer, host_buffer0.get(), true /*blocking*/);
    tt::tt_metal::EnqueueWriteBuffer(cq, output_dram_buffer, host_buffer0.get(), true /*blocking*/);

    // Fill host buffer0.
    auto host_buffer0_ptr = reinterpret_cast<uint16_t *>(host_buffer0.get());
    for (int i = 0; i < num_tiles * tt::constants::TILE_HW; ++i) {
        host_buffer0_ptr[i] = round_to_nearest_even(i);
    }
    tt::tt_metal::EnqueueWriteBuffer(cq, input_dram_buffer, host_buffer0.get(), true /*blocking*/);

    /////////////////////////////////////////////////////////////////////////////////
    // Create program instance.
    /////////////////////////////////////////////////////////////////////////////////
    Program program = tt::tt_metal::CreateProgram();
    CoreCoord core = CoreCoord{0, 0};

    /////////////////////////////////////////////////////////////////////////////////
    // Create kernels
    /////////////////////////////////////////////////////////////////////////////////
    auto data_movement_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/noc_practice/kernels/reader_and_writer_unicast_exercise1.cpp", /*reader kernel path. */
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

    /////////////////////////////////////////////////////////////////////////////////
    // Set runtime args
    /////////////////////////////////////////////////////////////////////////////////
    CoreCoord input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    CoreCoord output_dram_noc_xy = output_dram_buffer->noc_coordinates();
    uint32_t l1_buffer_addr = 400 * 1024;
    const std::vector<uint32_t> reader_runtime_args = {
        l1_buffer_addr,
        input_dram_buffer->address(),
        static_cast<uint32_t>(input_dram_noc_xy.x),
        static_cast<uint32_t>(input_dram_noc_xy.y),
        output_dram_buffer->address(),
        static_cast<uint32_t>(output_dram_noc_xy.x),
        static_cast<uint32_t>(output_dram_noc_xy.y),
        dram_buffer_size};
    tt::tt_metal::SetRuntimeArgs(program, data_movement_kernel, core, reader_runtime_args);


    //////////////////////////////////////////////////////////////////////////////////
    // EnqueueProgram and Copy output_dram_buffer to host buffer1
    //////////////////////////////////////////////////////////////////////////////////
    // Clear L1 buffer
    std::vector<uint32_t> zero (dram_buffer_size / sizeof(uint32_t));
    tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_buffer_addr, zero);

    EnqueueProgram(cq, program, true /*blocking*/);

    auto host_buffer1 = std::shared_ptr<void>(malloc(host_buffer_size), free);
    EnqueueReadBuffer(cq, output_dram_buffer, host_buffer1.get(), true /*blocking*/);
    auto host_buffer1_ptr = reinterpret_cast<uint16_t *>(host_buffer1.get());

    bool pass = true;
    for (int i = 0; i < num_tiles * tt::constants::TILE_HW; ++i) {
        if (host_buffer0_ptr[i] != host_buffer1_ptr[i]) {
            pass = false;
            break;
        }
    }
    tt::tt_metal::CloseDevice(device);

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }
    return 0;
}
