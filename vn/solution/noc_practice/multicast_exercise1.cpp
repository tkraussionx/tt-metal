#include <cstring>
#include <random>
#include <string>

#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

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

    // Get Device Buffer 0 Info
    print_buffer_info(input_dram_buffer, "input_dram_buffer");

    /////////////////////////////////////////////////////////////////////////////////
    // Copy host buffer to dram buffer
    /////////////////////////////////////////////////////////////////////////////////
    // Clear dram buffers
    std::memset(host_buffer0.get(), 0, host_buffer_size);
    tt::tt_metal::EnqueueWriteBuffer(cq, input_dram_buffer, host_buffer0.get(), true /*blocking*/);

    // Fill random values to host buffer0.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    auto host_buffer0_ptr = reinterpret_cast<uint16_t *>(host_buffer0.get());
    for (int i = 0; i < num_tiles * tt::constants::TILE_HW; ++i) {
        float random_value = dis(gen);
        host_buffer0_ptr[i] = round_to_nearest_even(random_value);
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
        "tt_metal/programming_examples/noc_practice/kernels/multicast_exercise1.cpp", /* reader and writer kernel path.
                                                                                       */
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_1_default});

    /////////////////////////////////////////////////////////////////////////////////
    // Set runtime args
    /////////////////////////////////////////////////////////////////////////////////
    CoreCoord input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    uint32_t l1_buffer_addr = 400 * 1024;
    CoreCoord grid_size = device->logical_grid_size();
    CoreCoord core_start = core;
    CoreCoord core_end = {core_start.x + (grid_size.x - 1), core_start.y + (grid_size.y - 1)};
    auto core_start_physical = device->worker_core_from_logical_core(core_start);
    auto core_end_physical = device->worker_core_from_logical_core(core_end);

    log_info(tt::LogTest, "logical coord");
    log_info(tt::LogTest, "start = {}", core_start);
    log_info(tt::LogTest, "end = {}", core_end);
    log_info(tt::LogTest, "physical coord");
    log_info(tt::LogTest, "start = {}", core_start_physical);
    log_info(tt::LogTest, "end = {}", core_end_physical);
    const std::vector<uint32_t> runtime_args = {
        l1_buffer_addr,                              // src l1 addr: Source L1 address from where data is read
        input_dram_buffer->address(),                // Address of the input DRAM buffer
        static_cast<uint32_t>(input_dram_noc_xy.x),  // X coordinate of the input DRAM in NOC
        static_cast<uint32_t>(input_dram_noc_xy.y),  // Y coordinate of the input DRAM in NOC
        dram_buffer_size,                            // Size of the DRAM buffer
        l1_buffer_addr,  // dest l1 addr: Destination L1 address to where data is to be written

        // Grid info for destination cores
        static_cast<uint32_t>(core_end_physical.x),    // X coordinate of the end core in the physical grid
        static_cast<uint32_t>(core_end_physical.y),    // Y coordinate of the end core in the physical grid
        static_cast<uint32_t>(core_start_physical.x),  // X coordinate of the start core in the physical grid
        static_cast<uint32_t>(core_start_physical.y),  // Y coordinate of the start core in the physical grid

        static_cast<uint32_t>(
            grid_size.x * grid_size.y - 1)  // num_dests: Number of destination cores, excluding core {0, 0}
    };
    tt::tt_metal::SetRuntimeArgs(program, data_movement_kernel, core, runtime_args);

    //////////////////////////////////////////////////////////////////////////////////
    // EnqueueProgram and read output from L1 of destination cores
    //////////////////////////////////////////////////////////////////////////////////
    // Clear L1 buffer
    std::vector<uint32_t> zero(dram_buffer_size / sizeof(uint32_t));
    tt::tt_metal::detail::WriteToDeviceL1(device, core, l1_buffer_addr, zero);

    EnqueueProgram(cq, program, true /*blocking*/);

    // Verify data fro all cores
    bool pass = true;
    for (int i = 0; i < grid_size.y; i++) {
        for (int j = 0; j < grid_size.x; j++) {
            CoreCoord dest_core = {(std::size_t)core_start.x + j, (std::size_t)core_start.y + i};
            std::vector<uint32_t> dest_core_data;
            tt::tt_metal::detail::ReadFromDeviceL1(device, dest_core, l1_buffer_addr, dram_buffer_size, dest_core_data);
            std::vector<uint16_t> vec16 = u16_from_u32_vector(dest_core_data);
            for (int i = 0; i < num_tiles * tt::constants::TILE_HW; ++i) {
                if (host_buffer0_ptr[i] != vec16[i]) {
                    pass = false;
                    log_info(tt::LogTest, "Mismatch on core {}, {}", dest_core.x, dest_core.y);
                    break;
                }
            }
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
