#include <cstring>
#include <random>
#include <string>

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
    std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = tt::tt_metal::CreateBuffer(dram_config);

    // Get Device Buffer 0 Info
    print_buffer_info(input_dram_buffer, "input_dram_buffer");
    print_buffer_info(output_dram_buffer, "output_dram_buffer");

    /////////////////////////////////////////////////////////////////////////////////
    // Copy host buffer to dram buffer
    /////////////////////////////////////////////////////////////////////////////////
    // Clear dram buffers first
    std::memset(host_buffer0.get(), 0, host_buffer_size);
    tt::tt_metal::EnqueueWriteBuffer(cq, input_dram_buffer, host_buffer0.get(), true /*blocking*/);
    tt::tt_metal::EnqueueWriteBuffer(cq, output_dram_buffer, host_buffer0.get(), true /*blocking*/);

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
    // Allocate circular buffer 0 and 1.
    /////////////////////////////////////////////////////////////////////////////////
    uint32_t cb_num_tiles = 2;
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat cb0_data_format = tt::DataFormat::Float16_b;
    tt::tt_metal::CircularBufferConfig cb0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_tiles * tile_size, {{cb0_id, cb0_data_format}})
            .set_page_size(cb0_id, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb0_config);

    uint32_t cb1_id = tt::CB::c_out0;
    tt::DataFormat cb1_data_format = tt::DataFormat::Float16_b;
    tt::tt_metal::CircularBufferConfig cb1_config =
        tt::tt_metal::CircularBufferConfig(cb_num_tiles * tile_size, {{cb1_id, cb1_data_format}})
            .set_page_size(cb1_id, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb1_config);

    /////////////////////////////////////////////////////////////////////////////////
    // Create kernels
    /////////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::KernelHandle reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/noc_practice/kernels/reader_unicast_exercise2.cpp", /* reader kernel path. */
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

    tt::tt_metal::KernelHandle writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/noc_practice/kernels/writer_unicast_exercise2.cpp", /* writer kernel path. */
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default});

    tt::tt_metal::KernelHandle compute_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/noc_practice/kernels/compute_unicast_exercise2.cpp", /* compute kernel path. */
        core,
        ComputeConfig{
            .compile_args = {},
            .defines = {},
        });

    /////////////////////////////////////////////////////////////////////////////////
    // Set runtime args
    /////////////////////////////////////////////////////////////////////////////////
    CoreCoord input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    CoreCoord output_dram_noc_xy = output_dram_buffer->noc_coordinates();
    std::cout << input_dram_noc_xy.x << " " << input_dram_noc_xy.y << std::endl;
    std::cout << output_dram_noc_xy.x << " " << output_dram_noc_xy.y << std::endl;
    // const std::vector<uint32_t> reader_runtime_args = {
    //     input_dram_buffer->address(),
    //     static_cast<uint32_t>(input_dram_noc_xy.x),
    //     static_cast<uint32_t>(input_dram_noc_xy.y),
    //     num_tiles,
    //     tile_size};
    // tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);

    // const std::vector<uint32_t> writer_runtime_args = {
    //     output_dram_buffer->address(),
    //     static_cast<uint32_t>(output_dram_noc_xy.x),
    //     static_cast<uint32_t>(output_dram_noc_xy.y),
    //     num_tiles,
    //     tile_size};
    // tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core, writer_runtime_args);

    // const std::vector<uint32_t> compute_runtime_args = { num_tiles };
    // tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
    // //////////////////////////////////////////////////////////////////////////////////
    // // EnqueueProgram and Copy output_dram_buffer to host buffer1
    // //////////////////////////////////////////////////////////////////////////////////
    // tt::tt_metal::EnqueueProgram(cq, program, true /*blocking*/);

    // auto host_buffer1 = std::shared_ptr<void>(malloc(host_buffer_size), free);
    // tt::tt_metal::EnqueueReadBuffer(cq, output_dram_buffer, host_buffer1.get(), true /*blocking*/);
    // auto host_buffer1_ptr = reinterpret_cast<uint16_t *>(host_buffer1.get());

    // bool pass = true;
    // for (int i = 0; i < num_tiles * tt::constants::TILE_HW; ++i) {
    //     if (host_buffer0_ptr[i] != host_buffer1_ptr[i]) {
    //         pass = false;
    //         break;
    //     }
    // }
    // tt::tt_metal::CloseDevice(device);

    // if (pass) {
    //     log_info(tt::LogTest, "Test Passed");
    // } else {
    //     TT_THROW("Test Failed");
    // }
    // return 0;
}
