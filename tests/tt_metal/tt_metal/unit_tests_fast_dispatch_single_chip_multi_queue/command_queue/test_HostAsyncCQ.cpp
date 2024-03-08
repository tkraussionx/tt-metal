#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt::tt_metal;

namespace host_cq_test_utils {
    struct TestBufferConfig {
    uint32_t num_pages;
    uint32_t page_size;
    BufferType buftype;
};

Program create_simple_unary_program(Buffer& input, Buffer& output) {
    Program program = CreateProgram();
    Device* device = input.device();
    CoreCoord worker = {0, 0};
    auto reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        worker,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        worker,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});


    vector<uint32_t> compute_kernel_args = {
        32
    };

    auto sfpu_kernel = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        worker,
        ComputeConfig{
            .math_approx_mode = true,
            .compile_args = {1, 1},
            .defines = {{"SFPU_OP_IDENTITY_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "identity_tile_init(); identity_tile(0);"}}});

    CircularBufferConfig input_cb_config = CircularBufferConfig(2048, {{0, tt::DataFormat::Float32}})
            .set_page_size(0, 2048);

    CoreRange core_range({0, 0});
    CreateCircularBuffer(program, core_range, input_cb_config);
    std::shared_ptr<RuntimeArgs> writer_runtime_args = std::make_shared<RuntimeArgs>();
    std::shared_ptr<RuntimeArgs> reader_runtime_args = std::make_shared<RuntimeArgs>();

    *writer_runtime_args = {
        &output,
        (uint32_t)output.noc_coordinates().x,
        (uint32_t)output.noc_coordinates().y,
        output.num_pages()
    };

    *reader_runtime_args = {
        &input,
        (uint32_t)input.noc_coordinates().x,
        (uint32_t)input.noc_coordinates().y,
        input.num_pages()
    };
    // SetRuntimeArgs(program, writer_kernel, worker, writer_runtime_args);
    // SetRuntimeArgs(program, reader_kernel, worker, reader_runtime_args);
    SetRuntimeArgs(device->command_queue(0), detail::GetKernel(program, writer_kernel), worker, writer_runtime_args);
    SetRuntimeArgs(device->command_queue(0), detail::GetKernel(program, reader_kernel), worker, reader_runtime_args);

    CircularBufferConfig output_cb_config = CircularBufferConfig(2048, {{16, tt::DataFormat::Float32}})
            .set_page_size(16, 2048);

    CreateCircularBuffer(program, core_range, output_cb_config);
    return program;
}
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestMultiAsyncCQ) {
    CommandQueue& cq_0 = this->device_->command_queue(0);
    CommandQueue& cq_1 = this->device_->command_queue(1);
    std::vector<std::shared_ptr<Buffer>> inputs = {};
    std::vector<std::shared_ptr<Buffer>> intermeds = {};
    std::vector<std::shared_ptr<Buffer>> outputs  = {};

    std::vector<std::vector<float>> golden_outputs = {};
    std::vector<uint32_t> golden_buf_addrs = {};
    std::vector<float> input_data(1024, 0);

    log_info(tt::LogTest, "Generating Reference Tensors.");
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < input_data.size(); j++) {
            input_data[j] = static_cast<float>(i + j);
        }
        inputs.push_back(std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 0));
        intermeds.push_back(std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 0));
        outputs.push_back(std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 0));
        golden_outputs.push_back({});

        Program op0 = host_cq_test_utils::create_simple_unary_program(*(inputs[i]), *(intermeds[i]));
        Program op1 = host_cq_test_utils::create_simple_unary_program(*(intermeds[i]), *(outputs[i]));
        EnqueueWriteBuffer(cq_0, inputs[i], input_data.data(), true);
        EnqueueProgram(cq_0, op0, true);
        EnqueueProgram(cq_0, op1, true);
        golden_outputs.back() = std::vector<float>(1024, 0);
        EnqueueReadBuffer(cq_0, outputs[i], golden_outputs.back().data(), true);
        // Store buffer addrs in synchronous mode. Compare with Async mode
        golden_buf_addrs.push_back(inputs[i]->address());
        golden_buf_addrs.push_back(intermeds[i]->address());
        golden_buf_addrs.push_back(outputs[i]->address());
    }

    inputs.clear();
    intermeds.clear();
    outputs.clear();
    Finish(cq_0);

    // Get output in async mode with non blocking calls (but syncs inserted between required calls).
    // Buffers stay around for the entire duration of this workload.
    // Using both cqs.
    auto current_mode = CommandQueue::default_mode();
    cq_0.set_mode(CommandQueue::CommandQueueMode::ASYNC);
    cq_1.set_mode(CommandQueue::CommandQueueMode::ASYNC);

    log_info(tt::LogTest, "Running workload with multiple CQs in Async Mode (Phase 1).");
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < input_data.size(); j++) {
            input_data[j] = static_cast<float>(i + j);
        }
        std::shared_ptr<Event> write_event = std::make_shared<Event>();
        std::shared_ptr<Event> compute_event = std::make_shared<Event>();

        inputs.push_back(std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 0));
        intermeds.push_back(std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 1));
        outputs.push_back(std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 0));

        std::vector<float> output_data(1024, 0);
        Program op0 = host_cq_test_utils::create_simple_unary_program(*(inputs[i]), *(intermeds[i]));
        Program op1 = host_cq_test_utils::create_simple_unary_program(*(intermeds[i]), *(outputs[i]));
        EnqueueWriteBuffer(cq_1, inputs[i], input_data.data(), false);
        EnqueueRecordEvent(cq_1, write_event);
        EnqueueWaitForEvent(cq_0, write_event);
        EnqueueProgram(cq_0, op0, false);
        EnqueueProgram(cq_0, op1, false);
        EnqueueRecordEvent(cq_0, compute_event);
        EventSynchronize(compute_event);

        EnqueueReadBuffer(cq_1, outputs[i], output_data.data(), false);
        Finish(cq_0);
        Finish(cq_1);
        ASSERT_EQ(golden_outputs[i], output_data);
        // Verify that memory allocation order in async mode matches sync mode.
        ASSERT_EQ(inputs[i]->address(), golden_buf_addrs[i * 3]);
        ASSERT_EQ(intermeds[i]->address(), golden_buf_addrs[i * 3 + 1]);
        ASSERT_EQ(outputs[i]->address(), golden_buf_addrs[i * 3 + 2]);
    }
    // Deallocate device memory
    inputs.clear();
    intermeds.clear();
    outputs.clear();

    log_info(tt::LogTest, "Running workload with multiple CQs in Async Mode (Phase 2).");
    // Use both CQs in Async mode. Buffers are temporary for this workload (similar to what tt-eager does).
    for (int i = 0; i < 200; i++) {
        for (int j = 0; j < input_data.size(); j++) {
            input_data[j] = static_cast<float>(i + j);
        }
        std::shared_ptr<Event> write_event = std::make_shared<Event>();
        std::shared_ptr<Event> compute_event = std::make_shared<Event>();

        std::shared_ptr<Buffer> input = std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 1);
        std::shared_ptr<Buffer> intermed = std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 0);
        std::shared_ptr<Buffer> output = std::make_shared<Buffer>(this->device_, 4096, 4096, BufferType::DRAM, TensorMemoryLayout::INTERLEAVED, nullopt, 0);

        std::vector<float> output_data(1024, 0);

        Program op0 = host_cq_test_utils::create_simple_unary_program(*input, *intermed);
        Program op1 = host_cq_test_utils::create_simple_unary_program(*intermed, *output);

        EnqueueWriteBuffer(cq_1, input, input_data.data(), false);

        AssignGlobalBufferToProgram(input, op0, 0);
        AssignGlobalBufferToProgram(intermed, op1, 0);

        input.reset();
        intermed.reset();

        EnqueueRecordEvent(cq_1, write_event);
        EnqueueWaitForEvent(cq_0, write_event);

        EnqueueProgram(cq_0, op0, false);
        EnqueueProgram(cq_0, op1, false);
        EnqueueRecordEvent(cq_0, compute_event);
        EventSynchronize(compute_event);

        EnqueueReadBuffer(cq_1, output, output_data.data(), false);
        Finish(cq_0);
        Finish(cq_1);
        ASSERT_EQ(golden_outputs[i], output_data);
    }
    cq_0.set_mode(current_mode);
    cq_1.set_mode(current_mode);
}
