// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/types.hpp"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_impl.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "ttnn/cpp/ttnn/runtime_apis.hpp"
#include "tt_numpy/functions.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;


// auto op = EltwiseBinary {
    //     BinaryOpType::ADD,
    //     std::nullopt,
    //     operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    //     DataType::BFLOAT16,
    //     false
    // };
TEST_F(CommonFixture, TestBuffer) {
    Device* device = this->devices_[0];
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};


    uint32_t num_test_loops = tt::parse_env("NUM_TEST_LOOPS", 1);

    for (uint32_t test_idx = 0; test_idx < num_test_loops; test_idx++) {

        int test_val_int = tt::parse_env("VALUE", 16);

        // Set test_val_int to perfect squares increasing based on test_idx:
        if (num_test_loops > 1) {
            test_val_int = (test_idx + 1) * (test_idx + 1);
        }

        float test_val = (float) test_val_int;
        auto expected = static_cast<int>(std::round(std::sqrt(test_val)));

        log_info(tt::LogTest, "=====================================================");
        log_info(tt::LogTest, "Running test: {} with value: {} expected: {}", test_idx, test_val_int, expected);

        ttnn::Shape shape = ttnn::Shape(Shape({1, 1, 128, 128}));
        auto host_data_0 = std::shared_ptr<bfloat16 []>(new bfloat16[16384]);
        auto host_data_1 = std::shared_ptr<bfloat16 []>(new bfloat16[16384]);
        auto readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[16384]);
        for (int i = 0; i < 16384; i++) {
            float val = static_cast<float>(i);
            host_data_0[i] = bfloat16(test_val);
            host_data_1[i] = bfloat16(2 * val);
        }

        auto event0 = std::make_shared<Event>();
        auto event1 = std::make_shared<Event>();
        auto buffer0 = ttnn::allocate_buffer_on_device(16384 * 2, device, shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
        auto buffer1 = ttnn::allocate_buffer_on_device(16384 * 2, device, shape, DataType::BFLOAT16, Layout::ROW_MAJOR, mem_cfg);
        auto storage0 = tt::tt_metal::DeviceStorage{buffer0};
        auto storage1 = tt::tt_metal::DeviceStorage{buffer1};
        Tensor tensor0 = Tensor(storage0, shape, DataType::BFLOAT16, Layout::TILE);


        bool write_cq0 = tt::parse_env("WRITE_CQ0", false);

        if (write_cq0) {
            log_info(tt::LogTest, "KCM Writing using CQ0");
            ttnn::write_buffer(0, tensor0, {host_data_0}); // Write using cq 0
        } else {
            log_info(tt::LogTest, "KCM Writing using CQ1 w/ event sync");
            ttnn::write_buffer(1, tensor0, {host_data_0}); // Write using cq 1
            ttnn::record_event(device->command_queue(1), event0); // Record write on cq 1
            // Wait until cq 1 write is complete
            // ttnn::queue_synchronize(device->command_queue(1));
            ttnn::wait_for_event(device->command_queue(0), event0);
        }



        std::vector<Tensor> tensors = {tensor0};

        auto op0 = tt::tt_metal::EltwiseUnary{std::vector{tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::SQRT}}};
        // Run operation on cq 0
        Tensor tensor1 = ttnn::run_operation(0, op0, tensors).at(0);

        // KCM - Read using CQ0
        bool read_cq0 = tt::parse_env("READ_CQ0", false);
        if (read_cq0) {
            log_info(tt::LogTest, "KCM Reading using CQ0");
            // Read using cq 1
            ttnn::read_buffer(0, tensor1, {readback_data});

        } else {

            log_info(tt::LogTest, "KCM Reading using CQ1 w/ event sync");

            // Record cq 0 prog execution
            ttnn::record_event(device->command_queue(0), event1);
            // Wait until cq 0 prog execution is done
            // ttnn::queue_synchronize(device->command_queue(0));
            ttnn::wait_for_event(device->command_queue(1), event1);
            // Read using cq 1
            ttnn::read_buffer(1, tensor1, {readback_data});

        }

        // KCM Tweak
        int num_words_to_check = 10;
        // log_info(tt::LogTest, "Test {} expected: {} readback data ({} words) is...", test_idx, expected, num_words_to_check);
        bool check_passed = true;

        int observed;
        for (int i = 0; i < num_words_to_check; i++) { // Was 16384
            // std::cout << bfloat16(readback_data[i]).to_float() << " ";
            observed = static_cast<int>(std::round(bfloat16(readback_data[i]).to_float()));

            bool match = (observed == expected);
            // if (match) {
            //     log_info(tt::LogTest, "Rounded. Matching: {} Observed: {} Expected: {}", match, observed, expected);
            // } else {
            //     log_warning(tt::LogTest, "Rounded. Matching: {} Observed: {} Expected: {}", match, observed, expected);
            // }
            check_passed &= match;
        }

        if (check_passed) {
            log_info(tt::LogTest, "Test {} Passed. Observed: {} Expected: {}", test_idx, observed, expected);
        } else {
            log_warning(tt::LogTest, "Test {} Failed. Observed: {} Expected: {}", test_idx, observed, expected);
        }

        EXPECT_EQ(check_passed, true);

        // auto op1 = tt::tt_metal::EltwiseUnary{std::vector{tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::NEG}}};
        // std::vector<Tensor> tensors = {tensor0};

        // ttnn::read_buffer(0, tensor0, {readback_data});
        // Tensor tensor3 = ttnn::run_operation(0, op0, tensors).at(0);
        // std::vector<Tensor> tensors1 = {tensor3};
        // Tensor tensor4 = ttnn::run_operation(0, op1, tensors1).at(0);
        //
        // ttnn::wait_for_event(device->command_queue(1), event1);

        std::cout << std::endl;
        ttnn::queue_synchronize(device->command_queue());

    }


}

TEST_F(CommonFixture, TestTensorOwnershipSanity) {
    // Sanity test tensor read, write and update paths with synchronous
    // Ensure that tensor data is copied and owned as expected
    Device* device = this->devices_[0];
    Tensor host_tensor = tt::numpy::arange<float>(0, 32 * 32 * 4, 1);
    Tensor readback_tensor({}, 1);

    auto func = [device, host_tensor, readback_tensor]() mutable {
        // Ensure that both the lambda and global scope have ownership to this tensor
        EXPECT_EQ(host_tensor.tensor_attributes.use_count(), 2);
        std::visit([](auto&& storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                std::visit(
                    [](auto&& buf) {
                        using buf_type = std::decay_t<decltype(buf)>;
                        if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                            EXPECT_EQ(buf.use_count(), 1);
                        }
                    },
                storage.buffer);
            }
        }, host_tensor.get_storage());
        // Send tensor to device, read it back and copy it to empty tensor initialized by main thread
        Tensor reshaped_tensor = host_tensor.reshape(1, 1, 32, 128);
        auto device_tensor = reshaped_tensor.to(Layout::TILE).to(device);
        auto thread_local_tensor = device_tensor.cpu().to(Layout::ROW_MAJOR);
        readback_tensor.set_storage(thread_local_tensor.get_storage());
        readback_tensor.set_shape(thread_local_tensor.get_shape());
        readback_tensor.set_dtype(thread_local_tensor.get_dtype());
        readback_tensor.set_layout(thread_local_tensor.get_layout());
        readback_tensor.set_populated();
        // Ensure that the readback buffer is owned inside and outside the lambda
        std::visit([](auto&& storage) {
            using T = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<T, OwnedStorage>) {
                std::visit(
                    [](auto&& buf) {
                        using buf_type = std::decay_t<decltype(buf)>;
                        if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                            EXPECT_EQ(buf.use_count(), 2);
                        }
                    },
                storage.buffer);
            }
        }, readback_tensor.get_storage());
    };

    func();
     std::visit([](auto&& storage) {
        using T = std::decay_t<decltype(storage)>;
        if constexpr (std::is_same_v<T, OwnedStorage>) {
            std::visit(
                [](auto&& buf) {
                    using buf_type = std::decay_t<decltype(buf)>;
                    if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                        EXPECT_EQ(buf.use_count(), 1);
                        for (int i = 0; i < 128 * 32; i++) {
                            EXPECT_EQ(buf[i], i);
                        }
                    }
                },
            storage.buffer);
        }
    },
    readback_tensor.get_storage());
    EXPECT_EQ(readback_tensor.get_dtype(), DataType::FLOAT32);
    EXPECT_EQ(readback_tensor.get_layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(readback_tensor.get_shape(), ttnn::Shape(Shape({1, 1, 32, 128})));
}

TEST_F(CommonFixture, TestAsyncEltwiseBinary) {
    Device* device = this->devices_[0];
    device->set_worker_mode(WorkExecutorMode::ASYNCHRONOUS);
    // Populate these in first loop and verify that deallocation worked - addresses should be identical across loops
    std::size_t input_a_addr = 0;
    std::size_t input_b_addr = 0;
    std::size_t input_c_addr = 0;
    std::size_t output_1_addr = 0;
    std::size_t output_2_addr = 0;

    for (int i = 0; i < 5; i++) {
        // Initialize tensors and move them to DRAM
        Tensor input_tensor_a = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(i), DataType::BFLOAT16).to(device);
        Tensor input_tensor_b = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(i), DataType::BFLOAT16).to(device);
        Tensor input_tensor_c = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(i), DataType::BFLOAT16).to(device);
        Tensor output_tensor_device = mul(add(input_tensor_a, input_tensor_b), input_tensor_c);
        Tensor output_tensor_device_2 = neg(sub(output_tensor_device, input_tensor_c));

        EXPECT_EQ(output_tensor_device.get_shape(), ttnn::Shape(Shape({1, 1, 1024, 1024})));
        EXPECT_EQ(output_tensor_device.get_dtype(), DataType::BFLOAT16);

        Tensor output_tensor_host = output_tensor_device_2.cpu();
        // Test tensor deallocation in async mode: deallocate tensors after using them
        if (i == 0) {
            input_a_addr = std::get<DeviceStorage>(input_tensor_a.get_storage()).buffer->address();
            input_b_addr = std::get<DeviceStorage>(input_tensor_b.get_storage()).buffer->address();
            input_c_addr = std::get<DeviceStorage>(input_tensor_c.get_storage()).buffer->address();
            output_1_addr = std::get<DeviceStorage>(output_tensor_device.get_storage()).buffer->address();
            output_2_addr = std::get<DeviceStorage>(output_tensor_device_2.get_storage()).buffer->address();
        }
        else {
            EXPECT_EQ(std::get<DeviceStorage>(input_tensor_a.get_storage()).buffer->address(), input_a_addr);
            EXPECT_EQ(std::get<DeviceStorage>(input_tensor_b.get_storage()).buffer->address(), input_b_addr);
            EXPECT_EQ(std::get<DeviceStorage>(input_tensor_c.get_storage()).buffer->address(), input_c_addr);
            EXPECT_EQ(std::get<DeviceStorage>(output_tensor_device.get_storage()).buffer->address(), output_1_addr);
            EXPECT_EQ(std::get<DeviceStorage>(output_tensor_device_2.get_storage()).buffer->address(), output_2_addr);
        }
        input_tensor_a.deallocate();
        input_tensor_b.deallocate();
        input_tensor_c.deallocate();
        output_tensor_device.deallocate();
        output_tensor_device_2.deallocate();
        // Verify output data
        auto& buf = std::get<owned_buffer::Buffer<bfloat16>>(std::get<OwnedStorage>(output_tensor_host.get_storage()).buffer);
        EXPECT_EQ(buf.use_count(), 1);
        for (int j = 0; j < 1024 * 1024; j++) {
            EXPECT_EQ(bfloat16(buf[j]), bfloat16(static_cast<float>(i - 2 * i * i)));
        }
    }
    device->set_worker_mode(WorkExecutorMode::SYNCHRONOUS);
}

Tensor tensor_identity_copy_function(const Tensor& tensor) { return tensor; }

TEST_F(CommonFixture, TestAsyncRefCountManager) {
    Device* device = this->devices_[0];
    device->set_worker_mode(WorkExecutorMode::ASYNCHRONOUS);

    log_info(LogTest, "Testing Device tensor copy assignment");
    for (int i = 0; i < 5; i++) {
        // Run for multiple loops to ensure deterministic behaviour with device addresses
        // Initialize 2 tensors on device
        Tensor tensor1 = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(i), DataType::BFLOAT16).to(device);
        Tensor tensor2 = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(i), DataType::BFLOAT16).to(device);
        uint32_t tensor2_device_buf_addr = tensor2.device_buffer()->address();
        // Assign tensor1 to tensor2 and ensure that ref counts are appropriately updated with the buffer for tensor2 deallocated
        tensor2 = tensor1;
        EXPECT_EQ(tensor2.tensor_attributes->main_thread_ref_count, 2);
        EXPECT_EQ(tensor1.tensor_attributes->main_thread_ref_count, 2);
        // To check if tensor2 is deallocated, create a third tensor on device and ensure that its address matches the prev addr for tensor2
        Tensor tensor3 = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(i), DataType::BFLOAT16).to(device);
        EXPECT_EQ(tensor3.device_buffer()->address(), tensor2_device_buf_addr);
        EXPECT_EQ(tensor1.device_buffer()->address(), tensor2.device_buffer()->address());
    }
    log_info(LogTest, "Testing Device tensor self-assignment through function");
    for (int i = 0; i < 5; i++) {
        Tensor device_tensor = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(i), DataType::BFLOAT16).to(device);
        uint32_t device_tensor_address = device_tensor.device_buffer()->address();
        // This step will copy the tensor to a temp rval and std::move it back to the caller's instance of device_tensor
        // Ensure ref count and address remain unchanged
        device_tensor = tensor_identity_copy_function(device_tensor);
        EXPECT_EQ(device_tensor.tensor_attributes->main_thread_ref_count, 1);
        EXPECT_EQ(device_tensor.device_buffer()->address(), device_tensor_address);
    }

    log_info(LogTest, "Testing Device tensor move assignment");
    for (int i = 0; i < 5; i++) {
        Tensor tensor1 = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(i), DataType::BFLOAT16).to(device);
        Tensor tensor2 = std::move(tensor1);
        EXPECT_EQ(tensor2.tensor_attributes->main_thread_ref_count, 1);
        EXPECT_EQ(tensor1.tensor_attributes, nullptr);
    }

    log_info(LogTest, "Testing Device tensor self-assignment");
    Tensor tensor_to_self_assign = tt::numpy::full<float>(Shape({1, 1, 1024, 1024}), static_cast<float>(0), DataType::BFLOAT16).to(device);
    uint32_t tensor_to_self_assign_address = tensor_to_self_assign.device_buffer()->address();
    tensor_to_self_assign = tensor_to_self_assign;
    EXPECT_EQ(tensor_to_self_assign.tensor_attributes->main_thread_ref_count, 1);
    tensor_to_self_assign = std::move(tensor_to_self_assign);
    EXPECT_EQ(tensor_to_self_assign.device_buffer()->address(), tensor_to_self_assign_address);
    auto barrier_tensor = tensor_to_self_assign.cpu();
    device->set_worker_mode(WorkExecutorMode::SYNCHRONOUS);
}

TEST_F(CommonFixture, TestAsyncEltwiseBinaryAutoFormat) {
    // Test usecase where both inputs and outputs are on host and autoformat is used
    Device* device = this->devices_[0];
    device->set_worker_mode(WorkExecutorMode::ASYNCHRONOUS);
    AutoFormat::SetDefaultDevice(device);

    for (int i = 0; i < 5; i++) {
        // Initialize tensors and keep them on host. Since none of the tensors are divisible by tile dims, the inputs and
        // outputs are on host.
        Tensor input_tensor_a = tt::numpy::full<float>(Shape({1, 1, 1023, 1023}), static_cast<float>(i), DataType::BFLOAT16);
        Tensor input_tensor_b = tt::numpy::full<float>(Shape({1, 1, 1023, 1023}), static_cast<float>(i), DataType::BFLOAT16);
        Tensor input_tensor_c = tt::numpy::full<float>(Shape({1, 1, 1023, 1023}), static_cast<float>(i), DataType::BFLOAT16);
        Tensor output_tensor_device = mul(add(input_tensor_a, input_tensor_b), input_tensor_c);
        Tensor output_tensor_device_2 = neg(sub(output_tensor_device, input_tensor_c));

        EXPECT_EQ(output_tensor_device.get_shape(), ttnn::Shape(Shape({1, 1, 1023, 1023})));
        EXPECT_EQ(output_tensor_device.get_dtype(), DataType::BFLOAT16);

        Tensor output_tensor_host = output_tensor_device_2.cpu();
        // Verify output data
        auto& buf = std::get<owned_buffer::Buffer<bfloat16>>(std::get<OwnedStorage>(output_tensor_host.get_storage()).buffer);
        for (int j = 0; j < 1023 * 1023; j++) {
            EXPECT_EQ(bfloat16(buf[j]), bfloat16(static_cast<float>(i - 2 * i * i)));
        }
    }
    device->set_worker_mode(WorkExecutorMode::SYNCHRONOUS);
}

TEST_F(CommonFixture, TestTensorAsyncDataMovement) {
    // Test 2 data paths here (resembles async mode):
    // 1. Main -> Worker: Create a tensor in the main thread. Ensure that it is accessible in the worker thread even after its destroyed
    //                    by the main thread. This resembles host -> device data movement
    // 2. Worker -> Main: Create an empty tensor in the mainb thread. Populate it in the worker thread. Ensure that the tensor is correctly
    //                    populated in the main thread once the worker is done.
    Device* device = this->devices_[0];
    uint32_t tensor_start = 0;
    uint32_t num_tiles = 128;
    uint32_t tensor_stop = TILE_HEIGHT * TILE_WIDTH * num_tiles;
    Tensor readback_tensor({}, 1);;
    std::thread worker;

    {
        // host_tensor only lives in this scope
        Tensor host_tensor = tt::numpy::arange<float>(tensor_start, tensor_stop, 1);
        log_info(LogTest, "Spawning worker thread");
        worker = std::thread([tensor_stop, host_tensor, readback_tensor, device] () mutable {
            // Sleep for 3 seconds to ensure that main thread deallocates host_tensor
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
            log_info(LogTest, "Worker started");
            // Main thread should have deallocated host_tensor by this point
            EXPECT_EQ(host_tensor.tensor_attributes.use_count(), 1);
            // Ensure that the buffer inside host_buffer is owned by a single tensor_attr object
            // This buffer will not go out of scope until the last object owning it is destroyed (i.e. until the thread is done)
            std::visit([](auto&& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    std::visit(
                        [](auto&& buf) {
                            using buf_type = std::decay_t<decltype(buf)>;
                            if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                                EXPECT_EQ(buf.use_count(), 1);
                            }
                        },
                    storage.buffer);
                }
            }, host_tensor.get_storage());

            Tensor reshaped_tensor = host_tensor.reshape(1, 1, 32, tensor_stop / 32);
            auto device_tensor = reshaped_tensor.to(Layout::TILE).to(device);
            auto thread_local_tensor = device_tensor.cpu().to(Layout::ROW_MAJOR);
            log_info(LogTest, "Worker populating empty host readback_tensor");
            readback_tensor.set_storage(thread_local_tensor.get_storage());
            readback_tensor.set_shape(thread_local_tensor.get_shape());
            readback_tensor.set_dtype(thread_local_tensor.get_dtype());
            readback_tensor.set_layout(thread_local_tensor.get_layout());
            readback_tensor.set_populated();
            // Ensure that this buffer is currently owned by both the thread_local and read_back tensors
            // This is because we explictly pass in the buffer to a new tensor_attr object
            std::visit([](auto&& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    std::visit(
                        [](auto&& buf) {
                            using buf_type = std::decay_t<decltype(buf)>;
                            if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                                EXPECT_EQ(buf.use_count(), 2);
                            }
                        },
                    storage.buffer);
                }
            }, readback_tensor.get_storage());
            log_info(LogTest, "Worker Done");
        });
        // Call deallocate on the tensor in the main thread to ensure that this call is safe
        // i.e.: the tensor should not be deallocated until the thread is done with it
        log_info(LogTest, "Main thread calling deallocate on tensor passed to worker");
        host_tensor.deallocate();
    }
    worker.join();
    log_info(LogTest, "Verifying populated tensor in main thread");
    std::visit(
    [tensor_start, tensor_stop](auto&& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, OwnedStorage>) {
                    std::visit(
                        [tensor_start, tensor_stop](auto&& buf) {
                            using buf_type = std::decay_t<decltype(buf)>;
                            if constexpr (std::is_same_v<buf_type, owned_buffer::Buffer<float>>) {
                                EXPECT_EQ(buf.use_count(), 1);
                                for (int i = tensor_start; i < tensor_stop; i++) {
                                    EXPECT_EQ(buf[i], i);
                                }
                            }
                        },
                    storage.buffer);
                }
            },
        readback_tensor.get_storage());
    EXPECT_EQ(readback_tensor.get_dtype(), DataType::FLOAT32);
    EXPECT_EQ(readback_tensor.get_layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(readback_tensor.get_shape(), ttnn::Shape(Shape({1, 1, 32, tensor_stop / 32})));
}
