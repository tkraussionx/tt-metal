// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/constants.hpp"
#include "gtest/gtest.h"
#include "impl/event/event.hpp"
#include "impl/program/program.hpp"
#include "tt_metal/common/logger.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_constraints.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

#include "ttnn/tensor/types.hpp"

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"

#include "ttnn/tensor/types.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

// uint32_t calculate_buffer_size_per_tile(const tt::tt_metal::MemoryConfig& memory_config)
// {

// }

enum class mlir_enabled_op
{
    add,
    subtract,
    multiply,
    matmul,
    reduce_sum,
    softmax,
};

// typedef ttnn::Tensor op_one_input(const ttnn::Tensor& input_tensor);
// typedef ttnn::Tensor op_two_inputs(const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b);
// typedef ttnn::Tensor op_three_inputs(const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b, const ttnn::Tensor& input_tensor_c);

struct InputShapeTestParam {
    ttnn::types::Shape shape;
    tt::tt_metal::MemoryConfig memory_config;
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;
};

struct AddOpGraphTestParam {
    std::vector<std::string> expected_calltrace;
    uint32_t expected_peak_L1_memory_usage = 0;
    uint32_t expected_intermediate_tensors_count = 0;
    std::vector<graph::TensorInfo> expected_output_info;
    // mlir_enabled_op mlir_op;
    // std::function<ttnn::Tensor()> op;
};

class AddOpGraphTestFixture : public TTNNFixtureWithDevice,
                              public testing::WithParamInterface<std::tuple<InputShapeTestParam, InputShapeTestParam, AddOpGraphTestParam, tt::tt_metal::IGraphProcessor::RunMode>> {};


TEST_P(AddOpGraphTestFixture, AddGraphTrace) {
    auto param_combination = GetParam();
    auto input_a = std::get<0>(param_combination);
    auto input_b = std::get<1>(param_combination);
    auto params = std::get<2>(param_combination);
    auto run_mode = std::get<3>(param_combination);

    // pad shapes
    auto pad_shape_to_tile = [] (const ttnn::Shape& shape) {
        std::vector<uint32_t> shape_og;
        std::vector<uint32_t> shape_padded;

        auto rank = shape.rank();
        for (auto i = 0; i < rank; i++) {
            shape_og.push_back(shape[i]);

            if (i >= rank - 2)
            {
                shape_padded.push_back((shape[i] + 31) / 32 * 32);
            } else {
                shape_padded.push_back(shape[i]);
            }
        }
        return ttnn::Shape(shape_og, shape_padded);
    };
    input_a.shape = pad_shape_to_tile(input_a.shape);
    input_b.shape = pad_shape_to_tile(input_b.shape);

    // constexpr EltwiseOpConstraintsBuilder constraints_builder = EltwiseOpConstraintsBuilder();

    try {
        std::unique_ptr<EltwiseOpConstraintsBuilder> builder = EltwiseOpConstraintsFactory::Make(input_a.shape, input_a.memory_config, input_b.shape, input_b.memory_config);
        if (builder)
        {
            const auto op_constraints = (*builder)
                .setBufferTypeA(input_a.memory_config.buffer_type)
                .setBufferTypeB(input_b.memory_config.buffer_type)
                .setBufferTypeO(input_a.memory_config.buffer_type) // assuming output buffer type is the same as input_a
                .setDataTypeA(input_b.data_type)
                .setDataTypeB(input_b.data_type)
                .setDataTypeO(input_a.data_type) // assuming output data type is the same as input_a
                .setIsShardedA(input_a.memory_config.is_sharded())
                .setIsShardedB(input_b.memory_config.is_sharded())
                .setIsShardedO(input_a.memory_config.is_sharded()) // assuming output is sharded if input_a is sharded
                .build_constraints();
            std::cout << "size(op_contraints) =  " << op_constraints.size() << std::endl;

            if (op_constraints.size() == 0)
            {
                std::cout << "op_constraints is empty" << std::endl;
                GTEST_SKIP();
            }
        }
        else {
            std::cout << "builder is nullptr" << std::endl;
            GTEST_SKIP();
        }
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        GTEST_FAIL();
    }

    // for (const auto& constraint : op_constraints) {
    //     std::cout << "("
    //         << (int)std::get<0>(constraint) << ", "
    //         // << std::get<1>(constraint) << ", "
    //         // << std::get<2>(constraint) << ", "
    //         // << std::get<3>(constraint) << ", "
    //         // << std::get<4>(constraint) << ", "
    //         // << std::get<5>(constraint) << ", "
    //         // << std::get<6>(constraint) << ", "
    //         // << std::get<7>(constraint) << ", "
    //         // << std::get<8>(constraint) << ", "
    //         // << std::get<9>(constraint) << ", "
    //         // << std::get<10>(constraint) << ", "
    //         // << std::get<11>(constraint) << ", "
    //         // << std::get<12>(constraint) << ", "
    //         // << std::get<13>(constraint) << ", "
    //         // << std::get<14>(constraint) << ", "
    //         // << std::get<15>(constraint) << ", "
    //         // << std::get<16>(constraint) << ", "
    //         // << std::get<17>(constraint) << ")"
    //         << std::endl;
    // }

    // // === mockup
    // auto get_number_of_circular_buffers = [](const mlir_enabled_op& op)
    // {
    //     switch (op)
    //     {
    //     case mlir_enabled_op::add:
    //         return 3;
    //     case mlir_enabled_op::subtract:
    //         return 3;
    //     case mlir_enabled_op::multiply:
    //         return 3;
    //     case mlir_enabled_op::matmul:
    //         return 3;
    //     case mlir_enabled_op::reduce_sum:
    //         return 2;
    //     case mlir_enabled_op::softmax:
    //         return 2;
    //     default:
    //         // TT_THROW("Unsupported mlir op");;
    //         return 0;
    //     }
    // };

    // auto get_number_of_l1_allocations_buffers = [](const mlir_enabled_op& op)
    // {
    //     switch (op)
    //     {
    //     case mlir_enabled_op::add:
    //         return 1;
    //     case mlir_enabled_op::subtract:
    //         return 1;
    //     case mlir_enabled_op::multiply:
    //         return 1;
    //     case mlir_enabled_op::matmul:
    //         return 1;
    //     case mlir_enabled_op::reduce_sum:
    //         return 1;
    //     case mlir_enabled_op::softmax:
    //         return 1;
    //     default:
    //         // TT_THROW("Unsupported mlir op");;
    //         return 0;
    //     }
    // };

    // auto get_number_of_buffers = [](const mlir_enabled_op& op)
    // {
    //     switch (op)
    //     {
    //     case mlir_enabled_op::add:
    //         return 3;
    //     case mlir_enabled_op::subtract:
    //         return 3;
    //     case mlir_enabled_op::multiply:
    //         return 3;
    //     case mlir_enabled_op::matmul:
    //         return 3;
    //     case mlir_enabled_op::reduce_sum:
    //         return 2;
    //     case mlir_enabled_op::softmax:
    //         return 2;
    //     default:
    //         // TT_THROW("Unsupported mlir op");;
    //         return 0;
    //     }
    // };

    // std::cout << __cplusplus << "\n";

    // auto calculate_circular_buffer_l1_size = [](
    //     const tt::tt_metal::DataType& dtype,
    //     const tt::tt_metal::Layout& layout,
    //     const tt::tt_metal::MemoryConfig& memory_config,
    //     const uint32_t& total_size_bytes, // could we drop this one?
    //     const Shape& shape,
    //     bool is_intermediate = false) {
    //     auto page_size = tt::tt_metal::tensor_impl::get_page_size(dtype, layout, total_size_bytes, shape.value);
    //     auto num_pages = 2;
    //     if (memory_config.is_sharded())
    //     {
    //         num_pages = memory_config.shard_spec.value().shape[0] * memory_config.shard_spec.value().shape[1] / TILE_HEIGHT / TILE_WIDTH;
    //         // if (is_intermediate)
    //         // {
    //         //    num_pages = memory_config.shard_spec.value().shape[0] * memory_config.shard_spec.value().shape[1] / TILE_HEIGHT / TILE_WIDTH;
    //         // }
    //         // else
    //         // {
    //         //     num_pages = 1;
    //         // }
    //     };

    //     return page_size * num_pages;
    // };

    // auto calculate_buffer_size_per_l1_core = [](
    //     // const tt::tt_metal::Shape& shape,
    //     const Shape& shape,
    //     const tt::tt_metal::Layout& layout,
    //     const tt::tt_metal::DataType data_type,
    //     // const ShardSpecBuffer& shard_params,
    //     const MemoryConfig& memory_config) {

    //     if (memory_config.is_l1())
    //     {
    //         if (memory_config.is_sharded()) {
    //             tt::tt_metal::ShardSpecBuffer shard_spec_buffer(
    //                 memory_config.shard_spec.value().grid,
    //                 memory_config.shard_spec.value().shape,
    //                 memory_config.shard_spec.value().orientation,
    //                 memory_config.shard_spec.value().halo,
    //                 {32, 32}, // fake, unused in validate_sharded_buffer_allocation
    //                 {32, 32} // fake, unused in validate_sharded_buffer_allocation
    //             );
    //             tt::tt_metal::tensor_impl::validate_sharded_buffer_allocation(shape.value, layout, data_type, shard_spec_buffer, memory_config);

    //             auto total_size_bytes = shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
    //             auto num_of_cores = memory_config.shard_spec.value().grid.num_cores();
    //             return total_size_bytes / num_of_cores;
    //         } else {
    //             //  Storage cores are cores dedicated usage as L1 space, no compute/kernels run on them
    //             //  There is 120KB of reserved space on worker cores
    //             //
    //             //  GS (10x12 grid, 9x12 grid of workers)
    //             //      10 Storage cores, each with 2 banks
    //             //      108 Worker cores each with 1 bank
    //             //      Banks are ½ size of L1 (512KB)
    //             //  WH
    //             //      Nebula_x1 (9x8 grid, 8x8 grid of workers)
    //             //          No storage cores
    //             //          64 Worker cores each with 1 bank
    //             //          Banks are almost full size of L1 (1344KB)
    //             //      Nebula_x2 (8x8 grid, 7x8 grid of workers)
    //             //          4 Storage cores
    //             //          56 Worker cores each with 1 bank
    //             //          Banks are ½ size of L1 (732KB)
    //             auto total_size_bytes = shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(data_type);
    //             auto num_of_cores = 64; // Nebula_x1
    //             return total_size_bytes / num_of_cores;
    //         }
    //     }
    //     return (uint32_t)0; // dram
    // };

    // auto get_num_of_cores = [](const std::optional<tt::tt_metal::ShardSpec>& shard_spec) {
    //     if (shard_spec.has_value()) {
    //         return shard_spec.value().grid.num_cores();
    //     }
    //     return (uint32_t)64; // Nebula_x1
    // };

    // auto calculate_buffer_size = [calculate_buffer_size_per_l1_core, get_num_of_cores](InputShapeTestParam& param) {
    //     auto total_size_bytes = param.shape.with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(param.data_type);
    //     auto calculated_size = calculate_buffer_size_per_l1_core(param.shape, param.layout, param.data_type, param.memory_config);

    //     // debug print
    //     std::cout << " :: " << param.shape  << " calculated_size_total " << calculated_size * get_num_of_cores(param.memory_config.shard_spec) << std::endl;

    //     return calculated_size;
    // };

    // auto get_larger_shape_by_volume = [] (const ttnn::Shape& a, const ttnn::Shape& b) {
    //     if (a.volume() > b.volume()) {
    //         return a;
    //     }
    //     else {
    //         return b;
    //     }
    // };

    // // batch broadcasts have internal tensor buffers
    // auto is_batch_broadcast = [](const ttnn::Shape& a, const ttnn::Shape& b) {
    //     // return a != b;
    //     if ((a.rank() == 4) && (b.rank() == 4)) {
    //         if (a[0] != b[0]) {
    //             return true;
    //         }
    //     }
    //     return false;
    // };
    //  std::function<ttnn::Tensor(const ttnn::Tensor&)> unary_lambdas[] = {
    //     [](ttnn::Tensor& input) { return ttnn::softmax(input); },
    //     [](ttnn::Tensor& input) { return ttnn::sum(input, 1); },
    //  };


    // if(is_binary_op_valid(input_a.shape, input_a.memory_config, input_b.shape, input_b.memory_config) == false)
    // {
    //     GTEST_SKIP();
    // }

    // = sandbox test

    // auto shape1 = ttnn::Shape(std::array<uint32_t, 2>{14,32});
    // std::cout << "DBG 1 " << shape1 << " " << shape1.with_tile_padding() << std::endl;

    // auto shape2 = ttnn::Shape(std::array<uint32_t, 3>{1, 14, 28}, std::array<uint32_t, 3>{1, 32,32});
    // std::cout << "DBG 2 " << shape2 << " " << shape2.with_tile_padding() << std::endl;

    // auto shape3 = ttnn::Shape(std::array<uint32_t, 4>{2, 5, 14, 28}, std::array<uint32_t, 4>{1, 1, 32,32});
    // std::cout << "DBG 3 " << shape3 << " " << shape3.with_tile_padding() << std::endl;


    // auto shape4 = ttnn::Shape(std::array<uint32_t, 4>{2, 5, 14, 28}, std::array<uint32_t, 4>{2, 5, 32, 32});
    // std::cout << "DBG 4 " << shape4 << " " << shape4.with_tile_padding() << std::endl;

    // auto shape5 = ttnn::Shape(std::array<uint32_t, 2>{14, 28});
    // std::cout << "DBG 5 " << shape5 << " " << pad_shape_to_tile(shape5) << std::endl;

    // auto shape6 = ttnn::Shape(std::array<uint32_t, 3>{5, 14, 28});
    // std::cout << "DBG 6 " << shape6 << " " << pad_shape_to_tile(shape6) << std::endl;

    // auto shape7 = ttnn::Shape(std::array<uint32_t, 4>{2, 5, 14, 28});
    // std::cout << "DBG 7 " << shape7 << " " << pad_shape_to_tile(shape7) << std::endl;
    // std::cout << "DBG 7 " << shape7.with_tile_padding().volume() << " " << pad_shape_to_tile(shape7).with_tile_padding().volume() << std::endl;



    // === end of mockup
    {
        auto input_tensor_a = ttnn::zeros(pad_shape_to_tile(input_a.shape), input_a.data_type, input_a.layout, this->getDevice(), input_a.memory_config);
        auto input_tensor_b = ttnn::zeros(pad_shape_to_tile(input_b.shape), input_b.data_type, input_b.layout, this->getDevice(), input_b.memory_config);
        auto call = [&] {
            const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b);
            return output_tensor;
        };

        auto json_trace = graph::query_trace(call);

        // tt::log_info("Trace: {}", json_trace.dump(4));

        std::cout << "OP = " << pad_shape_to_tile(input_a.shape) << " + " << pad_shape_to_tile(input_b.shape) << std::endl;


        // new interface test
        {
            auto l1_input_a = std::make_tuple(input_a.shape, input_a.data_type, input_a.layout, input_a.memory_config);
            auto l1_input_b = std::make_tuple(input_b.shape, input_b.data_type, input_b.layout, input_b.memory_config);

            auto l1_usage = EltwiseOpL1UsageFactory::Make(l1_input_a, l1_input_b);

            auto l1_cb_usage = l1_usage->get_circular_buffer_l1_allocations_per_core();
            auto graph_circular_buffer_allocations = graph::extract_circular_buffer_allocations_per_core(json_trace);
            EXPECT_EQ(l1_cb_usage.size(), graph_circular_buffer_allocations.size());
            for (int i = 0; i < l1_cb_usage.size(); i++) {
                std::cout << "DBG cb[" << i << "]" << std::get<0>(l1_cb_usage[i]) << " " << graph_circular_buffer_allocations[i] << std::endl;
                EXPECT_EQ(std::get<0>(l1_cb_usage[i]), graph_circular_buffer_allocations[i]);
            }

            auto l1_tensor_usage = l1_usage->get_tensor_l1_allocations_per_core(); // what about output tensor allocation?
            auto graph_l1_buffer_allocations = graph::extract_l1_buffer_allocations(json_trace); // total
            EXPECT_EQ(l1_tensor_usage.size(), graph_l1_buffer_allocations.size());
            for (int i = 0; i < l1_tensor_usage.size(); i++) {
                std::cout << "DBG l1[" << i << "]" << std::get<0>(l1_tensor_usage[i]) << " " << graph_l1_buffer_allocations[i] << std::endl;
                EXPECT_EQ(std::get<0>(l1_tensor_usage[i]) * std::get<1>(l1_tensor_usage[i]), graph_l1_buffer_allocations[i]);
            }
        }

        // // mockup test
        // { // circular_buffers
        //     auto circular_buffers = graph::extract_circular_buffer_allocations_per_core(json_trace);
        //     EXPECT_EQ(circular_buffers.size(), get_number_of_circular_buffers(mlir_enabled_op::add) + is_batch_broadcast(input_a.shape, input_b.shape));

        //     // unfolded implementation
        //     // check input 0
        //     {
        //         auto total_size_bytes = pad_shape_to_tile(input_a.shape).with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(input_a.data_type);
        //         auto calculated_size = calculate_circular_buffer_l1_size(input_a.data_type, input_a.layout, input_a.memory_config, total_size_bytes, pad_shape_to_tile(input_a.shape));
        //         EXPECT_EQ(circular_buffers[0], calculated_size);
        //     }

        //     // check input 1
        //     {
        //         auto total_size_bytes = pad_shape_to_tile(input_b.shape).with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(input_b.data_type);
        //         auto calculated_size = calculate_circular_buffer_l1_size(input_b.data_type, input_b.layout, input_b.memory_config, total_size_bytes, pad_shape_to_tile(input_b.shape));
        //         EXPECT_EQ(circular_buffers[1], calculated_size);
        //     }

        //     auto  get_output_circular_buffer_size = [] (std::vector<uint32_t>& circular_buffers) {
        //         if (circular_buffers.size() == 4) {
        //             return circular_buffers[3];
        //         }
        //         else if (circular_buffers.size() == 3) {
        //             return circular_buffers[2];
        //         }
        //         else {
        //             // failed earlier
        //         }
        //         return (uint32_t)0;
        //     };

        //     auto get_intermediate_circular_buffer_size = [] (std::vector<uint32_t>& circular_buffers) {
        //         if (circular_buffers.size() == 4) {
        //             return std::make_optional<uint32_t>(circular_buffers[2]);
        //         }
        //         std::optional<uint32_t> opt = std::nullopt; // opt is empty
        //         return opt;
        //     };

        //     {   // check intermediate
        //         auto size = get_intermediate_circular_buffer_size(circular_buffers);
        //         if (size.has_value()) {
        //             auto intermediate = (input_a.shape[0] > input_b.shape[0]) ? input_b : input_a;
        //             EXPECT_EQ(intermediate.shape.rank(), 4); // my implementation limitation

        //             auto batch_size = (input_a.shape[0] > input_b.shape[0]) ? input_a.shape[0] : input_b.shape[0];;
        //             vector<uint32_t> new_shape;
        //             new_shape.push_back(batch_size);
        //             for (int i = 1; i < 4; i++) {
        //                 new_shape.push_back(intermediate.shape[i]);
        //             }

        //             intermediate.shape = ttnn::Shape{tt::tt_metal::Shape{new_shape, tt::tt_metal::Padding{intermediate.shape.rank()}}};

        //             auto total_size_bytes = pad_shape_to_tile(intermediate.shape).volume() * tt::tt_metal::tensor_impl::element_size_bytes(intermediate.data_type);
        //             auto calculated_size = calculate_circular_buffer_l1_size(intermediate.data_type, intermediate.layout, intermediate.memory_config, total_size_bytes, pad_shape_to_tile(intermediate.shape));

        //             // debug print
        //             std::cout << input_a.shape << " " << input_b.shape << " " << intermediate.shape << std::endl;
        //             std::cout << "::" << intermediate.shape << " size " << size.value() << " calculated_size" << calculated_size << std::endl;

        //             EXPECT_EQ(size.value(), calculated_size);
        //         }
        //     }

        //     {   // check output
        //         auto size = get_output_circular_buffer_size(circular_buffers);
        //         InputShapeTestParam output = input_a;
        //         output.shape =  pad_shape_to_tile(get_larger_shape_by_volume(input_a.shape, input_b.shape));

        //         std::cout << "1)" << output.shape.volume() << std::endl;
        //         std::cout << "2)" << pad_shape_to_tile(output.shape).volume() << std::endl;
        //         std::cout << "3)" << pad_shape_to_tile(output.shape).with_tile_padding().volume() << std::endl;


        //         auto total_size_bytes = pad_shape_to_tile(output.shape).with_tile_padding().volume() * tt::tt_metal::tensor_impl::element_size_bytes(output.data_type);
        //         auto calculated_size = calculate_circular_buffer_l1_size(output.data_type, output.layout, output.memory_config, total_size_bytes, pad_shape_to_tile(output.shape));

        //         EXPECT_EQ(size, calculated_size);
        //     }

        //     // for (const auto& size : circular_buffers) {
        //     //     auto total_size_bytes = pad_shape_to_tile(input_a.shape).volume() * tt::tt_metal::tensor_impl::element_size_bytes(input_a.data_type);
        //     //     auto calculated_size = calculate_circular_buffer_l1_size(input_a.data_type, input_a.layout, input_a.memory_config, total_size_bytes, pad_shape_to_tile(input_a.shape));

        //     //     EXPECT_EQ(size, calculated_size);
        //     // }
        // }

        // {   // check l1 buffers
        //     auto l1_buffers = graph::extract_l1_buffer_allocations(json_trace);

        //     EXPECT_EQ(l1_buffers.size(), get_number_of_l1_allocations_buffers(mlir_enabled_op::add) + is_batch_broadcast(input_a.shape, input_b.shape));
        //     EXPECT_LE(l1_buffers.size(), 2);


        //     auto  get_output_buffer_size = [] (std::vector<uint32_t>& l1_buffers) {
        //         if (l1_buffers.size() == 1) {
        //             return l1_buffers[0];
        //         }
        //         else {
        //             return l1_buffers[1];
        //         }
        //     };

        //     auto get_intermediate_buffer_size = [] (std::vector<uint32_t>& l1_buffers) {
        //         if (l1_buffers.size() == 2) {
        //             return std::make_optional<uint32_t>(l1_buffers[0]);
        //         }
        //         std::optional<uint32_t> opt = std::nullopt; // opt is empty
        //         return opt;
        //         // else {
        //         //     return std::make_optional<uint32_t>();
        //         // }
        //     };

        //     {
        //         auto intermediate_buffer_size = get_intermediate_buffer_size(l1_buffers);
        //         if (intermediate_buffer_size.has_value()) {

        //             EXPECT_EQ(is_batch_broadcast(input_a.shape, input_b.shape), true); // currently implemented, to figure out if there are other uses for intermediate buffer in eltwise ops
        //             // take everything but batch size from the smaller buffer
        //             InputShapeTestParam intermediate = (input_a.shape[0] > input_b.shape[0]) ? input_b : input_a;
        //             EXPECT_EQ(intermediate.shape.rank(), 4); // my implementation limitation

        //             auto batch_size = (input_a.shape[0] > input_b.shape[0]) ? input_a.shape[0] : input_b.shape[0];;
        //             vector<uint32_t> new_shape;
        //             new_shape.push_back(batch_size);
        //             for (int i = 1; i < 4; i++) {
        //                 new_shape.push_back(intermediate.shape[i]);
        //             }

        //             intermediate.shape = ttnn::Shape{tt::tt_metal::Shape{new_shape, tt::tt_metal::Padding{intermediate.shape.rank()}}};

        //             auto calculated_size = calculate_buffer_size_per_l1_core(intermediate.shape, intermediate.layout, intermediate.data_type, intermediate.memory_config);

        //             // debug print
        //             std::cout << input_a.shape << " " << input_b.shape << " " << intermediate.shape << std::endl;
        //             std::cout << "::" << intermediate.shape << " size " << intermediate_buffer_size.value() << " calculated_size" << calculated_size * get_num_of_cores(intermediate.memory_config.shard_spec) << std::endl;

        //             EXPECT_EQ(intermediate_buffer_size.value(), calculated_size * get_num_of_cores(intermediate.memory_config.shard_spec));
        //         }
        //     }

        //     {
        //         auto output_buffer_size = get_output_buffer_size(l1_buffers);
        //         InputShapeTestParam output = input_a;
        //         output.shape =  pad_shape_to_tile(get_larger_shape_by_volume(input_a.shape, input_b.shape));
        //         auto calculated_size = calculate_buffer_size_per_l1_core(output.shape, output.layout, output.data_type, output.memory_config);

        //         // debug print
        //         std::cout << pad_shape_to_tile(input_a.shape) << " " << pad_shape_to_tile(input_b.shape) << " " << pad_shape_to_tile(output.shape) << std::endl;
        //         std::cout << "::" << pad_shape_to_tile(output.shape) << " size " << output_buffer_size << " calculated_size" << calculated_size * get_num_of_cores(output.memory_config.shard_spec) << std::endl;
        //         EXPECT_EQ(output_buffer_size, calculated_size * get_num_of_cores(output.memory_config.shard_spec));
        //     }
        // }
        // auto calltrace = graph::extract_calltrace(json_trace);
        // for (const auto& trace : calltrace) {
        //     std::cout << trace << std::endl;
        // }

        // // Direct calls
        // {
        //     EXPECT_EQ(graph::extract_calltrace(json_trace), params.expected_calltrace);
        //     EXPECT_EQ(graph::extract_peak_L1_memory_usage(json_trace), params.expected_peak_L1_memory_usage);
        //     EXPECT_EQ(graph::extract_output_tensors(json_trace).size(), params.expected_output_info.size());

        //     auto [intermediate_tensors_count, output_tensors_count] = graph::count_intermediate_and_output_tensors(json_trace);
        //     EXPECT_EQ(intermediate_tensors_count, params.expected_intermediate_tensors_count);
        //     EXPECT_EQ(output_tensors_count, params.expected_output_info.size());
        // }

        // // Query calls
        // {
        //     auto peak_L1_memory_usage = graph::query_peak_L1_memory_usage(call);
        //     auto output_info = graph::query_output_info(call);

        //     EXPECT_EQ(peak_L1_memory_usage, params.expected_peak_L1_memory_usage);


        //     if(output_info.size() != params.expected_output_info.size()) {
        //         auto print = [](const auto& infos){
        //             for (const auto& info : infos) {
        //                 tt::log_info("{}", info);
        //             }
        //         };

        //         tt::log_info("Output info size mismatch. Expected {} but got {}", params.expected_output_info.size(), output_info.size());

        //         tt::log_info("Expected output info:");
        //         print(params.expected_output_info);

        //         tt::log_info("Actual output info:");
        //         print(output_info);
        //         ASSERT_TRUE(false);
        //     }

        //     for (int i = 0; i < output_info.size(); ++i) {
        //         EXPECT_EQ(output_info[i], params.expected_output_info[i]);
        //     }
        // }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AddOpGraphTests, // Prefix for the instantiated test suite
    AddOpGraphTestFixture, // Test suite name
    ::testing::Combine(
        ::testing::Values(
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32*64, 32}),
                .memory_config =
                {
                    .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                    .buffer_type = tt::tt_metal::BufferType::L1,
                    .shard_spec = tt::tt_metal::ShardSpec{
                        CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                        {160, 32},
                        ShardOrientation::COL_MAJOR
                        }
                },
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }
        ),

        ::testing::Values(
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32*64, 32}),
                .memory_config =
                {
                    .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                    .buffer_type = tt::tt_metal::BufferType::L1,
                    .shard_spec = tt::tt_metal::ShardSpec{
                        CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                        {160, 32},
                        ShardOrientation::COL_MAJOR
                        }
                },
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 4, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 4, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 1, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }
        ),

        ::testing::Values(
            // AddOpGraphTestParam instances for different test cases
            AddOpGraphTestParam{
                .expected_calltrace = { "ttnn::add", "ttnn::prim::binary", "Device Operation", "create_device_tensor" },
                .expected_peak_L1_memory_usage = 3 * /*tensor size*/ (4 * 32 * 64 / 4 * 32 * 2) + 3 * /* cb size */ (32 * 32 * 2),
                .expected_intermediate_tensors_count = 0,
                .expected_output_info = {
                    graph::TensorInfo{
                        .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 4, 32*64/4, 32}),
                        .size = 4 * 32 * 64 / 4 * 32 * 2,
                        .type = tt::tt_metal::BufferType::L1}},
                // .mlir_op = mlir_enabled_op::add,
                // .op = ttnn::add,
            }
        ),
        // ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH, tt::tt_metal::IGraphProcessor::RunMode::NORMAL)
        ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH)
    )
);


}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
