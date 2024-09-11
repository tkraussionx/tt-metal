// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "common/constants.hpp"
#include "common/core_coord.h"
#include "gtest/gtest.h"
#include "impl/event/event.hpp"
#include "impl/program/program.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "third_party/json/json.hpp"
#include "tt_metal/common/logger.hpp"
#include "ttnn/device.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_constraints.hpp"
#include "ttnn/operations/eltwise/binary/binary_l1_interface.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_l1_interface.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/normalization/softmax/softmax_l1_interface.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

struct InputShapeTestParam {
    ttnn::types::Shape shape;
    tt::tt_metal::MemoryConfig memory_config;
    tt::tt_metal::DataType data_type = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;
};

ttnn::Shape pad_shape_to_tile(const ttnn::Shape& shape) {
    std::vector<uint32_t> shape_og;
    std::vector<uint32_t> shape_padded;

    auto rank = shape.rank();
    for (auto i = 0; i < rank; i++) {
        shape_og.push_back(shape[i]);

        if (i >= rank - 2) {
            shape_padded.push_back((shape[i] + 31) / 32 * 32);
        } else {
            shape_padded.push_back(shape[i]);
        }
    }
    return ttnn::Shape(shape_og, shape_padded);
}

void compare_l1_circular_buffer_allocations(
    const std::vector<std::tuple<uint32_t, uint32_t>>& usage_estimator_result, const nlohmann::json& json_trace) {
    auto graph_circular_buffer_allocations = graph::extract_circular_buffer_allocations_per_core(json_trace);
    EXPECT_EQ(usage_estimator_result.size(), graph_circular_buffer_allocations.size());

    for (const auto& [size, cores] : usage_estimator_result) {
        std::cout << size << " ";
    }
    std::cout << std::endl;
    for (int size : graph_circular_buffer_allocations) {
        std::cout << size << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < graph_circular_buffer_allocations.size(); i++) {
        std::cout << "DBG cb[" << i << "]" << std::get<0>(usage_estimator_result[i]) << std::endl
                  << " " << graph_circular_buffer_allocations[i] << std::endl;
        EXPECT_EQ(std::get<0>(usage_estimator_result[i]), graph_circular_buffer_allocations[i]);
    }
}

void compare_l1_tensor_allocations(
    const std::vector<std::tuple<uint32_t, uint32_t>>& usage_estimator_result, const nlohmann::json& json_trace) {
    auto graph_l1_buffer_allocations = graph::extract_l1_buffer_allocations(json_trace);  // total
    EXPECT_EQ(usage_estimator_result.size(), graph_l1_buffer_allocations.size());
    for (int i = 0; i < graph_l1_buffer_allocations.size(); i++) {
        std::cout << "DBG l1[" << i << "]" << std::get<0>(usage_estimator_result[i])
                  << std::endl;  // << " " << graph_l1_buffer_allocations[i] << std::endl;
        EXPECT_EQ(
            std::get<0>(usage_estimator_result[i]) * std::get<1>(usage_estimator_result[i]),
            graph_l1_buffer_allocations[i]);
    }
}

class EltwiseUnaryOpInterfaceTestFixture : public TTNNFixtureWithDevice,
                                           public testing::WithParamInterface<InputShapeTestParam> {};

class EltwiseBinaryOpInterfaceTestFixture
    : public TTNNFixtureWithDevice,
      public testing::WithParamInterface<std::tuple<InputShapeTestParam, InputShapeTestParam>> {};

class SoftmaxOpInterfaceTestFixture : public TTNNFixtureWithDevice,
                                      public testing::WithParamInterface<std::tuple<InputShapeTestParam, int>> {};

class MatmulOpInterfaceTestFixture
    : public TTNNFixtureWithDevice,
      public testing::WithParamInterface<
          std::tuple<InputShapeTestParam, InputShapeTestParam, ttnn::operations::matmul::MatmulProgramConfig>> {};

TEST_P(MatmulOpInterfaceTestFixture, MlirInterfaceTest) {
    auto param_combination = GetParam();
    auto input_a = std::get<0>(param_combination);
    auto input_b = std::get<1>(param_combination);
    auto program_config = std::get<2>(param_combination);

    // pad input shapes (this isn't happening automagically)
    input_a.shape = pad_shape_to_tile(input_a.shape);
    input_b.shape = pad_shape_to_tile(input_b.shape);
    std::cout << "OP = matmul(" << input_a.shape << ", " << input_b.shape << ")" << std::endl;

    // TODO: Test constraints

    // Run the test
    {
        auto input_tensor_a =
            ttnn::zeros(input_a.shape, input_a.data_type, input_a.layout, this->getDevice(), input_a.memory_config);
        auto input_tensor_b =
            ttnn::zeros(input_b.shape, input_b.data_type, input_b.layout, this->getDevice(), input_b.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::matmul(
                input_tensor_a,
                input_tensor_b,
                false /* transpose_a */,
                false /* transpose_b */,
                std::nullopt /* memory_config */,
                std::nullopt /* dtype */,
                program_config);
            return output_tensor;
        };

        // // get graph trace for ground truth
        auto json_trace = graph::query_trace(call);
        tt::log_info("Trace: {}", json_trace.dump(4));

        // L1 interface calls and checks against graph trace
        {
            auto l1_input_a = std::make_tuple(input_a.shape, input_a.data_type, input_a.layout, input_a.memory_config);
            auto l1_input_b = std::make_tuple(input_b.shape, input_b.data_type, input_b.layout, input_b.memory_config);
            // auto l1_usage = SoftmaxOpL1UsageFactory::Make(l1_input, dim_arg);

            const auto cfg = std::get<matmul::MatmulMultiCoreReuseProgramConfig>(program_config);

            uint32_t M = std::get<ttnn::Shape>(l1_input_a).value[-2] / TILE_HEIGHT;
            uint32_t K = std::get<ttnn::Shape>(l1_input_a).value[-1] / TILE_WIDTH;
            uint32_t num_blocks = K / cfg.in0_block_w;
            uint32_t batch_scale_factor = cfg.per_core_M > M ? cfg.per_core_M / M : 1;
            uint32_t per_core_M_per_batch = cfg.per_core_M > M ? M : cfg.per_core_M;

            uint32_t in0_CB_tiles = std::get<tt::tt_metal::MemoryConfig>(l1_input_a).is_sharded()
                                        ? cfg.per_core_M * K
                                        : 2 * per_core_M_per_batch * cfg.in0_block_w;

            const uint32_t cb_in0_size =
                in0_CB_tiles * tt::tt_metal::detail::TileSize(tt::tt_metal::datatype_to_dataformat_converter(
                                   std::get<tt::tt_metal::DataType>(l1_input_a)));

            uint32_t in1_CB_tiles = std::get<tt::tt_metal::MemoryConfig>(l1_input_a).is_sharded()
                                        ? num_blocks * batch_scale_factor * cfg.per_core_N * cfg.in0_block_w
                                        : 2 * cfg.per_core_N * cfg.in0_block_w;

            const uint32_t cb_in1_size =
                in1_CB_tiles * tt::tt_metal::detail::TileSize(tt::tt_metal::datatype_to_dataformat_converter(
                                   std::get<tt::tt_metal::DataType>(l1_input_b)));

            uint32_t out_block_tiles = cfg.per_core_M * cfg.per_core_N;

            const uint32_t cb_out_size =
                out_block_tiles * tt::tt_metal::detail::TileSize(tt::tt_metal::datatype_to_dataformat_converter(
                                      std::get<tt::tt_metal::DataType>(l1_input_a)));

            std::cout << "COMPUTED cb[0] " << cb_in0_size << std::endl;
            std::cout << "COMPUTED cb[1] " << cb_in1_size << std::endl;
            std::cout << "COMPUTED cb[2] " << cb_out_size << std::endl;

            auto graph_circular_buffer_allocations = graph::extract_circular_buffer_allocations_per_core(json_trace);
            for (int i = 0; i < graph_circular_buffer_allocations.size(); i++) {
                std::cout << "DBG cb[" << i << "]" << (graph_circular_buffer_allocations[i]) << std::endl;
            }

            std::cout << "==============================================" << std::endl;

            auto graph_l1_buffer_allocations = graph::extract_l1_buffer_allocations(json_trace);
            for (int i = 0; i < graph_l1_buffer_allocations.size(); i++) {
                std::cout << "DBG l1[" << i << "]" << (graph_l1_buffer_allocations[i]) << std::endl;
            }

            // compare_l1_circular_buffer_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core(),
            // json_trace); compare_l1_tensor_allocations(l1_usage->get_tensor_l1_allocations_per_core(), json_trace);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests,            // Prefix for the instantiated test suite
    MatmulOpInterfaceTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(InputShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 4 * 32, 8 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }),

        ::testing::Values(InputShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 8 * 32, 4 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }),

        ::testing::Values(ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
            .compute_with_storage_grid_size = CoreCoord(2, 2),
            .in0_block_w = 4,
            .out_subblock_h = 2,
            .out_subblock_w = 2,
            .per_core_M = 2,
            .per_core_N = 4}))

);

TEST_P(SoftmaxOpInterfaceTestFixture, MlirInterfaceTest) {
    auto param_combination = GetParam();
    auto input = std::get<0>(param_combination);
    auto dim_arg = std::get<1>(param_combination);

    // pad input shapes (this isn't happening automagically)
    input.shape = pad_shape_to_tile(input.shape);
    std::cout << "OP = softmax(" << input.shape << ", dim=" << dim_arg << ")" << std::endl;

    // TODO: Test constraints

    // Run the test
    {
        auto input_tensor =
            ttnn::zeros(input.shape, input.data_type, input.layout, this->getDevice(), input.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::softmax(input_tensor, dim_arg);
            return output_tensor;
        };

        // // get graph trace for ground truth
        auto json_trace = graph::query_trace(call);
        tt::log_info("Trace: {}", json_trace.dump(4));

        // L1 interface calls and checks against graph trace
        {
            auto l1_input = std::make_tuple(input.shape, input.data_type, input.layout, input.memory_config);
            auto l1_usage = SoftmaxOpL1UsageFactory::Make(l1_input, dim_arg);

            compare_l1_circular_buffer_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core(), json_trace);
            compare_l1_tensor_allocations(l1_usage->get_tensor_l1_allocations_per_core(), json_trace);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests,             // Prefix for the instantiated test suite
    SoftmaxOpInterfaceTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
                .layout = ttnn::ROW_MAJOR_LAYOUT},
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                             {6 * 32, 32 * 32},
                             ShardOrientation::COL_MAJOR}},
            }),

        ::testing::Values(-1))

);

TEST_P(EltwiseUnaryOpInterfaceTestFixture, MlirInterfaceTest) {
    auto input = GetParam();

    // pad input shapes (this isn't happening automagically)
    input.shape = pad_shape_to_tile(input.shape);
    std::cout << "OP = relu(" << input.shape << ")" << std::endl;

    // TODO: Test constraints

    // Run the test
    {
        auto input_tensor =
            ttnn::zeros(input.shape, input.data_type, input.layout, this->getDevice(), input.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::relu(input_tensor);
            return output_tensor;
        };

        // // get graph trace for ground truth
        auto json_trace = graph::query_trace(call);
        tt::log_info("Trace: {}", json_trace.dump(4));

        // L1 interface calls and checks against graph trace
        {
            auto l1_input = std::make_tuple(input.shape, input.data_type, input.layout, input.memory_config);
            auto l1_usage = UnaryOpL1UsageFactory::Make(l1_input);

            compare_l1_circular_buffer_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core(), json_trace);
            compare_l1_tensor_allocations(l1_usage->get_tensor_l1_allocations_per_core(), json_trace);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests,                  // Prefix for the instantiated test suite
    EltwiseUnaryOpInterfaceTestFixture,  // Test suite name
    ::testing::Values(
        InputShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
            .memory_config =
                {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                 .buffer_type = tt::tt_metal::BufferType::L1,
                 .shard_spec =
                     tt::tt_metal::ShardSpec{
                         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                         {6 * 32, 32 * 32},
                         ShardOrientation::COL_MAJOR}},
        },
        InputShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 7 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        },
        InputShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
            .memory_config =
                {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                 .buffer_type = tt::tt_metal::BufferType::L1,
                 .shard_spec =
                     tt::tt_metal::ShardSpec{
                         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                         {6 * 32, 32 * 32},
                         ShardOrientation::COL_MAJOR}},
            .layout = tt::tt_metal::Layout::ROW_MAJOR,
        })

);

TEST_P(EltwiseBinaryOpInterfaceTestFixture, MlirInterfaceTest) {
    auto param_combination = GetParam();
    auto input_a = std::get<0>(param_combination);
    auto input_b = std::get<1>(param_combination);

    // pad input shapes (this isn't happening automagically)
    input_a.shape = pad_shape_to_tile(input_a.shape);
    input_b.shape = pad_shape_to_tile(input_b.shape);
    std::cout << "OP = " << input_a.shape << " + " << input_b.shape << std::endl;

    // Check input params against op constraints
    try {
        std::unique_ptr<EltwiseOpConstraintsBuilder> builder = EltwiseOpConstraintsFactory::Make(
            input_a.shape, input_a.memory_config, input_b.shape, input_b.memory_config);
        if (builder) {
            const auto op_constraints =
                (*builder)
                    .setBufferTypeA(input_a.memory_config.buffer_type)
                    .setBufferTypeB(input_b.memory_config.buffer_type)
                    .setBufferTypeO(
                        input_a.memory_config.buffer_type)  // assuming output buffer type is the same as input_a
                    .setDataTypeA(input_b.data_type)
                    .setDataTypeB(input_b.data_type)
                    .setDataTypeO(input_a.data_type)  // assuming output data type is the same as input_a
                    .setIsShardedA(input_a.memory_config.is_sharded())
                    .setIsShardedB(input_b.memory_config.is_sharded())
                    .setIsShardedO(
                        input_a.memory_config.is_sharded())  // assuming output is sharded if input_a is sharded
                    .build_constraints();
            std::cout << "size(op_contraints) =  " << op_constraints.size() << std::endl;

            if (op_constraints.size() == 0) {
                std::cout << "op_constraints is empty" << std::endl;
                GTEST_SKIP();
            }
        } else {
            std::cout << "builder is nullptr" << std::endl;
            GTEST_SKIP();
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        GTEST_FAIL();
    }

    // Run the test
    {
        auto input_tensor_a =
            ttnn::zeros(input_a.shape, input_a.data_type, input_a.layout, this->getDevice(), input_a.memory_config);
        auto input_tensor_b =
            ttnn::zeros(input_b.shape, input_b.data_type, input_b.layout, this->getDevice(), input_b.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b);
            return output_tensor;
        };

        // // get graph trace for ground truth
        auto json_trace = graph::query_trace(call);
        tt::log_info("Trace: {}", json_trace.dump(4));

        // L1 interface calls and checks against graph trace
        {
            auto l1_input_a = std::make_tuple(input_a.shape, input_a.data_type, input_a.layout, input_a.memory_config);
            auto l1_input_b = std::make_tuple(input_b.shape, input_b.data_type, input_b.layout, input_b.memory_config);
            auto l1_usage = EltwiseOpL1UsageFactory::Make(l1_input_a, l1_input_b);

            compare_l1_circular_buffer_allocations(l1_usage->get_circular_buffer_l1_allocations_per_core(), json_trace);
            compare_l1_tensor_allocations(l1_usage->get_tensor_l1_allocations_per_core(), json_trace);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTestsREPEAT_MAX_BLOCK_SCALE,  // Prefix for the instantiated test suite
    EltwiseBinaryOpInterfaceTestFixture,       // Test suite name
    ::testing::Combine(
        ::testing::Values(InputShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
            .memory_config =
                {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                 .buffer_type = tt::tt_metal::BufferType::L1,
                 .shard_spec =
                     tt::tt_metal::ShardSpec{
                         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                         {6 * 32, 32 * 32},
                         ShardOrientation::COL_MAJOR}},
        }),

        ::testing::Values(InputShapeTestParam{
            .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32 * 32, 32 * 32}),
            .memory_config = ttnn::L1_MEMORY_CONFIG,
        }))

);

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests,                   // Prefix for the instantiated test suite
    EltwiseBinaryOpInterfaceTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32, 32 * 64}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {5 * 32, 160},
                             ShardOrientation::COL_MAJOR}},
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 5 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }),

        ::testing::Values(
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32, 32 * 64}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {5 * 32, 160},
                             ShardOrientation::COL_MAJOR}},
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
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 5 * 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{4, 2, 5 * 32, 1}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
            }))

);

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTestsHEIGHT_SHARDED,     // Prefix for the instantiated test suite
    EltwiseBinaryOpInterfaceTestFixture,  // Test suite name
    ::testing::Combine(
        ::testing::Values(
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            }),

        ::testing::Values(
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 5, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            },
            InputShapeTestParam{
                .shape = ttnn::Shape(tt::tt_metal::Array4D{1, 1, 32 * 64, 32}),
                .memory_config =
                    {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                     .buffer_type = tt::tt_metal::BufferType::L1,
                     .shard_spec =
                         tt::tt_metal::ShardSpec{
                             CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}}},
                             {160, 32},
                             ShardOrientation::COL_MAJOR}},
            }

            ))

);

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
