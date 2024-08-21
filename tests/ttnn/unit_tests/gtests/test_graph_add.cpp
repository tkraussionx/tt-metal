// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "tt_metal/common/logger.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn_test_fixtures.hpp"

#include "ttnn/tensor/types.hpp"

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"

#include "ttnn/tensor/types.hpp"

#include <cstdint>
#include <string>

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

struct AddOpGraphTestParam {
    ttnn::Shape a_Shape;
    ttnn::Shape b_Shape;
    ttnn::MemoryConfig memory_config; //DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, L1_BLOCK_SHARDED_MEMORY_CONFIG, L1_HEIGHT_SHARDED_MEMORY_CONFIG, L1_WIDTH_SHARDED_MEMORY_CONFIG
    std::vector<std::string> expected_calltrace;
    uint32_t expected_peak_memory_usage = 0;
    uint32_t expected_intermediate_tensors_count = 0;
    uint32_t expected_output_tensors_count = 0;
    uint32_t expected_output_L1_size = 0;
    ttnn::Shape expected_output_shape;
};

class AddOpGraphTestFixture : public TTNNFixtureWithDevice,
                              public testing::WithParamInterface<std::tuple<AddOpGraphTestParam, tt::tt_metal::IGraphProcessor::RunMode>> {};


TEST_P(AddOpGraphTestFixture, AddGraphTrace) {
    auto param_combination = GetParam();
    auto params = std::get<0>(param_combination);
    auto run_mode = std::get<1>(param_combination);

    {
        const auto input_tensor_a = ttnn::zeros(params.a_Shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, this->getDevice(), params.memory_config);
        const auto input_tensor_b = ttnn::zeros(params.b_Shape, ttnn::bfloat16, ttnn::TILE_LAYOUT, this->getDevice(), params.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::add(input_tensor_a, input_tensor_b);
            return output_tensor;
        };

        auto json_trace = graph::query_trace(call);

        tt::log_info("Trace: {}", json_trace.dump(4));

        // Direct calls
        {
            EXPECT_EQ(graph::extract_calltrace(json_trace), params.expected_calltrace);
            EXPECT_EQ(graph::extract_peak_memory_usage(json_trace), params.expected_peak_memory_usage);

            auto [intermediate_tensors_count, output_tensors_count] = graph::count_intermediate_and_output_tensors(json_trace);
            EXPECT_EQ(intermediate_tensors_count, params.expected_intermediate_tensors_count);
            EXPECT_EQ(output_tensors_count, params.expected_output_tensors_count);

            EXPECT_EQ(graph::extract_output_L1_size(json_trace), params.expected_output_L1_size);
        }

        // Query calls
        {
            auto peak_memory_load = graph::query_peak_memory_load(call);
            auto output_L1_size = graph::query_output_L1_size(call);
            auto output_shape = graph::query_output_shape(call);

            EXPECT_EQ(peak_memory_load, params.expected_peak_memory_usage);
            EXPECT_EQ(output_L1_size, params.expected_output_L1_size);
            EXPECT_EQ(output_shape.size(), 1);
            EXPECT_EQ(output_shape[0], params.expected_output_shape);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AddOpGraphTests, // Prefix for the instantiated test suite
    AddOpGraphTestFixture, // Test suite name
    ::testing::Combine(
        ::testing::Values(
            // AddOpGraphTestParam instances for different test cases
            AddOpGraphTestParam{
                .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
                .expected_calltrace = { "ttnn::add", "ttnn::prim::binary", "Device Operation", "create_device_tensor" },
                .expected_peak_memory_usage = 30720,
                .expected_intermediate_tensors_count = 0,
                .expected_output_tensors_count = 1,
                .expected_output_L1_size = 30720,
                .expected_output_shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
            },
            AddOpGraphTestParam{
                .a_Shape = ttnn::Shape(tt::tt_metal::Array4D{4, 3, 32, 32}),
                .b_Shape = ttnn::Shape(tt::tt_metal::Array4D{1, 3, 32, 32}),
                .memory_config = ttnn::L1_MEMORY_CONFIG,
                .expected_calltrace = { "ttnn::add", "ttnn::repeat", "ttnn::prim::old_infra_device_operation", "Device Operation", "create_device_tensor", "ttnn::prim::binary", "Device Operation", "create_device_tensor"},
                .expected_peak_memory_usage = 92160,
                .expected_intermediate_tensors_count = 1,
                .expected_output_tensors_count = 1,
                .expected_output_L1_size = 30720,
                .expected_output_shape = ttnn::Shape(tt::tt_metal::Array4D{4, 3, 32, 32}),
            }
        ),
        ::testing::Values(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH, tt::tt_metal::IGraphProcessor::RunMode::NORMAL)
    )
);


}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
