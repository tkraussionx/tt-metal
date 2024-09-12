// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

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
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {

namespace graph {
std::vector<uint32_t> extract_circular_buffer_allocations_per_core(const nlohmann::json& trace) {
    std::vector<uint32_t> circular_buffer_sizes;
    for (const auto& v : trace) {
        if (v["node_type"] == "circular_buffer_allocate") {
            circular_buffer_sizes.emplace_back(std::stoi(v["params"]["size"].get<std::string>()));
        }
    }
    return circular_buffer_sizes;
}
};  // namespace graph

namespace operations {
namespace binary {
namespace test {

struct OperandShapeTestParam {
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

class EltwiseUnaryOpInterfaceTestFixture : public TTNNFixtureWithDevice,
                                           public testing::WithParamInterface<OperandShapeTestParam> {};

TEST_P(EltwiseUnaryOpInterfaceTestFixture, MlirInterfaceTest) {
    auto input = GetParam();

    input.shape = pad_shape_to_tile(input.shape);
    std::cout << "OP = relu(" << input.shape << ")" << std::endl;

    // Run the test
    {
        auto input_tensor =
            ttnn::zeros(input.shape, input.data_type, input.layout, this->getDevice(), input.memory_config);

        auto call = [&] {
            const auto output_tensor = ttnn::relu(input_tensor);
            return output_tensor;
        };

        auto json_trace = graph::query_trace(call);
        // tt::log_info("Trace: {}", json_trace.dump(4));

        auto graph_circular_buffer_allocations = graph::extract_circular_buffer_allocations_per_core(json_trace);

        std::cout << "circular buffers:" << std::endl;
        for (int i = 0; i < graph_circular_buffer_allocations.size(); i++) {
            std::cout << graph_circular_buffer_allocations[i] << std::endl;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MlirInterfaceTests,                  // Prefix for the instantiated test suite
    EltwiseUnaryOpInterfaceTestFixture,  // Test suite name
    ::testing::Values(OperandShapeTestParam{
        .shape = ttnn::Shape(tt::tt_metal::Array4D{3, 1, 32 * 32, 32 * 32}),
        .memory_config =
            {.memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
             .buffer_type = tt::tt_metal::BufferType::L1,
             .shard_spec =
                 tt::tt_metal::ShardSpec{
                     CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 3}}}},
                     {6 * 32, 32 * 32},
                     ShardOrientation::COL_MAJOR}},
    }));

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
