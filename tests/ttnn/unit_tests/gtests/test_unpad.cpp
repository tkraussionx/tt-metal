// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <tt_metal/detail/persistent_kernel_cache.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>
#include <ttnn/operations/eltwise/ternary/where.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <vector>

#include "common/bfloat16.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace tt {
namespace tt_metal {
namespace test {

struct UnpadParam {
    float scalar;
    uint32_t h;
    uint32_t w;
};

static tt::tt_metal::Tensor make_random_tensor(tt::tt_metal::Shape s) {
    static int seed = 42;
    using namespace ttnn::operations::experimental::auto_format;
    auto b = tt::tt_metal::owned_buffer::create(
        create_random_vector_of_bfloat16_native(s[0] * s[1] * s[2] * s[3] * 2, 2, seed++, -1));
    tt::tt_metal::Tensor t(
        OwnedStorage{std::move(b)}, s, tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR);
    return ttnn::tilize_with_zero_padding(t.to(AutoFormat::GetDefaultDevice()));
}

float dump_first_tile_of_tensor(tt::tt_metal::Tensor tensor) {
    using namespace ttnn::operations::experimental::auto_format;
    std::cout << "dump_first_tile_of_tensor" << std::endl;
    assert(tensor.dtype() == tt::tt_metal::DataType::BFLOAT16);
    auto t = tensor;
    if (t.storage_type() == tt::tt_metal::StorageType::DEVICE) {
        std::cout << "To CPU " << std::endl;
        t = t.cpu();
    }
    if (t.layout() != tt::tt_metal::Layout::ROW_MAJOR) {
        std::cout << "To ROW" << std::endl;
        t = t.to(tt::tt_metal::Layout::ROW_MAJOR);
    }

    std::cout << "Copy to device" << std::endl;
    t = t.to(AutoFormat::GetDefaultDevice());

    std::vector<::bfloat16> buf(1024);

    memcpy(buf.data(), t);

    const auto shape = t.get_shape();
    const auto dim = shape.rank();
    const auto width = shape[-1];
    const auto height = shape[-2];
    const auto device_width = t.get_legacy_shape()[-1];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << buf[y * device_width + x].to_float() << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // This works
    t = t.cpu();
    std::cout << t.write_to_string() << std::endl;

    // return first item from second row to validate against
    return buf[1 * device_width + 0].to_float();

    // And  This works,
    // auto storage = std::get<tt::tt_metal::OwnedStorage>(t.storage());
    // auto buf2 = std::get<tt::tt_metal::owned_buffer::Buffer<::bfloat16>>(storage.get_buffer());
    //  auto ps = t.shape().with_tile_padding();

    // for (int y = 0; y < ps[2]; y++) {
    //     for (int x = 0; x < ps[3]; x++) {
    //         std::cout << buf2[y * ps[3] + x].to_float() << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
}

class VerifyUnpadFixture : public ttnn::TTNNFixture, public testing::WithParamInterface<UnpadParam> {};

TEST_P(VerifyUnpadFixture, VerifyUnpad) {
    auto param = GetParam();
    const auto device_id = 0;
    auto& device = ttnn::open_device(device_id);
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);

    using namespace ttnn::operations::experimental::auto_format;

    {
        AutoFormat::SetDefaultDevice(&device);
        ttnn::enable_program_cache(device);
        tt::tt_metal::detail::EnablePersistentKernelCache();

        auto a = make_random_tensor({1, 1, 10, 10});

        tt::tt_metal::Shape start(std::vector<uint32_t>{0, 0, 0, 0});
        tt::tt_metal::Shape end(std::vector<uint32_t>{0, 0, 5, 5});
        auto b = a.cpu().to(tt::tt_metal::Layout::ROW_MAJOR).unpad(start, end);

        std::cout << "A:";
        auto expected = dump_first_tile_of_tensor(a);
        std::cout << "B:\n";
        auto actual = dump_first_tile_of_tensor(b);

        TT_FATAL(expected == actual);
    }
    ttnn::close_device(device);
}

INSTANTIATE_TEST_SUITE_P(Add1DTensorAndScalarTests, VerifyUnpadFixture, ::testing::Values(UnpadParam{3.0f, 32, 64}));

}  // namespace test
}  // namespace tt_metal
}  // namespace tt
