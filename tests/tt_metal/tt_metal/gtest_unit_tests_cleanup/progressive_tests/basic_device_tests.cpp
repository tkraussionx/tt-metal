#include <memory>

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
using ::testing::Test;  // GTest test fixture
using ::testing::TestWithParam;  // GTest parametric test fixture
using ::testing::Values;

struct BufferConfig {
    BufferType buf_type;
    u32 buf_size;
    u32 page_size;
    u32 bank_start;
};

class HostDevSweep : public TestWithParam<BufferConfig> {
   protected:
    tt::ARCH arch;
    Device* device;
    BufferConfig buffer_config;

    void SetUp() override {
        this->arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int pci_express_slot = 0;
        this->device = tt::tt_metal::CreateDevice(arch, pci_express_slot);
        tt::tt_metal::InitializeDevice(this->device);
        this->buffer_config = GetParam();
    }

    void TearDown() override { tt::tt_metal::CloseDevice(this->device); }
};

TEST(DevSetup, InitDevice) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    const int pci_express_slot = 0;
    auto device = tt::tt_metal::CreateDevice(arch, pci_express_slot);
    EXPECT_NO_THROW(tt::tt_metal::InitializeDevice(device));
    EXPECT_NO_THROW(tt::tt_metal::CloseDevice(device));
}

TEST_P(HostDevSweep, Loopback) {
    if (this->arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    auto buffer = tt::tt_metal::Buffer(device, this->buffer_config.buf_size, this->buffer_config.bank_start, this->buffer_config.page_size, tt::tt_metal::BufferType::DRAM);
    vector<u32> src_vec(this->buffer_config.buf_size / sizeof(u32), 0);
    EXPECT_NO_THROW(WriteToBuffer(buffer, src_vec));
    vector<u32> result_vec;
    EXPECT_NO_THROW(ReadFromBuffer(buffer, result_vec));
    EXPECT_EQ(result_vec, src_vec);
}

INSTANTIATE_TEST_SUITE_P(HostDevTransfers, HostDevSweep,
                        Values(BufferConfig{.buf_type = BufferType::DRAM, .buf_size = 2048, .page_size = 2048, .bank_start = 0},
                               BufferConfig{.buf_type = BufferType::DRAM, .buf_size = 2048, .page_size = 1024, .bank_start = 0},
                               BufferConfig{.buf_type = BufferType::DRAM, .buf_size = 2048, .page_size = 256, .bank_start = 0},
                               BufferConfig{.buf_type = BufferType::L1, .buf_size = 2048, .page_size = 2048, .bank_start = 0},
                               BufferConfig{.buf_type = BufferType::L1, .buf_size = 4096, .page_size = 4096, .bank_start = 0}),
                               [](const testing::TestParamInfo<HostDevSweep::ParamType>& info) {
                                    auto buffer_config = info.param;
                                    string buffer_type = buffer_config.buf_type == BufferType::DRAM ? "DRAM_Transfer" : "L1_Transfer";
                                    string name =
                                        buffer_type +  "_" +
                                        "Size_" +
                                        std::to_string(buffer_config.buf_size) + "B_" +
                                        "NumBanks_" +
                                        std::to_string(buffer_config.buf_size / buffer_config.page_size) + "_" +
                                        "StartBankId_" +
                                        std::to_string(buffer_config.bank_start);
                                    return name;
                                    });
