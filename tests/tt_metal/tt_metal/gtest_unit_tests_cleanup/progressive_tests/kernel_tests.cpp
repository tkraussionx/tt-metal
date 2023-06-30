#include <memory>

#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
using ::testing::Test;  // GTest test fixture
using ::testing::TestWithParam;  // GTest parametric test fixture
using ::testing::Values;

class KernelCompile : public Test {
   protected:
    tt::ARCH arch;
    Device* device;

    void SetUp() override {
        this->arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        const int pci_express_slot = 0;
        this->device = tt::tt_metal::CreateDevice(arch, pci_express_slot);
        tt::tt_metal::InitializeDevice(this->device);
    }

    void TearDown() override { tt::tt_metal::CloseDevice(this->device); }
};


TEST_F(KernelCompile, CompileBlankKernelsOnAllProcessors) {
    tt::tt_metal::Program program = tt::tt_metal::Program();
    EXPECT_TRUE(tt::tt_metal::CompileProgram(this->device, program));
}
