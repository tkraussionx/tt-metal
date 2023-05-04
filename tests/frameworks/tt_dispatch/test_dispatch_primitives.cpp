#include "frameworks/tt_dispatch/impl/dispatch.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;

int main() {
    constexpr int pci_express_slot = 0;
    tt::tt_metal::Device *device =
        CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    InitializeDevice(device);

    CommandQueue queue;
}
