#include "tt_metal/host_api.hpp"
#include "libs/tensor/tensor.hpp"
#include "libs/tt_dnn/op_library/softmax/softmax_op.hpp"
#include <tt_numpy/functions.hpp>

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;


void run_test(Host* host, Device* device, Shape shape) {
    Tensor input = numpy::random::random(shape).to(Layout::TILE).to(device);
    Tensor smax_host = softmax_in_place(input).to(host);
    Tensor input_host = input.to(host);
}

int main(int argc, char **argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    std::tie(arch_name, input_args) = test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    const ARCH arch = get_arch_from_string(arch_name);
    Host *host = GetHost();
    int pci_express_slot = 0;
    Device *device = CreateDevice(arch, pci_express_slot);
    bool pass = true;
    pass &= InitializeDevice(device);

    Shape shape1 = {1, 1, TILE_HEIGHT, TILE_WIDTH};
    run_test(host, device, shape1);

    pass &= CloseDevice(device);
    TT_ASSERT(pass);
    return 0;
}
