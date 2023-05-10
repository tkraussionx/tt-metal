#pragma once
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

class AutoPad {
    private:
        inline static Device * device = nullptr;
        AutoPad() {}
    public:
        static void SetDefaultDevice(Device * dev) { device = dev; }
        static Device * GetDefaultDevice() { return device; }

        static Tensor format_input_tensor(const Tensor &a, Device * device, bool pad_c=false, bool pad_n=false, float pad_value=0) {
            auto n = pad_n ? roundup(a.shape()[0], TILE_HEIGHT) : a.shape()[0];
            auto c = pad_c ? roundup(a.shape()[1], TILE_HEIGHT) : a.shape()[1];
            auto h = roundup(a.shape()[2], TILE_HEIGHT);
            auto w = roundup(a.shape()[3], TILE_WIDTH);
            const std::array<uint32_t, 4> padded_shape = {n, c, h, w};

            if (a.layout() != Layout::TILE || a.shape() != padded_shape) {
                auto host = GetHost();
                auto input = a.to(host);
                if (a.shape()!= padded_shape) {
                    if (a.layout() != Layout::ROW_MAJOR) {
                        input = input.to(Layout::ROW_MAJOR);
                    }
                    input = input.pad(padded_shape, {0, 0, 0, 0}, pad_value);
                }
                if(input.layout() != Layout::TILE) {
                    input = input.to(Layout::TILE);
                }
                input = input.to(device);

                delete host;

                return input;

            } else if (a.on_host()) {
                return a.to(device);
            } else {
                return a;
            }
        }

        static Tensor format_output_tensor(const Tensor &a, Tensor &output, const std::array<uint32_t, 4>& shape, Device * device) {

            // Hack env variable to leave outputs on device if no unpadding needed
            if (std::getenv("TT_LEAVE_TILE_OUTPUT_ON_DEVICE") != nullptr) {
                if (output.shape() == shape && output.layout() == Layout::TILE) {
                    return output;
                }
            }
            auto host = GetHost();
            // Unpad output if necessary, result is always on host
            if (output.shape() != shape) {
                output = output.to(host);

                // Requires RM for unpad
                if (output.layout() != Layout::ROW_MAJOR){
                    output = output.to(Layout::ROW_MAJOR);
                }
                output = output.unpad({0, 0, 0, 0}, {shape[0] - 1, shape[1] - 1, shape[2] - 1, shape[3] - 1});
            }

            // Converts layout if necessary, result will always be on host
            if (a.layout() != output.layout()) {
                if (!output.on_host()) {
                    output = output.to(host);
                }

                // Default to RM layout if we can't match the input layout
                if (a.layout() == Layout::TILE && !(output.shape()[2] % TILE_HEIGHT == 0 && output.shape()[3]% TILE_WIDTH == 0)){
                    if (output.layout() != Layout::ROW_MAJOR) {
                        output = output.to(Layout::ROW_MAJOR);
                    }
                } else {
                    output = output.to(a.layout());
                }
            }

            // Send output to device if a was on device
            if (!a.on_host() && output.on_host()) {
                output = output.to(device);
            // Send output to host if a was on host
            } else if (a.on_host() && !output.on_host()) {
                output = output.to(host);
            }

            delete host;

            return output;
        }
};


}
}
