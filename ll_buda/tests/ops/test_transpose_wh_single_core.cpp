#include "ll_buda/host_api.hpp"
#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/op_library/transpose/transpose_op.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace ll_buda;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////

Tensor perform_transpose_wh(Tensor& a) {
    TT_ASSERT(a.on_host());
    auto ashape = a.shape();
    TT_ASSERT(ashape.size() == 4);
    auto bshape = ashape;
    bshape[2] = ashape[3];
    bshape[3] = ashape[2];
    TT_ASSERT(a.layout() == tt::ll_buda::Layout::TILE, "This transpose assumes that the data layout is tiled!");
    std::vector<bfloat16> a_vec = *reinterpret_cast<std::vector<bfloat16>*>(a.data_ptr());
    std::vector<bfloat16> b_vec(a_vec.size(), 0);
    auto N = ashape[0];
    auto C = ashape[1];
    auto H = ashape[2];
    auto W = ashape[3];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;
    assert(H%TILE_HEIGHT == 0);
    assert(W%TILE_WIDTH == 0);
    for(auto n = 0; n < N; n++) {
        for(auto c = 0; c < C; c++) {
            for(auto ht = 0; ht < Ht; ht++) {
                for(auto wt = 0; wt < Wt; wt++) {
                    for(auto fh = 0; fh < 2; fh++) {
                        for(auto fw = 0; fw < 2; fw++) {
                            for(auto sth = 0; sth < 16; sth++) {
                                for(auto stw = 0; stw < 16; stw++) {
                                    auto a_index = n*C*H*W + c*H*W + ht*Wt*TILE_HEIGHT*TILE_WIDTH + wt*TILE_HEIGHT*TILE_WIDTH + fh*32*16 + fw*16*16 + sth*16 + stw;
                                    auto b_index = n*C*H*W + c*H*W + wt*Ht*TILE_HEIGHT*TILE_WIDTH + ht*TILE_HEIGHT*TILE_WIDTH + fw*16*32 + fh*16*16 + stw*16 + sth;
                                    b_vec[b_index] = a_vec[a_index];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ll_buda::Tensor b = ll_buda::Tensor(b_vec, bshape, a.dtype(), tt::ll_buda::Layout::TILE);
    return b;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        Device *device = CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
        ll_buda::Host *host = ll_buda::GetHost();

        pass &= InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        std::array<uint32_t, 4> shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = Tensor(shape, Initialize::INCREMENT, DataType::BFLOAT16, Layout::TILE, device);

        ll_buda::Tensor c = ll_buda::transpose(a);

        ll_buda::Tensor d = c.to(host);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Tensor host_a = a.to(host); // Move tensor a to host to validate
        auto host_vec = *reinterpret_cast<std::vector<bfloat16>*>(host_a.data_ptr());
        for(uint32_t i = 0; i < host_vec.size(); i++) {
            std::cout << "host vec at i=" << i << ", =" << host_vec[i] << std::endl;
        }
        auto golden_vec = *reinterpret_cast<std::vector<bfloat16>*>(perform_transpose_wh(host_a).data_ptr());
        auto result_vec = *reinterpret_cast<std::vector<bfloat16>*>(d.data_ptr());
        if(golden_vec != result_vec) {
            assert(golden_vec.size() == result_vec.size());
            for(uint32_t i = 0; i < golden_vec.size(); i++) {
                if(golden_vec[i] != result_vec[i]) {
                    std::cout << "Error at i=" << i << ", golden=" << golden_vec[i]  << ", result=" << result_vec[i] << std::endl;
                }
            }
        }
        pass &= (golden_vec == result_vec);

        pass &= ll_buda::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
