#include "doctest.h"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "constants.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;


namespace op_tests::bmm_op {

struct SingleCoreBMMConfig {
    // tensor size
    uint32_t batch;
    uint32_t channel;
    // tensor decomposition (h and w inferred from decomp config)
    uint32_t a_nblocks_h;
    uint32_t a_nblocks_w;
    uint32_t b_nblocks_w;
    uint32_t a_block_ntiles_h;
    uint32_t a_block_ntiles_w;
    uint32_t b_block_ntiles_w;
    uint32_t out_subblock_ntiles_h;
    uint32_t out_subblock_ntiles_w;
    // tensor types
    DataType a_dtype;
    DataType b_dtype;
    DataType out_dtype;
    // additional operations
    bool tilize_a;
    bool untilize_out;
}; // struct SingleCoreBMMConfig


bool bmm_single_core(Device* device, const SingleCoreBMMConfig& config) {
    bool pass = true;

    if ((config.tilize_a && config.a_dtype != DataType::BFLOAT16)
        || (config.untilize_out && config.out_dtype != DataType::BFLOAT16)) {
        log_info("Skipping invalid case.");
        return true;
    }

    if (config.tilize_a && config.a_dtype != config.out_dtype) {
        log_warn("A known case to debug. Skipping for now.");
        return true;
    }

    uint32_t a_h = config.a_nblocks_h * config.a_block_ntiles_h * constants::TILE_HEIGHT;
    uint32_t a_w = config.a_nblocks_w * config.a_block_ntiles_w * constants::TILE_WIDTH;
    uint32_t b_w = config.b_nblocks_w * config.b_block_ntiles_w * constants::TILE_WIDTH;

    Shape a_shape = { config.batch, config.channel, a_h, a_w };
    Shape b_shape = { config.batch, config.channel, a_w, b_w};  // inner dims ==
    Shape out_shape = { config.batch, config.channel, a_h, b_w };

    Tensor a = Tensor(a_shape, Initialize::RANDOM, config.a_dtype, config.tilize_a ? Layout::ROW_MAJOR : Layout::TILE, device);
    Tensor b = Tensor(b_shape, Initialize::RANDOM, config.b_dtype, Layout::TILE, device);

    Tensor out = bmm_tilize_untilize(a, b, config.out_dtype,
                                     config.a_nblocks_h, config.a_nblocks_w, config.b_nblocks_w,
                                     config.a_block_ntiles_h, config.a_block_ntiles_w, config.b_block_ntiles_w,
                                     config.out_subblock_ntiles_h, config.out_subblock_ntiles_w,
                                     config.tilize_a, config.untilize_out);
    Host *host = GetHost();
    // copy input and output from the device to host
    Tensor a_host = a.to(host);
    Tensor b_host = b.to(host);
    Tensor out_host = out.to(host);

    Tensor& a_ref = config.tilize_a ? a : a.to(Layout::ROW_MAJOR);
    Tensor& b_ref = b;

    // compute golden reference
    // Tensor out_gold = _sgemm

    // pass &= is_close(out_host, out_gold);

    return pass;
}

Tensor& gemm_cpu(const Tensor& a, const Tensor& b, ) {

} // gemm_cpu

// Basic gold batch matmul implementation.
// Returns C=A*B, A and B are row-major untilized
// Accumulates in FP32
inline vector<u16> gold_bmm(const vector<uint32_t> shapeA, const vector<u16>& A, const vector<uint32_t>& shapeB, const vector<u16>& B, bool acc16 = false) {
    TT_ASSERT(shapeB[0] == 1 && shapeA[0] == 1);
    uint32_t nb = shapeA[1];
    TT_ASSERT(shapeB[1] == nb);
    uint32_t M = shapeA[2];
    uint32_t K = shapeA[3];
    TT_ASSERT(shapeB[2] == K);
    uint32_t N = shapeB[3];

    vector<uint32_t> shapeC{1, nb, M, N};
    TensAddr addrC(shapeC);
    TensAddr addrA(shapeA);
    TensAddr addrB(shapeB);
    vector<u16> result(addrC.numel());
    vector<float> resultf(addrC.numel());
    std::fill(resultf.begin(), resultf.end(), 0);

    for (int ib = 0; ib < nb; ib++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    auto offsA = addrA.offs(0, ib, m, k);
                    auto offsB = addrB.offs(0, ib, k, n);
                    auto offsC = addrC.offs(0, ib, m, n);

                    float aa = bfloat16(A[offsA]).to_float();
                    float bb = bfloat16(B[offsB]).to_float();
                    resultf[offsC] += aa * bb;
                    if (acc16)
                        resultf[offsC] = bfloat16(resultf[offsC]).to_float();
                }
            }
        }
    }

    // write back to fp16 after we accumulated in fp32
    for (int ib = 0; ib < nb; ib++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                auto offsC = addrC.offs(0, ib, m, n);
                result[offsC] = bfloat16(resultf[offsC]).to_uint16();
            }
        }
    }

    return result;
}

} // namespace op_tests::bmm_op

TEST_SUITE("BMMTilizeUntilize") {
    using op_tests::bmm_op;

    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "SingleCore") {
        SingleCoreBMMConfig test_config = {};

        SUBCASE("SingleTile") {
            REQUIRE(bmm_single_core(device_, test_config));
        }

        SUBCASE("MultiTile") {
            REQUIRE(bmm_single_core(device_, test_config));
        }
    } // TEST_CASE_FIXTURE
} // TEST_SUITE
