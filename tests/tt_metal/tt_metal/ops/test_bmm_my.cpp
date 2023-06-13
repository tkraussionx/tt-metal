#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
// #include "test_gold_impls.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "llrt/tt_debug_print_server.hpp"

using namespace tt;
using namespace tt::tt_metal;


// // Given a tilized data (each tile's data is contiguous and row major within the tile)
// // transform it back to row major full tensor. (This function inverts the tilize() function)
// template <typename T>
// std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
//     TT_ASSERT(rows % 32 == 0);
//     TT_ASSERT(cols % 32 == 0);
//     int num_tiles_r = rows / 32;
//     int num_tiles_c = cols / 32;
//     std::vector<T> result;
//     for(auto r = 0; r < num_tiles_r; r++) {
//         for(auto i = 0; i < 32; i++) {
//             for(auto c = 0; c < num_tiles_c; c++) {
//                 int offset = r * 32 * 32 * num_tiles_c + c * 32 * 32 + i * 32;
//                 for(auto j = 0; j < 32; j++) {
//                     result.push_back(data.at(offset + j));
//                 }
//             }
//         }
//     }

//     return result;
// }

// template <typename T>
// std::vector<T> add_ref(const vector<T>& a, const vector<T>& b) {
//     vector<T> out(a.size());
//     for (uint32_t i = 0; i < a.size(); ++i) {
//         out[i] = a[i] + b[i];
//     }
//     return out;
// }

int main(int argc, char **argv) {
    bool pass = true;

    try {
        const ARCH arch = get_arch_from_string("grayskull");
        Device* device = CreateDevice(arch, 0);
        Host *host = GetHost();

        pass &= InitializeDevice(device);

        array<uint32_t, 4> shapeA = {1, 1, 32, 32};
        array<uint32_t, 4> shapeB = {1, 1, 32, 32};
        array<uint32_t, 4> shapeC = {1, 1, 32, 32};


        int ta = 0, tb = 0, tc = 0;
        if (argc == 4) {
            ta = atoi(argv[1]) == 0 ? 0 : 1;
            tb = atoi(argv[2]) == 0 ? 0 : 1;
            tc = atoi(argv[3]) == 0 ? 0 : 1;
        }
        cout << "ARGS: " << ta << "," << tb << "," << tc << endl;

        DataType typeA = ta == 0 ? DataType::BFLOAT8_B : DataType::BFLOAT16;
        DataType typeB = tb == 0 ? DataType::BFLOAT8_B : DataType::BFLOAT16;
        DataType typeC = tc == 0 ? DataType::BFLOAT8_B : DataType::BFLOAT16;

        // // DataType typeA = DataType::BFLOAT16;
        // DataType typeA = DataType::BFLOAT8_B;
        // // DataType typeB = DataType::BFLOAT16;
        // DataType typeB = DataType::BFLOAT8_B;
        // // DataType typeC = DataType::BFLOAT16;
        // DataType typeC = DataType::BFLOAT8_B;

        auto tta = Tensor(shapeA, Initialize::RANDOM, typeA, Layout::TILE, device);
        auto ttb = Tensor(shapeB, Initialize::RANDOM, typeB, Layout::TILE, device);

        auto out = bmm_test(tta, ttb, typeC);

        auto tta_host = tta.to(host);
        auto ttb_host = ttb.to(host);
        auto out_host = out.to(host);

        // auto out_host = *reinterpret_cast<std::vector<bfloat16>*>(out.data_ptr());
        // {
        //     // Read the result back from device DRAM and ref comparisone
        //     int argfail = -1;
        //     auto comparison_function = [](float a, float b) {
        //         const float rtol = 0.05f; // TODO(AP): need a spec for reference
        //         const float atol = 0.05f;
        //         float maxabs = fmaxf(fabsf(a), fabsf(b));
        //         float absdiff = fabsf(a - b);
        //         auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
        //         return result;
        //     };
        // }

        pass &= tt_metal::CloseDevice(device);

        // std::cout << "AOIJJDHIUBSNNEIOUWNED" << std::endl;

        cout << "Computed Output:" << endl;
        out_host.pretty_print();

        // auto out_ref = my_matmul(untilize(tta.to(host)), untilize(ttb.to(host)));

        // auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);

        cout << "Input A:" << endl;
        tta_host.pretty_print();
        cout << "Input B:" << endl;
        ttb_host.pretty_print();

        // uint32_t sz = sizeof(tta_host) / sizeof(tta.dtype());
        // auto tta_vec = vector<float>(tta_host.buffer(), tta_host.buffer() + sz);
        // auto ttb_vec = vector<float>(ttb_host.buffer(), ttb_host.buffer() + sz);

        // auto out_ref = add_ref(untilize(tta_vec, 32, 32), untilize(ttb_vec, 32, 32));


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
