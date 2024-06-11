#include<iostream>
#include<iomanip>
#include<filesystem>
#include<chrono>
// #include "tt_metal/host_api.hpp"
// #include "tt_metal/common/constants.hpp"
// #include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
// #include "tt_metal/common/test_tiles.hpp"
// #include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
// #include "tt_metal/programming_examples/matmul_common/work_split.hpp"
// #include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include "tt_metal/common/tilize_untilize.hpp"

namespace fs = std::filesystem;
void golden_matmul(vector<bfloat16>& a, vector<bfloat16>& b, vector<bfloat16>& output,
                        uint32_t M, uint32_t N, uint32_t K) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j+ (i * N);
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                c_f += float_tmp;
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

void matmul(Device* device, int argc, char** argv);

int main(int argc, char ** argv)
{
    const int device_id = 0;
    Device *device = CreateDevice(device_id);
    try{
        matmul(device, argc, argv);
    }
    catch(const std::exception& e)
    {
        std::cerr<<e.what()<<std::endl;
    }
    CloseDevice(device);
}

void matmul(Device* device, int argc, char** argv)
{
    std::cout<<"Hello World"<<std::endl;
    if(argc < 4)
    {
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("Matmul dimensions: M=%d, N=%d, K=%d\n", M, N, K);


    auto input_a = create_random_vector_of_bfloat16_native(M*K*sizeof(bfloat16), 2.0f, 0, -1);
    auto input_b = create_random_vector_of_bfloat16_native(K*N*sizeof(bfloat16), 2.0f, 2, -1);
    std::vector<bfloat16> output(M*N, 0);
    std::vector<bfloat16> ref_output(M*N, 0);

    // std::vector<bfloat16> input_a(M*K, bfloat16((float)1));
    // std::vector<bfloat16> input_b(K*N, bfloat16((float)1));
    // float index = 0;
    // for(auto&val : input_a)
    // {
    //     val = (index++);
    // }
    // index = 0;
    // for(auto&val : input_b)
    // {
    //     val = (index++);
    // }

    // std::cout<<std::fixed << std::setprecision(2)<<"\nInput A "<<std::endl;
    // for(int i = 0; i< M;i++)
    // {
    //     for(int j = 0; j< K; j++)
    //     {
    //         // input_a[i*K+j] = bfloat16((float)(i%3)-1);
    //         std::cout<<input_a[i*K+j].to_float()<<" ";
    //     }
    //     std::cout<<std::endl;
    // }


    // std::cout<<"\nInput B "<<std::endl;
    // for(int i = 0; i< K;i++)
    // {
    //     for(int j = 0; j< N; j++)
    //     {
    //         // input_b[i*N+j] = bfloat16((float)(j%5)-2);
    //         std::cout<<input_b[i*N+j].to_float()<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    printf("Computing Reference Output\n");
    golden_matmul(input_a, input_b, ref_output, M, N, K);

    // std::cout<<"\nRef Output "<<std::endl;
    // for(int i = 0; i< M;i++)
    // {
    //     for(int j = 0; j< N; j++)
    //     {
    //         std::cout<<ref_output[i*N+j].to_float()<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    printf("Tilizing Input \n");
    tilize(input_a,M,K);
    tilize(input_b,K,N);
    printf("Done Tilizing Input\n");

    // std::cout<<"\nOutput "<<std::endl;
    // for(int i = 0; i< M;i++)
    // {
    //     for(int j = 0; j< K; j++)
    //     {
    //         std::cout<<output[i*K+j].to_float()<<" ";
    //     }
    //     std::cout<<std::endl;
    // }



    const int device_id = 0;
    CommandQueue& command_q = device->command_queue();
    CoreCoord compute_core(0, 0);
    Program program;
    const auto tile_size = sizeof(bfloat16)*tt::constants::TILE_HEIGHT*tt::constants::TILE_WIDTH;
    tt::tt_metal::InterleavedBufferConfig inA_buffer_config{
                .device= device,
                .size = input_a.size() * sizeof(bfloat16),
                .page_size = tile_size,
                .buffer_type = tt::tt_metal::BufferType::L1
    };

    tt::tt_metal::InterleavedBufferConfig inB_buffer_config{
                .device= device,
                .size = input_b.size() * sizeof(bfloat16),
                .page_size = tile_size,
                .buffer_type = tt::tt_metal::BufferType::L1
    };

    tt::tt_metal::InterleavedBufferConfig out_buffer_config{
                .device= device,
                .size = output.size() * sizeof(bfloat16),
                .page_size = tile_size,
                .buffer_type = tt::tt_metal::BufferType::L1
    };

    std::shared_ptr<tt::tt_metal::Buffer> inputA_buffer = CreateBuffer(inA_buffer_config);
    std::shared_ptr<tt::tt_metal::Buffer> inputB_buffer = CreateBuffer(inB_buffer_config);
    std::shared_ptr<tt::tt_metal::Buffer> output_buffer = CreateBuffer(out_buffer_config);

    const int buffer_size_in_tiles = 2;
    std::cout<<"Buffer created @ "<<inputA_buffer->address()<<std::endl;
    auto inputA_CB = CreateCircularBuffer(program,compute_core,CircularBufferConfig(
        buffer_size_in_tiles*tile_size,
        {{tt::CB::c_in0,tt::DataFormat::Float16_b}}
        ).set_page_size(tt::CB::c_in0, tile_size)
    );
    auto inputB_CB = CreateCircularBuffer(program,compute_core,CircularBufferConfig(
        buffer_size_in_tiles*tile_size,
        {{tt::CB::c_in1,tt::DataFormat::Float16_b}}
        ).set_page_size(tt::CB::c_in1, tile_size)
    );
    auto output_CB = CreateCircularBuffer(program,compute_core,CircularBufferConfig(
        buffer_size_in_tiles*tile_size,
        {{tt::CB::c_out0,tt::DataFormat::Float16_b}}
        ).set_page_size(tt::CB::c_out0, tile_size)
    );



    auto root_dir = fs::path(__FILE__).parent_path();

    auto reader_kernel_path = root_dir/"reader_kernel.cpp";
    auto reader_id = tt::tt_metal::CreateKernel(
    program,
    reader_kernel_path.string(),
    compute_core,
    tt::tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {}}
    );

    auto compute_kernel_path = root_dir/"compute_kernel.cpp";
    auto compute_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path.string(),
        compute_core,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {M,K,N}}
    );

    tt::tt_metal::SetRuntimeArgs(program, reader_id, compute_core,
    {
        inputA_buffer->address(),
        inputB_buffer->address(),
        M,
        K,
        N,
        output_buffer->address()
    });


    EnqueueWriteBuffer(command_q, inputA_buffer, input_a.data(), false);
    EnqueueWriteBuffer(command_q, inputB_buffer, input_b.data(), true);

    EnqueueProgram(command_q, program, true);

    auto start = std::chrono::high_resolution_clock::now();
    EnqueueProgram(command_q, program, true);
    auto end = std::chrono::high_resolution_clock::now();

    EnqueueReadBuffer(command_q, output_buffer, output.data(), true);


    untilize(output,M,N);
    std::cout<<"\nOutput "<<std::endl;
    double diff_sum = 0;
    int count = 0;
    for(int i = 0; i< M;i++)
    {
        for(int j = 0; j< N; j++)
        {
            diff_sum += std::abs(output[i*N+j].to_float() - ref_output[i*N+j].to_float());
            count++;
        }
    }

    printf("Average Diff = %f\n", diff_sum/count);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"Kernel Execution Time "<<duration.count()<<std::endl;
}
