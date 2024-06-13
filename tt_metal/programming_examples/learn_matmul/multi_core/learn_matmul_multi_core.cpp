#include<iostream>
#include<iomanip>
#include<filesystem>
#include<algorithm>
#include<chrono>
#include<time.h>
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
    int device_id = 0;
    if(argc>=5)
    {
        device_id = atoi(argv[4]);
    }
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
    const int TILE_SIZE = 32;
    std::cout<<"Hello World"<<std::endl;
    if(argc < 4)
    {
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    printf("Matmul dimensions: M=%d, N=%d, K=%d\n", M, N, K);



    auto input_a = create_random_vector_of_bfloat16_native(M*K*sizeof(bfloat16), 2.0f, time(NULL)+100, -1);
    auto input_b = create_random_vector_of_bfloat16_native(K*N*sizeof(bfloat16), 2.0f, time(NULL)-100, -1);
    std::vector<bfloat16> output(M*N, 0);
    std::vector<bfloat16> ref_output(M*N, 0);

    printf("Computing Reference Output\n");
    golden_matmul(input_a, input_b, ref_output, M, N, K);

    printf("Tilizing Input \n");
    tilize(input_a,M,K);
    tilize(input_b,K,N);
    printf("Done Tilizing Input\n");


    int Mt = M / TILE_SIZE;
    int Kt = K / TILE_SIZE;
    int Nt = N / TILE_SIZE;

    const auto compute_core_grid = device->compute_with_storage_grid_size();
    printf("Compute Core Grid Size: %d x %d\n", (int)compute_core_grid.x,(int) compute_core_grid.y);
    //Max size
    int core_grid_width = std::min<int>(compute_core_grid.x, Nt);
    int core_grid_height = std::min<int>(compute_core_grid.y, Mt);


    int per_core_Mt = Mt / compute_core_grid.y;
    int rem_Mt = Mt % compute_core_grid.y;
    int max_per_core_Mt = per_core_Mt + (rem_Mt>0);

    int per_core_Nt = Nt / compute_core_grid.x;
    int rem_Nt = Nt % compute_core_grid.x;
    int max_per_core_Nt = per_core_Nt + (rem_Nt>0);

    printf("Active Grid Size: %d x %d, Per Core : %d x %d, Rem : %d x %d\n",  core_grid_height, core_grid_width, per_core_Mt, per_core_Nt, rem_Mt, rem_Nt);


    const int device_id = 0;
    CommandQueue& command_q = device->command_queue();
    CoreCoord compute_core(0, 0);
    CoreRange compute_core_range({0,0},{core_grid_width-1,core_grid_height-1});
    Program program_fetch;
    Program program_compute;
    const auto tile_size = sizeof(bfloat16)*tt::constants::TILE_HEIGHT*tt::constants::TILE_WIDTH;

    tt::tt_metal::InterleavedBufferConfig inA_buffer_config{
                .device= device,
                .size = input_a.size() * sizeof(bfloat16),
                .page_size = tile_size,
                .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    tt::tt_metal::InterleavedBufferConfig inB_buffer_config{
                .device= device,
                .size = input_b.size() * sizeof(bfloat16),
                .page_size = tile_size,
                .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    tt::tt_metal::InterleavedBufferConfig out_buffer_config{
                .device= device,
                .size = output.size() * sizeof(bfloat16),
                .page_size = tile_size,
                .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    const int per_core_inA_size = max_per_core_Mt * Kt * TILE_SIZE * TILE_SIZE * sizeof(bfloat16);
    const int per_core_inB_size = max_per_core_Nt * Kt * TILE_SIZE * TILE_SIZE * sizeof(bfloat16);

    printf("Max Per Core InA Size %d, InB Size %d\n", max_per_core_Mt, max_per_core_Nt);
    tt::tt_metal::InterleavedBufferConfig inA_slice_buf_config{
                .device= device,
                .size = per_core_inA_size,
                .page_size = per_core_inA_size,
                .buffer_type = tt::tt_metal::BufferType::L1
    };

    tt::tt_metal::InterleavedBufferConfig inB_slice_buf_config{
                .device= device,
                .size = per_core_inB_size,
                .page_size = per_core_inB_size,
                .buffer_type = tt::tt_metal::BufferType::L1
    };

    auto inA_slice_buffer = CreateBuffer(inA_slice_buf_config);
    auto inB_slice_buffer = CreateBuffer(inA_slice_buf_config);

    std::cout<<"inA Slice Address "<<inA_slice_buffer->address()<<std::endl;
    std::cout<<"inB Slice Address "<<inB_slice_buffer->address()<<std::endl;

    std::shared_ptr<tt::tt_metal::Buffer> inputA_buffer = CreateBuffer(inA_buffer_config);
    std::shared_ptr<tt::tt_metal::Buffer> inputB_buffer = CreateBuffer(inB_buffer_config);
    std::shared_ptr<tt::tt_metal::Buffer> output_buffer = CreateBuffer(out_buffer_config);

    const int buffer_size_in_tiles = 2;

    auto inputA_CB = CreateCircularBuffer(program_compute,compute_core_range,CircularBufferConfig(
        buffer_size_in_tiles*tile_size,
        {{tt::CB::c_in0,tt::DataFormat::Float16_b}}
        ).set_page_size(tt::CB::c_in0, tile_size)
    );
    auto inputB_CB = CreateCircularBuffer(program_compute,compute_core_range,CircularBufferConfig(
        buffer_size_in_tiles*tile_size,
        {{tt::CB::c_in1,tt::DataFormat::Float16_b}}
        ).set_page_size(tt::CB::c_in1, tile_size)
    );
    auto output_CB = CreateCircularBuffer(program_compute,compute_core_range,CircularBufferConfig(
        buffer_size_in_tiles*tile_size,
        {{tt::CB::c_out0,tt::DataFormat::Float16_b}}
        ).set_page_size(tt::CB::c_out0, tile_size)
    );

    auto root_dir = fs::path(__FILE__).parent_path();

    auto fetch_kernel_path = root_dir/"fetch_inputs_kernel.cpp";
    auto fetch_kernel_id = tt::tt_metal::CreateKernel(
        program_fetch,
        fetch_kernel_path.string(),
        compute_core_range,
        tt::tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = {}}
    );

    auto reader_kernel_path = root_dir/"matmul_mc_dataflow.cpp";
    auto reader_id = tt::tt_metal::CreateKernel(
        program_compute,
        reader_kernel_path.string(),
        compute_core_range,
        tt::tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = {}}
    );

    int start_Mt = 0;
    for(int core_y = 0; core_y < core_grid_height; core_y++)
    {
        int start_Nt = 0;
        int this_core_Mt = per_core_Mt+(core_y<rem_Mt);
        for(int core_x = 0; core_x < core_grid_width; core_x++)
        {
            CoreCoord core(core_x, core_y);
            int this_core_Nt = per_core_Nt+(core_x<rem_Nt);

            tt::tt_metal::SetRuntimeArgs(
                program_fetch, fetch_kernel_id, core,
                {
                    inputA_buffer->address(),
                    inputB_buffer->address(),
                    Mt,
                    Kt,
                    Nt,
                    inA_slice_buffer->address(),
                    inB_slice_buffer->address(),
                    start_Mt,
                    start_Nt,
                    this_core_Mt,
                    this_core_Nt
                }
            );

            tt::tt_metal::SetRuntimeArgs(
                program_compute, reader_id, core,
                {
                    inA_slice_buffer->address(),
                    inB_slice_buffer->address(),
                    Mt,
                    Kt,
                    Nt,
                    output_buffer->address(),
                    start_Mt,
                    start_Nt,
                    this_core_Mt,
                    this_core_Nt,
                    max_per_core_Mt,
                    max_per_core_Nt
                }
            );

            start_Nt += this_core_Nt;
        }
        start_Mt += this_core_Mt;
    }

    auto compute_kernel_path = root_dir/"compute_kernel.cpp";
    auto compute_id = tt::tt_metal::CreateKernel(
        program_compute,
        compute_kernel_path.string(),
        compute_core_range,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {
            Mt,
            Kt,
            Nt,
            max_per_core_Mt,
            max_per_core_Nt
            }
        }
    );

    // tt::tt_metal::SetRuntimeArgs(program, reader_id, compute_core,
    // {
    //     inputA_buffer->address(),
    //     inputB_buffer->address(),
    //     M,
    //     K,
    //     N,
    //     output_buffer->address()
    // });


    EnqueueWriteBuffer(command_q, inputA_buffer, input_a.data(), false);
    EnqueueWriteBuffer(command_q, inputB_buffer, input_b.data(), true);

    EnqueueProgram(command_q, program_fetch, false);
    EnqueueProgram(command_q, program_compute, true);


    auto start = std::chrono::high_resolution_clock::now();
    EnqueueProgram(command_q, program_fetch, false);
    EnqueueProgram(command_q, program_compute, true);

    auto end = std::chrono::high_resolution_clock::now();

    EnqueueReadBuffer(command_q, output_buffer, output.data(), true);


    untilize(output,M,N);

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

    // std::cout<<"\n Output "<<std::endl;
    // for(int i = 0; i< M;i++)
    // {
    //     for(int j = 0; j< N; j++)
    //     {
    //         std::cout<<output[i*N+j].to_float()<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

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
