#include<iostream>

#include <algorithm>
#include <functional>
#include <random>
#include <chrono>
#include <filesystem>
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
std::ostream& operator<<(std::ostream& os, const tt_xy_pair& xy_pair)
{
    os << "(x=" + std::to_string(xy_pair.x) + ",y=" + std::to_string(xy_pair.y) + ")";
    return os;
}

void single_bank_alloc(Device *device);


int main()
{

    Program program = CreateProgram();
    auto path = std::filesystem::path(__FILE__).parent_path();
    auto reader_path=path/"all_core_reader_kernel.cpp";
    auto writer_path=path/"all_core_writer_kernel.cpp";

    std::cout<<"Kernel Path = "<<path<<std::endl;

    bool pass = true;


    size_t page_size = 1024;
    size_t num_pages = 1024;
    size_t buffer_size =  num_pages*page_size;


    Device *device = CreateDevice(0);
    auto compute_cores = device->compute_with_storage_grid_size();
    auto num_compute_cores = compute_cores.x*compute_cores.y;

    CoreRange all_cores({0,0},{compute_cores.x-1,compute_cores.y-1});

    TT_ASSERT(num_compute_cores==all_cores.size());

    int pages_per_core = num_pages/num_compute_cores;
    int rem_pages = num_pages%num_compute_cores;
    CoreCoord core = {0,0};
    auto reader_kernel = CreateKernel(
        program,
        reader_path,
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    auto writer_kernel = CreateKernel(
        program,
        writer_path,
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );


    try{
    CommandQueue& cq = device->command_queue();


    InterleavedBufferConfig dram_config{
                .device= device,
                .size = buffer_size,
                .page_size = page_size,
                .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    InterleavedBufferConfig l1_config{
                .device= device,
                .size = buffer_size,
                .page_size = page_size,
                .buffer_type = tt::tt_metal::BufferType::L1
    };
    auto input_dram_buffer = CreateBuffer(dram_config);
    auto output_dram_buffer = CreateBuffer(dram_config);
    auto l1_buffer = CreateBuffer(l1_config);


    uint32_t data_cb_index = tt::CB::c_in0;
    tt::DataFormat cb_data_format = tt::DataFormat::Int8;
    auto data_cb_config = CircularBufferConfig(5*page_size,{{data_cb_index,cb_data_format}}).set_page_size(data_cb_index,page_size);
    auto data_cb = CreateCircularBuffer(program,all_cores,data_cb_config);


    std::vector<uint8_t>input_vec(buffer_size);
    srand((unsigned)time(0));
    int i;
    for(int index = 0; index < buffer_size; index++)
    {
        input_vec[index]=(rand()%256);
        // input_vec[index]=(index%256);

    }
    int current_page = 0;
    int index = 0;
    for(auto this_core: all_cores.iterate())
    {
        auto pages_this_core = pages_per_core;
        if(index<rem_pages)
        {
            pages_this_core++;
        }

        SetRuntimeArgs(
            program,
            reader_kernel,
            this_core,
            {
                input_dram_buffer->address(),
                (uint32_t)(input_dram_buffer->noc_coordinates().x),
                (uint32_t)(input_dram_buffer->noc_coordinates().y),
                (uint32_t)page_size,
                (uint32_t)current_page,
                (uint32_t)pages_this_core
            }
        );
        SetRuntimeArgs(
            program,
            writer_kernel,
            this_core,
            {
                output_dram_buffer->address(),
                (uint32_t)(output_dram_buffer->noc_coordinates().x),
                (uint32_t)(output_dram_buffer->noc_coordinates().y),
                (uint32_t)page_size,
                (uint32_t)num_pages - current_page - 1,
                (uint32_t)pages_this_core,
        });

        current_page += pages_this_core;
        index++;

    }

    std::vector<uint8_t> output_vec(l1_buffer->size());
    EnqueueWriteBuffer(cq,input_dram_buffer,input_vec.data(),true);
    EnqueueProgram(cq,program,true);
    EnqueueReadBuffer(cq,output_dram_buffer,output_vec.data(),true);

    EnqueueWriteBuffer(cq,input_dram_buffer,input_vec.data(),true);
    auto start = std::chrono::high_resolution_clock::now();
    EnqueueProgram(cq,program,true);
    auto end = std::chrono::high_resolution_clock::now();

    EnqueueReadBuffer(cq,output_dram_buffer,output_vec.data(),true);
    std::cout<<"Time Taken = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()<<" Microseconds"<<std::endl;
    // std::cout<<"Success = "<<(bool)(input_vec==output_vec)<<std::endl;
    int err_count = 0;
    for(int index = 0; index < input_vec.size(); index++)
    {
        if(input_vec[(output_vec.size()-1-index)]!=output_vec[index])
        {
            pass = false;
            printf("Mismatch @%d  ; %d != %d\n",index,input_vec[(output_vec.size()-1-index)],output_vec[index]);
            err_count++;
            if(err_count>100)
            {
                break;
            }
        }
    }

    }
    catch (const std::exception &e) {
        std::cerr<<"Failed with error "<<e.what()<<std::endl;
        pass = false;
    }
    device->close();
    TT_ASSERT(pass);
}
