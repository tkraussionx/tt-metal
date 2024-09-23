#include "../misc/rng.hpp"

void inc_semaphores(uint32_t argval_start, uint32_t semaphore_count, uint32_t inc_count, uint32_t receiver_count, uint32_t &rng_state)
{
    uint64_t semaphore_noc_addrs[receiver_count][semaphore_count];
    for (uint32_t r = 0; r < receiver_count; r++)
    {
        for (uint32_t s = 0; s < semaphore_count; s++)
        {
            const uint32_t semaphore_origin_x = get_arg_val<uint32_t>(argval_start + semaphore_count + 2 * r);
            const uint32_t semaphore_origin_y = get_arg_val<uint32_t>(argval_start + semaphore_count + 2 * r + 1);
            semaphore_noc_addrs[r][s] = get_noc_addr(semaphore_origin_x, semaphore_origin_y, get_semaphore(get_arg_val<uint32_t>(argval_start + s)));
        }
    }

    for (uint32_t s=0; s < semaphore_count; s++)
    {
        for (uint32_t i=0; i < inc_count; i++)
        {
            uint32_t shift = cheap_random(&rng_state, receiver_count);
            for (uint32_t r=0; r < receiver_count; r++)
            {
                uint32_t real_r = r + shift;
                while (real_r >= receiver_count)
                    real_r -= receiver_count;
                noc_semaphore_inc(semaphore_noc_addrs[real_r][s], 1);
            }
        }
    }
}


void kernel_main()
{
    uint32_t i = 0;

    uint32_t inc_count = get_arg_val<uint32_t>(0);
    uint32_t semaphore_count = get_arg_val<uint32_t>(1);
    uint32_t receiver_count = get_arg_val<uint32_t>(2);

    uint32_t debug_buffer_addr = get_arg_val<uint32_t>(2 + semaphore_count + receiver_count);
    uint64_t noc_debug_addr = get_noc_addr(0, 0, debug_buffer_addr);

    uint32_t sender_id = get_arg_val<uint32_t>(2 + semaphore_count + receiver_count + 1);
    uint32_t rng_seed = get_arg_val<uint32_t>(2 + semaphore_count + receiver_count + 2);

    inc_semaphores(3, semaphore_count, inc_count, receiver_count, rng_seed);

    uint32_t local_semaphore_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_semaphore_addr);
    noc_semaphore_set(local_semaphore_ptr, inc_count);

    noc_async_write(local_semaphore_addr, noc_debug_addr, 4);
    noc_async_write_barrier();
}
