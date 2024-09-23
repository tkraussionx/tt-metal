void record_semaphore(uint32_t first_semaphore_id, uint64_t noc_dest_addr, uint32_t semaphore_count)
{
    uint32_t semaphore_addr = get_semaphore(first_semaphore_id);
    noc_async_write(semaphore_addr, noc_dest_addr, L1_ALIGNMENT * semaphore_count);
    noc_async_write_barrier();
}

void wait_for_semaphore(uint32_t semaphore_id, uint32_t num_sender_cores, uint32_t inc_count, uint64_t noc_dest_addr)
{
    uint32_t semaphore_addr = get_semaphore(semaphore_id);
    volatile tt_l1_ptr uint32_t* semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    //noc_semaphore_wait(semaphore_ptr, num_sender_cores * inc_count);
    uint32_t i = 0;
    uint32_t last_value = *semaphore_ptr;
    while (*semaphore_ptr < num_sender_cores * inc_count)
    {
        record_semaphore(semaphore_id, noc_dest_addr, 1);
        if (*semaphore_ptr != last_value)
        {
            last_value = *semaphore_ptr;
            i = 0;
            continue;
        }
        i++;
        if (i > 1000000) // timeout
        {
            break;
        }
    }

}

void kernel_main()
{
    uint32_t num_sender_cores = get_arg_val<uint32_t>(0);
    uint32_t inc_count = get_arg_val<uint32_t>(1);

    uint32_t semaphore_count = get_arg_val<uint32_t>(2);

    uint32_t dest_buffer_addr = get_arg_val<uint32_t>(3 + semaphore_count);
    uint64_t noc_dest_addr = get_noc_addr(0, 0, dest_buffer_addr);

    for (uint32_t i = 0; i < semaphore_count; i++)
    {
        wait_for_semaphore(get_arg_val<uint32_t>(3 + i), num_sender_cores, inc_count, noc_dest_addr);
    }

    record_semaphore(get_arg_val<uint32_t>(3), noc_dest_addr, semaphore_count);
}
