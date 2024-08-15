using namespace tt::constants;
using namespace tt::tt_metal;

uint32_t nearest_n(uint32_t x, uint32_t n) {
    return ((x + n - 1) / n) * n;
}

std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<int>>
get_runtime_args(int cur_pos, int num_cores_per_batch, uint32_t k_chunk_size) {
    std::vector<int> all_chunk_assignments;
    all_chunk_assignments.reserve(num_cores_per_batch*2);

    uint32_t valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
    uint32_t pst_value = valid_seq_len / TILE_HEIGHT;
    uint32_t num_chunks_value = valid_seq_len / k_chunk_size;

    int chunks_per_core = num_chunks_value / num_cores_per_batch;
    for (int i = 0; i < num_cores_per_batch; ++i) {
        all_chunk_assignments.push_back((num_cores_per_batch-i-1) * chunks_per_core);
        all_chunk_assignments.push_back((num_cores_per_batch-i) * chunks_per_core);
    }
    all_chunk_assignments[2*num_cores_per_batch] += (num_chunks_value % num_cores_per_batch);

    return {pst_value, num_chunks_value, all_chunk_assignments};
}
