#include "tt_dnn/op_library/bert_large_tms/bert_large_tms.hpp"

#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

static const string perf_folder = "/tmp/tt_perf/ops/";

static Profiler op_profiler_split_qkv = Profiler();
static uint32_t call_count_split_qkv = 0;
static const string op_name_split_qkv = "bert_large_split_fused_qkv";

std::vector<Tensor> bert_large_split_fused_qkv(const Tensor& a, const MemoryConfig& mem_config) {
    op_profiler_split_qkv.markStart(op_name_split_qkv);
    op_profiler_split_qkv.setOutputDir(perf_folder + op_name_split_qkv);
    call_count_split_qkv ++;
    string prepend_name = to_string(call_count_split_qkv) + "-MULTI_CORE";

    tt_metal::SetProfilerDir(perf_folder + op_name_split_qkv + "/" + to_string(call_count_split_qkv));

    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 3072})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    std::vector<Tensor> output = multi_core_split_fused_qkv(a, mem_config, compute_and_storage_grid_size);

    op_profiler_split_qkv.markStop(op_name_split_qkv);
    op_profiler_split_qkv.dumpHostResults(prepend_name);
    return output;
}

// Q and V heads use transpose_hw=false, while K head requires the additional transpose with transpose_hw=true.
static Profiler op_profiler_q_head = Profiler();
static uint32_t call_count_q_head = 0;
static const string op_name_q_head = "bert_large_create_q_head";

Tensor bert_large_create_q_head(const Tensor& a, const MemoryConfig& mem_config) {
    op_profiler_q_head.markStart(op_name_q_head);
    op_profiler_q_head.setOutputDir(perf_folder + op_name_q_head);
    call_count_q_head ++;
    string prepend_name = to_string(call_count_q_head) + "-MULTI_CORE";

    tt_metal::SetProfilerDir(perf_folder + op_name_q_head + "/" + to_string(call_count_q_head));

    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    Tensor output = multi_core_create_qkv_heads(a, mem_config, compute_and_storage_grid_size, /*transpose_hw=*/false);

    op_profiler_q_head.markStop(op_name_q_head);
    op_profiler_q_head.dumpHostResults(prepend_name);
    return output;
}

static Profiler op_profiler_k_head = Profiler();
static uint32_t call_count_k_head = 0;
static const string op_name_k_head = "bert_large_create_k_head";

Tensor bert_large_create_k_head(const Tensor& a, const MemoryConfig& mem_config) {
    op_profiler_k_head.markStart(op_name_k_head);
    op_profiler_k_head.setOutputDir(perf_folder + op_name_k_head);
    call_count_k_head ++;
    string prepend_name = to_string(call_count_k_head) + "-MULTI_CORE";

    tt_metal::SetProfilerDir(perf_folder + op_name_k_head + "/" + to_string(call_count_k_head));
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    Tensor output = multi_core_create_qkv_heads(a, mem_config, compute_and_storage_grid_size, /*transpose_hw=*/true);

    op_profiler_k_head.markStop(op_name_k_head);
    op_profiler_k_head.dumpHostResults(prepend_name);
    return output;
}

static Profiler op_profiler_v_head = Profiler();
static uint32_t call_count_v_head = 0;
static const string op_name_v_head = "bert_large_create_v_head";

Tensor bert_large_create_v_head(const Tensor& a, const MemoryConfig& mem_config) {
    op_profiler_v_head.markStart(op_name_v_head);
    op_profiler_v_head.setOutputDir(perf_folder + op_name_v_head);
    call_count_v_head ++;
    string prepend_name = to_string(call_count_v_head) + "-MULTI_CORE";

    tt_metal::SetProfilerDir(perf_folder + op_name_v_head + "/" + to_string(call_count_v_head));
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 1, 384, 1024})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    Tensor output = multi_core_create_qkv_heads(a, mem_config, compute_and_storage_grid_size, /*transpose_hw=*/false);

    op_profiler_v_head.markStop(op_name_v_head);
    op_profiler_v_head.dumpHostResults(prepend_name);
    return output;
}

static Profiler op_profiler_concat_head = Profiler();
static uint32_t call_count_concat_head = 0;
static const string op_name_concat_head = "bert_large_concat_heads";

Tensor bert_large_concat_heads(const Tensor& a, const MemoryConfig& mem_config) {
    op_profiler_concat_head.markStart(op_name_concat_head);
    op_profiler_concat_head.setOutputDir(perf_folder + op_name_concat_head);
    call_count_concat_head ++;
    string prepend_name = to_string(call_count_concat_head) + "-MULTI_CORE";

    tt_metal::SetProfilerDir(perf_folder + op_name_concat_head + "/" + to_string(call_count_concat_head));
    TT_ASSERT((a.shape() == std::array<uint32_t, 4>({9, 16, 384, 64})), "Unsupported input shape");
    CoreCoord compute_and_storage_grid_size = {12, 9};
    auto device_compute_and_storage_grid_size = a.device()->compute_and_storage_grid_size();
    TT_ASSERT((compute_and_storage_grid_size.x <= device_compute_and_storage_grid_size.x && compute_and_storage_grid_size.y <= device_compute_and_storage_grid_size.y), "Unsupported grid shape");
    Tensor output = multi_core_concat_heads(a, mem_config, compute_and_storage_grid_size);

    op_profiler_concat_head.markStop(op_name_concat_head);
    op_profiler_concat_head.dumpHostResults(prepend_name);
    return output;
}

} // namespace tt_metal

} // namespace tt
