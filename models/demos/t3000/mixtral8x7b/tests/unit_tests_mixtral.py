import ttnn
import torch
from models.utility_functions import get_devices_for_t3000

compute_kernel = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


#########
# MATMULS
#########
# kernel duration [ns] = 382266.0, 18% of total device time (happens twice per layer, so total 36%)
# memory BW = 153.6 GB/s
def test_matmul_4096_14336(device):
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, 4096, 14336),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    result = ttnn.matmul(
        x,
        w,
        core_grid=ttnn.CoreGrid(y=7, x=8),
        use_1d_systolic_array=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel,
    )


# kernel duration [ns] = 378742.0, 18% of total device time
# memory BW = 155.04 GB/s
def test_matmul_14336_4096(device):
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 14336),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, 14336, 4096),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    result = ttnn.matmul(
        x,
        w,
        core_grid=ttnn.CoreGrid(y=7, x=8),
        use_1d_systolic_array=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel,
    )


# kernel duration [ns] = 75534.0, 3.6% of total device time
# memory BW = 41.65 GB/s
def test_matmul_4096_768(device):
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, 4096, 768),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    result = ttnn.matmul(
        x,
        w,
        core_grid=ttnn.CoreGrid(y=7, x=8),
        use_1d_systolic_array=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel,
    )


# kernel duration [ns] = 57689.0, 2.7% of total device time
# memory BW = 0.57 GB/s
def test_matmul_4096_8(device):
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, 4096, 8),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    result = ttnn.matmul(
        x,
        w,
        core_grid=ttnn.CoreGrid(y=7, x=8),
        use_1d_systolic_array=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel,
    )


#########
# ALL GATHER
#########
# kernel duration [ns] = 131025.0, 6% of total device time (happens twice per layer, so total 12%)
def test_all_gather(all_devices):
    devices = get_devices_for_t3000(all_devices, 8)
    x = [
        ttnn.from_torch(
            torch.randn(1, 1, 32, 4096),
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for dev in devices
    ]
    x = ttnn.experimental.tensor.all_gather(x, dim=2, num_links=1)


#########
# LAYER NORM
#########
# kernel duration [ns] = 84995.0, 4% of total device time (happens twice per layer, so total 8%)
def test_rmsnorm(device):
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    weight = ttnn.from_torch(
        torch.randn(1, 4096).expand(32, -1),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.rms_norm(x, weight=weight, epsilon=1e-05)


# This method can use 32 cores to compute the layer norm
# kernel duration [ns] = 22581.0 (layernorm) + 3370.0 (interleave to sharded) + 5500.0 (optional sharded to interleave)
def test_rmsnorm_sharded(device):
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    weight = ttnn.from_torch(
        # reshape the norm and use row major layout (this requires bf16 weight)
        torch.randn(1, 1, 1, 4096).reshape([1, 1, -1, 32]),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    shard_height = 32
    hidden_size = 4096
    shard_width_hidden_dim_across_32_cores = hidden_size // 32
    in_mem_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width_hidden_dim_across_32_cores),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    out_mem_config = in_mem_config
    comp_mem_config = ttnn.experimental.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=4,
        block_h=shard_height // 32,
        block_w=shard_width_hidden_dim_across_32_cores // 32,
        inplace=True,
    )

    x = ttnn.experimental.tensor.interleaved_to_sharded(x, sharded_mem_config=in_mem_config)

    x = ttnn.experimental.operations.primary.rmsnorm(
        x,
        1e-05,
        weight,
        program_config=comp_mem_config,
        output_mem_config=out_mem_config,
    )

    x = ttnn.experimental.tensor.sharded_to_interleaved(x)
