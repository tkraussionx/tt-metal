import ttnn
import torch


def test_matmul(device, use_program_cache):
    scores_program_config = lambda p: ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.experimental.tensor.CoreCoord(8, 4),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
    )
    q_config = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    k_config = ttnn.create_sharded_memory_config(
        shape=(128, 32),
        core_grid=ttnn.CoreGrid(y=4, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    q_heads_1BQD = ttnn.from_torch(
        torch.randn(1, 32, 32, 128),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )  # memory_config=q_config)
    keys_1BDP = ttnn.from_torch(
        torch.randn(1, 32, 128, 32),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )  # memory_config=k_config)
    results = []
    for _ in range(2):
        attn_1BQP = ttnn.matmul(  # ttnn.experimental.operations.primary.matmul(
            q_heads_1BQD,
            keys_1BDP,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
            core_grid=ttnn.CoreGrid(y=4, x=8),
        )  # program_config = scores_program_config(32))
        results.append(ttnn.to_torch(attn_1BQP))

    assert torch.allclose(results[0], results[1]), "fail"


def test_reshape(device):
    attn_output_1BQd = ttnn.from_torch(
        torch.randn(1, 32, 32, 128), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    attn_output_cat = ttnn.experimental.tensor.reshape(
        attn_output_1BQd, 1, 1, 32, 4096
    )  # ttnn.Shape([1, 1, 32, 4096]))
    print(attn_output_cat.shape)
