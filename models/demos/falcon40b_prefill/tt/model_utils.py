import tt_lib as ttl
import ttnn


def memcfg_1d_width_sharded_from_tensor_shape(shape, grid=ttnn.CoreGrid(8, 8)):
    start_core_coord = ttl.tensor.CoreCoord(0, 0)
    end_core_coord = ttl.tensor.CoreCoord(grid.x - 1, grid.y - 1)
    assert shape[3] % (grid.x * grid.y) == 0, f"Tensor width must be divisible by the number of cores"
    shard_width = int(shape[3] / (grid.x * grid.y))
    shard_height = int(shape[0] * shape[1] * shape[2])
    return ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        start_core_coord,
                        end_core_coord,
                    ),
                }
            ),
            [
                shard_height,
                shard_width,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )


def matmul_1d_config_from_tensor_shapes(in0_shape, in1_shape, grid=ttnn.CoreGrid(8, 8), act=None):
    m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
    return matmul_1d_config(m, k, n, grid, act)


def matmul_1d_config(m, k, n, grid=ttnn.CoreGrid(8, 8), act=None):
    tile_width = 32
    tile_height = 32

    per_core_m = m // tile_height
    per_core_k = k // tile_width // grid.num_cores
    per_core_n = n // tile_width // grid.num_cores

    # find the largest value between 1 and 8 that is a factor of per_core_n
    # e.g. if per_core_n is 14, then out_subblock_w = 7
    out_subblock_w = max([i for i in range(1, 9) if per_core_n % i == 0])

    # find the largest value that is a factor of per_core_m such that
    # out_subblock_w * out_subblock_h <= 8
    out_subblock_h = max([i for i in range(1, 9) if per_core_m % i == 0 and i * out_subblock_w <= 8])

    print(
        f"per_core_m: {per_core_m}, per_core_k: {per_core_k}, per_core_n: {per_core_n}, out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}"
    )

    return ttnn.ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttl.tensor.CoreCoord(grid.x, grid.y),
        in0_block_w=per_core_k,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )
