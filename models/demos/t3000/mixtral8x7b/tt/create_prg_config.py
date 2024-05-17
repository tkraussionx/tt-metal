import ttnn
import math
from typing import Optional, Tuple

MatmulDefaultProgramConfig = ttnn.experimental.operations.primary.MatmulDefaultProgramConfig
MatmulMultiCoreReuseProgramConfig = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig
MatmulMultiCoreReuseMultiCastProgramConfig = (
    ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig
)
MatmulMultiCoreReuseMultiCast1DProgramConfig = (
    ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig
)

# MatmulProgramConfig is the Union of the above types
MatmulProgramConfig = ttnn.experimental.operations.primary.MatmulProgramConfig


_DST_SUB_BLOCKS = [
    (2, 4),
    (4, 2),
    (1, 8),
    (8, 1),
    (1, 7),
    (7, 1),
    (2, 3),
    (3, 2),
    (1, 6),
    (6, 1),
    (1, 5),
    (5, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 3),
    (3, 1),
    (1, 2),
    (2, 1),
    (1, 1),
]

_FP32_DST_SUB_BLOCKS = [x for x in _DST_SUB_BLOCKS if x[0] * x[1] <= 4]


def _get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dst):
    candidate_sub_blocks = _FP32_DST_SUB_BLOCKS if fp32_dst else _DST_SUB_BLOCKS
    for m_subblock_size, n_subblock_size in candidate_sub_blocks:
        if m_tiles_per_core % m_subblock_size == 0 and n_tiles_per_core % n_subblock_size == 0:
            return m_subblock_size, n_subblock_size
    raise RuntimeError(
        f"Unable to find subblock sizes for m_size={m_tiles_per_core} and n_size={n_tiles_per_core} (fp32_dst={fp32_dst})"
    )


def get_fused_activation(activation):
    if activation is None:
        return None
    return ttnn._tt_lib.tensor.string_to_unary_with_param(activation)


def create_matmul_1d_systolic_array_program_config(
    *,
    input_shape_a: Tuple[int, ...],
    input_shape_b: Tuple[int, ...],
    core_grid: Optional[ttnn.CoreGrid] = None,
    activation: Optional[str] = None,
    fp32_dst: Optional[bool] = False,
):
    """
    Create a MatmulMultiCoreReuseMultiCast1DProgramConfig for a 1D systolic array.
    Args:
        * :attr:`input_shape_a` (Tuple[int, ...]): the shape of the first tensor
        * :attr:`input_shape_b` (Tuple[int, ...]): the shape of the second tensor
        * :attr:`core_grid` (ttnn.CoreGrid): the maximum core grid to use
        * :attr:`activation` (Optional[str]): the activation function to use. Defaults to None
    """

    if core_grid is None:
        raise RuntimeError(f"core_grid must be a valid CoreGrid object")

    if core_grid is not None and not isinstance(core_grid, ttnn.CoreGrid):
        raise RuntimeError(f"core_grid must be a valid CoreGrid object")

    *batch_shape_a, m_size, k_size = input_shape_a.with_tile_padding()
    *batch_shape_b, _, n_size = input_shape_b.with_tile_padding()
    if math.prod(batch_shape_b) != 1:
        raise RuntimeError("Second input cannot be currently batched when running matmul using 1d systolic array")

    batch_size = math.prod(batch_shape_a)
    input_b_is_batched = math.prod(batch_shape_b) > 1

    if input_b_is_batched:
        raise RuntimeError(f"Input b shouldn't be batched")

    if (batch_size * m_size) % ttnn.TILE_SIZE != 0 or k_size % ttnn.TILE_SIZE != 0 or n_size % ttnn.TILE_SIZE != 0:
        raise RuntimeError(
            f"The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of {ttnn.TILE_SIZE}"
        )

    batch_and_m_tiles = (batch_size * m_size) // ttnn.TILE_SIZE
    k_tiles = k_size // ttnn.TILE_SIZE
    n_tiles = n_size // ttnn.TILE_SIZE

    is_tall = batch_and_m_tiles > n_tiles
    is_wide = not is_tall
    # Tall output
    if is_tall:
        batch_and_m_tiles_per_core = int(math.ceil(batch_and_m_tiles / core_grid.num_cores))
        k_tiles_per_core = int(math.ceil(k_tiles / core_grid.num_cores))
        n_tiles_per_core = n_tiles

    # Wide output
    else:
        batch_and_m_tiles_per_core = batch_and_m_tiles
        k_tiles_per_core = int(math.ceil(k_tiles / core_grid.num_cores))
        n_tiles_per_core = int(math.ceil(n_tiles / core_grid.num_cores))

    while k_tiles % k_tiles_per_core != 0:
        k_tiles_per_core -= 1

    m_subblock_size, n_subblock_size = _get_subblock_sizes(batch_and_m_tiles_per_core, n_tiles_per_core, fp32_dst)

    return MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=k_tiles_per_core,
        out_subblock_h=m_subblock_size,
        out_subblock_w=n_subblock_size,
        per_core_M=batch_and_m_tiles_per_core,
        per_core_N=n_tiles_per_core,
        fuse_batch=True,
        fused_activation=get_fused_activation(activation=activation),
        mcast_in0=is_wide,
    )


def create_matmul_program_config(
    *, input_tensor_a, input_tensor_b, core_grid, activation, use_1d_systolic_array, compute_kernel_config
):
    *batch_shape_a, m_size, k_size = input_tensor_a.shape.with_tile_padding()
    *batch_shape_b, _, n_size = input_tensor_b.shape.with_tile_padding()
    *_, intended_k_size_of_a = input_tensor_a.shape
    *_, intended_k_size_of_b, _ = input_tensor_b.shape

    if intended_k_size_of_a != intended_k_size_of_b:
        raise RuntimeError(f"The k dimension does not match between tensors")

    batch_size = math.prod(batch_shape_a)
    input_b_is_batched = math.prod(batch_shape_b) > 1

    input_tensor_a_memory_config = ttnn.get_memory_config(input_tensor_a)
    input_tensor_b_memory_config = ttnn.get_memory_config(input_tensor_b)

    # Determine how many subblock tiles we can use based on dest register data format
    fp32_dst = (
        compute_kernel_config
        and isinstance(compute_kernel_config, ttnn.WormholeComputeKernelConfig)
        and compute_kernel_config.fp32_dest_acc_en
    )

    if use_1d_systolic_array is None and not input_b_is_batched:
        # Infer use_1d_systolic_array based on how rectangular the output matrix
        height_width_ratio = (math.prod(batch_shape_a) * m_size) / n_size
        if height_width_ratio < 1:
            height_width_ratio = 1 / height_width_ratio

        # 8 is an arbitrary choice. It should probably be inferred based on the device core grid
        threshold_of_being_rectangular = 8
        is_more_rectangular_than_square = height_width_ratio > threshold_of_being_rectangular
        use_1d_systolic_array = is_more_rectangular_than_square

    if use_1d_systolic_array:
        return create_matmul_1d_systolic_array_program_config(
            input_shape_a=input_tensor_a.shape,
            input_shape_b=input_tensor_b.shape,
            core_grid=core_grid,
            activation=activation,
            fp32_dst=fp32_dst,
        )

    # TODO: clean up the code below by mvoing it to separate create_*_config functions

    if (batch_size * m_size) % ttnn.TILE_SIZE != 0 or k_size % ttnn.TILE_SIZE != 0 or n_size % ttnn.TILE_SIZE != 0:
        raise RuntimeError(
            f"The last two dimensions of the first tensor and the last dimension of the second tensor must be a multiple of {ttnn.TILE_SIZE}"
        )

    if input_b_is_batched:
        if activation is not None:
            raise RuntimeError(f"Cannot use activation with batched input b")
        if (not ttnn.is_sharded(input_tensor_a)) and (not ttnn.is_sharded(input_tensor_b)):
            m_tiles_per_core = int(math.ceil((m_size / ttnn.TILE_SIZE)))
            n_tiles_per_core = int(math.ceil((n_size / ttnn.TILE_SIZE)))
            k_tiles_per_core = 1  # TODO(arakhmati): Can it be more than 1 without running out of memory?
        elif ttnn.is_sharded(input_tensor_a):
            if input_tensor_a_memory_config.memory_layout == ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                raise RuntimeError(f"MatmulMultiCoreReuseProgramConfig: Cannot be width sharded")
            shard_shape = input_tensor_a_memory_config.shard_spec.shape
            N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
            m_tiles_per_core = shard_shape[0] // ttnn.TILE_SIZE
            n_tiles_per_core = N
            k_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE
        elif ttnn.is_sharded(input_tensor_b):
            if input_tensor_b_memory_config.memory_layout == ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED:
                raise RuntimeError(f"MatmulMultiCoreReuseProgramConfig: Cannot be width sharded")
            shard_shape = input_tensor_b_memory_config.shard_spec.shape
            m_tiles_per_core = int(math.ceil((m_size / ttnn.TILE_SIZE)))
            n_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE
            k_tiles_per_core = 1

        m_subblock_size, n_subblock_size = _get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dst)

        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            per_core_M=m_tiles_per_core,
            per_core_N=n_tiles_per_core,
            in0_block_w=k_tiles_per_core,
            out_subblock_h=m_subblock_size,
            out_subblock_w=n_subblock_size,
        )
    else:
        if not ttnn.is_sharded(input_tensor_a):
            m_tiles_per_core = int(math.ceil(((batch_size * m_size) / ttnn.TILE_SIZE) / core_grid.y))
            n_tiles_per_core = int(math.ceil(n_size / ttnn.TILE_SIZE / core_grid.x))
            k_tiles_per_core = 4  # TODO(arakhmati): What is a good starting point?
            while (k_size // ttnn.TILE_SIZE) % k_tiles_per_core != 0:
                k_tiles_per_core -= 1
        else:
            if (
                not input_tensor_a_memory_config.memory_layout
                == ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED
            ):
                raise RuntimeError(f"MatmulMultiCoreReuseMultiCastProgramConfig: Must be block sharded")
            K = input_tensor_a.shape[-1] // ttnn.TILE_SIZE
            N = input_tensor_b.shape[-1] // ttnn.TILE_SIZE
            shard_shape = input_tensor_a_memory_config.shard_spec.shape
            m_tiles_per_core = shard_shape[0] // ttnn.TILE_SIZE
            n_tiles_per_core = (N * shard_shape[1]) // (K * ttnn.TILE_SIZE)
            k_tiles_per_core = shard_shape[1] // ttnn.TILE_SIZE

        m_subblock_size, n_subblock_size = _get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, fp32_dst)

        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            per_core_M=m_tiles_per_core,
            per_core_N=n_tiles_per_core,
            in0_block_w=k_tiles_per_core,
            out_subblock_h=m_subblock_size,
            out_subblock_w=n_subblock_size,
            transpose_mcast=False,
            fused_activation=get_fused_activation(activation=activation),
        )

    return program_config
