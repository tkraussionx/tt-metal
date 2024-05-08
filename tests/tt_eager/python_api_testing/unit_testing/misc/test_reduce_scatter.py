import torch
import pytest
from loguru import logger
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, get_devices_for_t3000
import itertools


def is_unsupported_case(input_shape, scatter_dim, math_op, mem_config, num_devices, num_links, input_dtype, layout):
    if scatter_dim != 3:
        return True, "Only support for scatter_dim=3 is tested so far"

    return False, ""


def run_reduce_scatter_test(
    all_devices,
    num_devices,
    per_chip_input_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    if len(all_devices) != 8:
        pytest.skip("Not T3000!")

    if num_devices != 4:
        pytest.skip("Only testing for 4 devices")

    logger.info(f"num_devices: {num_devices}")
    logger.info(f"per_chip_input_shape: {per_chip_input_shape}")
    logger.info(f"scatter_dim: {scatter_dim}")
    logger.info(f"num_links: {num_links}")
    logger.info(f"math_op: {math_op}")
    logger.info(f"input_dtype: {input_dtype}")
    logger.info(f"layout: {layout}")
    logger.info(f"mem_config: {mem_config}")

    (is_known_failure, message) = is_unsupported_case(
        per_chip_input_shape, scatter_dim, math_op, mem_config, num_devices, num_links, input_dtype, layout
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")
    devices = get_devices_for_t3000(all_devices, num_devices)

    # Generate input tensors
    canonical_input_shape = per_chip_input_shape
    canonical_input_shape[scatter_dim] *= num_devices
    canonical_input_tensor = torch.rand(canonical_input_shape).bfloat16()
    input_tensors = torch.chunk(canonical_input_tensor, num_devices, scatter_dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttl.tensor.Tensor(t, input_dtype).to(layout).to(devices[i], mem_config))

    # Run the op
    for i in range(num_iters):
        tt_out_tensors = ttl.tensor.reduce_scatter(
            tt_input_tensors,
            scatter_split_dim=scatter_dim,
            reduce_op=math_op,
            num_links=num_links,
            output_mem_config=mem_config,
        )

        for d in devices:
            ttl.device.Synchronize(d)
        logger.info(f"Done iteration {i}")

    # Compute golden
    golden_canonical_out_tensor = torch.zeros(per_chip_input_shape)
    for i, t in enumerate(input_tensors):
        golden_canonical_out_tensor += t
    golden_output_tensors = torch.chunk(golden_canonical_out_tensor, num_devices, scatter_dim)

    # Compare
    assert len(golden_output_tensors) == len(tt_out_tensors)
    for i, t in enumerate(tt_out_tensors):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if input_dtype == ttl.tensor.DataType.BFLOAT16:
            eq, output = comp_equal(tt_output_tensor, golden_output_tensors[i])
        else:
            eq, output = comp_pcc(tt_output_tensor, golden_output_tensors[i])
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        assert eq, f"{i} FAILED: {output}"


@pytest.mark.parametrize(
    "num_devices, num_links",
    [
        # (2, 1),
        # (2, 2),
        (4, 1),
        (4, 2),
        (8, 1),
    ],
)
@pytest.mark.parametrize(
    "per_chip_input_shape, scatter_dim, layout",
    [
        ([1, 1, 32, 32], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([4, 1, 32, 32], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 4, 32, 32], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([4, 4, 32, 32], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 1, 32, 64], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 1, 64, 32], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 1, 64, 64], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 1, 32, 256], 3, ttl.tensor.Layout.ROW_MAJOR),
        ([1, 1, 1024, 1024], 3, ttl.tensor.Layout.ROW_MAJOR),
    ],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(buffer_type=ttl.tensor.BufferType.L1),
    ],
)
@pytest.mark.parametrize("math_op", [ttl.tensor.ReduceOpMath.SUM])
def test_reduce_scatter_post_commit(
    all_devices,
    num_devices,
    per_chip_input_shape,
    scatter_dim,
    num_links,
    math_op,
    input_dtype,
    layout,
    mem_config,
    use_program_cache,
    function_level_defaults,
    num_iters=1,
):
    run_reduce_scatter_test(
        all_devices,
        num_devices,
        per_chip_input_shape,
        scatter_dim,
        num_links,
        math_op,
        input_dtype,
        layout,
        mem_config,
        use_program_cache,
        function_level_defaults,
        num_iters,
    )
