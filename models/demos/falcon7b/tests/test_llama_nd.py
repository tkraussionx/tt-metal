import pytest
from loguru import logger
import ttnn

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
    tt2torch_tensor,
    get_devices_for_t3000,
)
import torch
from models.utility_functions import nearest_32

# {'bcast_batch': '1';
#  'compute_kernel_config': 'WormholeComputeKernelConfig(math_fidelity=LoFi;math_approx_mode=1;fp32_dest_acc_en=1;packer_l1_acc=1)';
#  'output_dtype': 'DataType::BFLOAT16';
#  'output_mem_config':
#  'MemoryConfig(memory_layout=TensorMemoryLayout::WIDTH_SHARDED;buffer_type=BufferType::L1;shard_spec=std::nullopt)';
#  'program_config': 'MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=16;per_core_M=1;per_core_N=8;fused_activation=std::nullopt)';
#  'transpose_a': 'false'; 'transpose_b': 'false'; 'untilize_out': 'false'; 'user_core_coord': 'std::nullopt'; 'user_fused_activation': 'std::nullopt'; 'user_run_batched': 'false'}


@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_reproduce_llama_mlp(all_devices, num_devices, use_program_cache, determinism_check_enabled=False):
    devices = []
    if num_devices == 8:
        devices = get_devices_for_t3000(all_devices, num_devices)
    else:
        devices = all_devices

    if num_devices == 8:
        logical_chip_id_to_coordinates = [None] * num_devices
        logical_chip_id_to_coordinates[0] = (1, 0)
        logical_chip_id_to_coordinates[1] = (0, 0)
        logical_chip_id_to_coordinates[2] = (0, 1)
        logical_chip_id_to_coordinates[3] = (1, 1)
        logical_chip_id_to_coordinates[4] = (2, 1)
        logical_chip_id_to_coordinates[5] = (3, 1)
        logical_chip_id_to_coordinates[6] = (3, 0)
        logical_chip_id_to_coordinates[7] = (2, 0)

    print("Running on: ", num_devices, " devices.")

    core_grid = ttnn.CoreGrid(y=1, x=8)

    M = 32
    N = 1024
    K = 8192

    in0_mem_config = ttnn.create_sharded_memory_config(
        shape=(M // core_grid.y, K // core_grid.x),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(devices[0].dram_grid_size().x - 1, devices[0].dram_grid_size().y - 1),
            )
        }
    )

    shard_shape = (K, nearest_32(N // 12))  # padded cols to divide by 12
    shard_spec = ttnn.ShardSpec(weight_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)

    in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    in0_dtype = ttl.tensor.DataType.BFLOAT16
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT16

    torch.manual_seed(1234)

    a_shape = [1, 1, M, K]
    b_shape = [1, 1, K, N]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = []
    b_t = []

    for device_idx in range(num_devices):
        a_t.append(ttl.tensor.Tensor(A, in0_dtype).to(ttl.tensor.Layout.TILE).to(devices[device_idx], in0_mem_config))
        b_t.append(ttl.tensor.Tensor(B, in1_dtype).to(ttl.tensor.Layout.TILE).to(devices[device_idx], in1_mem_config))

    bias_t = None

    wh_compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    # in0_block_w=16;per_core_M=1;per_core_N=8

    DRAM_SHARDED_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=16,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=8,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fused_activation=None,
    )

    num_nd_outputs = [0] * num_devices
    out = []
    reference_out = []

    for device_idx in range(num_devices):
        out.append(
            ttnn.matmul(
                a_t[device_idx],
                b_t[device_idx],
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=DRAM_SHARDED_PROGCFG,
                dtype=out_dtype,
                compute_kernel_config=wh_compute_kernel_config,
            )
        )

    if determinism_check_enabled:
        for device_idx in range(num_devices):
            reference_out.append(tt2torch_tensor(out[device_idx]))

    for i in range(100000):
        # run matmul on all devices
        for device_idx in range(num_devices):
            out[device_idx].deallocate(True)
            out[device_idx] = ttnn.matmul(
                a_t[device_idx],
                b_t[device_idx],
                program_config=DRAM_SHARDED_PROGCFG,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=out_dtype,
                compute_kernel_config=wh_compute_kernel_config,
            )

        # synchronize
        for device_idx in range(num_devices):
            if num_devices != 1:
                if num_devices == 2:
                    print("Start sync logicalDeviceID: ", device_idx)
                if num_devices == 8:
                    print(
                        "Start sync logicalDeviceID: ",
                        device_idx,
                        " eth coordinates: ",
                        logical_chip_id_to_coordinates[device_idx],
                    )
            else:
                print("Start single device sync:")
            ttl.device.Synchronize(devices[device_idx])
            if num_devices != 1:
                if num_devices == 2:
                    print("End sync logicalDeviceID: ", device_idx)
                if num_devices == 8:
                    print(
                        "End sync logicalDeviceID: ",
                        device_idx,
                        " eth coordinates: ",
                        logical_chip_id_to_coordinates[device_idx],
                    )
            else:
                print("End single device sync")

        # check if the output matches the first run output
        if determinism_check_enabled:
            for device_idx in range(num_devices):
                pt_out = tt2torch_tensor(out[device_idx])
                if torch.equal(reference_out[device_idx], pt_out):
                    logger.info(f"Device {device_idx} PCC: 1.0")
                else:
                    # for determinism check, we avoid calling comp_pcc func as it is heavy and with too many operations,
                    # part of the code that replaces nans/infs with zeros starts leaking memory, even if deallocation is forced,
                    # so we call it only in case we see tensors are not equal
                    _, pcc = comp_pcc(reference_out[device_idx], pt_out)
                    logger.info(f"Device {device_idx} PCC: {pcc}")
                    num_nd_outputs[device_idx] += 1

        logger.info(f"Iteration = {i}")

    if determinism_check_enabled:
        for device_idx in range(num_devices):
            logger.info(f"Number of non-deterministic outputs on device {device_idx} is {num_nd_outputs[device_idx]}")


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_specific_chip_llama_mlp(all_devices, logical_chip_index, use_program_cache):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")
    devices = get_devices_for_t3000(all_devices, num_devices_t3000)

    logical_chip_id_to_coordinates = [None] * num_devices_t3000
    logical_chip_id_to_coordinates[0] = (1, 0)
    logical_chip_id_to_coordinates[1] = (0, 0)
    logical_chip_id_to_coordinates[2] = (0, 1)
    logical_chip_id_to_coordinates[3] = (1, 1)
    logical_chip_id_to_coordinates[4] = (2, 1)
    logical_chip_id_to_coordinates[5] = (3, 1)
    logical_chip_id_to_coordinates[6] = (3, 0)
    logical_chip_id_to_coordinates[7] = (2, 0)

    print(
        "Selecting logical device id: ",
        logical_chip_index,
        " eth coordinates: ",
        logical_chip_id_to_coordinates[logical_chip_index],
    )
    target_device = devices[logical_chip_index]
    devices = [target_device]
    test_reproduce_llama_mlp(devices, 1, use_program_cache)


@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_determinism(
    all_devices,
    num_devices,
    use_program_cache,
):
    test_reproduce_llama_mlp(all_devices, num_devices, use_program_cache, determinism_check_enabled=True)


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_determinism_specific_chip(
    all_devices,
    logical_chip_index,
    use_program_cache,
):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")
    devices = get_devices_for_t3000(all_devices, num_devices_t3000)

    logical_chip_id_to_coordinates = [None] * num_devices_t3000
    logical_chip_id_to_coordinates[0] = (1, 0)
    logical_chip_id_to_coordinates[1] = (0, 0)
    logical_chip_id_to_coordinates[2] = (0, 1)
    logical_chip_id_to_coordinates[3] = (1, 1)
    logical_chip_id_to_coordinates[4] = (2, 1)
    logical_chip_id_to_coordinates[5] = (3, 1)
    logical_chip_id_to_coordinates[6] = (3, 0)
    logical_chip_id_to_coordinates[7] = (2, 0)

    print(
        "Selecting logical device id: ",
        logical_chip_index,
        " eth coordinates: ",
        logical_chip_id_to_coordinates[logical_chip_index],
    )
    target_device = devices[logical_chip_index]
    devices = [target_device]

    test_reproduce_llama_mlp(devices, 1, use_program_cache, determinism_check_enabled=True)
