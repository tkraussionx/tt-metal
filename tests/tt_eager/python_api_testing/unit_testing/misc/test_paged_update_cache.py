# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import tt_lib as ttl
from loguru import logger
from models.utility_functions import nearest_32, pad_by_zero
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from models.utility_functions import is_grayskull


@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [32])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
@pytest.mark.parametrize("cache_idx", [0, 1, 127, 1057])
@pytest.mark.parametrize("cache_dtype", [ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B])
def test_update_cache_decode(
    self,
    cache_idx,
    head_dim,
    max_seq_len,
    num_users,
    num_heads,
    input_dtype,
    cache_dtype,
    device,
    use_program_cache,
):
    input_shape = [1, num_users, num_heads, head_dim]
    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    cache = torch.randn(cache_shape).bfloat16().float()
    cachett = ttl.tensor.Tensor(cache, cache_dtype).to(ttl.tensor.Layout.TILE).to(device)
    x = torch.randn(input_shape).bfloat16().float()

    xt = ttl.tensor.Tensor(x, input_dtype).to(ttl.tensor.Layout.TILE)
    # Input is sharded
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = num_users
    shard_grid = ttl.tensor.CoreRangeSet(ttl.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True))
    input_shard_spec = ttl.tensor.ShardSpec(
        shard_grid,
        [
            xt.volume() // xt.get_legacy_shape()[-1] // num_cores,
            xt.get_legacy_shape()[-1],
        ],
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )
    input_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, input_shard_spec
    )
    xt = xt.to(device, input_mem_config)

    # Create arbitrary update indices
    cache_idxs = [cache_idx + i * 17 for i in range(num_users)]

    cachett = ttl.operations.primary.paged_update_cache(cachett, xt, cache_idxs)

    for i in range(num_users):
        update_idx = cache_idxs[i]
        cache[i, 0:num_heads, update_idx : update_idx + x.shape[-2], 0 : x.shape[-1]] = x

    tt_got_back = cachett.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    if input_dtype == ttl.tensor.DataType.BFLOAT16 and cache_dtype == input_dtype:
        eq_cache, output_cache = comp_equal(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_equal(
            x, tt_got_back[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]]
        )  # checks the updated parts
    else:
        eq_cache, output_cache = comp_pcc(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_pcc(
            x, tt_got_back[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]]
        )  # checks the updated parts
    logger.info(output_cache)
    logger.info(output_update)
    assert eq_cache and eq_update
