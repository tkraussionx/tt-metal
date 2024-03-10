# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import tt_lib


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    assert device.num_program_cache_entries() == 0, f"Unused program cache has non-zero entries?"
    device.disable_and_clear_program_cache()
    pass


@pytest.mark.eager_host_side
@pytest.mark.post_commit
def test_device_arch():
    assert tt_lib.device.Arch.GRAYSKULL.name == "GRAYSKULL"
    assert tt_lib.device.Arch.WORMHOLE_B0.name == "WORMHOLE_B0"
    pass
