# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import tempfile
import os
import shutil

from ttnn.profile import (
    setup_torch_profiling,
    setup_tt_profiling,
    teardown_torch_profiling,
    teardown_tt_profiling,
    set_profiler_location,
)


@pytest.fixture(scope="function")
def torch_performance_csv():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    print(f"Created file {path}")
    yield path
    os.remove(path)


@pytest.fixture(scope="function")
def tt_profile_location():
    path = tempfile.TemporaryDirectory(dir="/home/ubuntu/git/tt-metal/.profiler")
    print(f"Created directory {path}")
    temp_dir_name_only = os.path.basename(path.name)
    set_profiler_location(temp_dir_name_only)
    yield path.name
    shutil.rmtree(path.name)


@pytest.fixture(scope="function")
def use_torch_profiling():
    setup_torch_profiling()
    yield
    teardown_torch_profiling()


@pytest.fixture(scope="function")
def use_tt_profiling():
    setup_tt_profiling()
    yield
    teardown_tt_profiling()
