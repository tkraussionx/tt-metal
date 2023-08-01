import pytest
import tt_lib as ttl

@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    """Fixture to execute asserts before and after a test is run"""
    ttl.profiler.start_tracy_zone("TEST")
    yield # this is where the testing happens
    ttl.profiler.stop_tracy_zone()
