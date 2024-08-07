import pytest


@pytest.mark.parametrize("num_devices", [(8)])
def test_hang(
    all_devices,
    num_devices,
):
    print("HANG")
