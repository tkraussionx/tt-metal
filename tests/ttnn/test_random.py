import ttnn
import torch
import pytest
import tt_lib as ttl


@pytest.mark.parametrize("shape", [(1, 1, 3, 4), (2, 3, 4, 5)])
def test_random_can_be_created(device, shape):
    tt_output = ttnn.random(shape=shape)
    assert tt_output != None


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1), (1, 1, 1, 1, 1)])
def test_random_can_not_be_created(device, shape):
    with pytest.raises(RuntimeError) as ex:
        ttnn.random(shape=shape)
    assert "TT_ASSERT @ tt_eager/tensor/tensor.cpp:34: this->shape_.rank() == 4" in str(ex.value)
