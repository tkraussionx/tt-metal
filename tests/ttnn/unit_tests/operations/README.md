## This commit contains Mean unit test for VoVNet model.

1. `test_vovnet_mean` method in `tests/ttnn/unit_tests/operations/test_vovnet_mean.py` file will contain unit test for ttnn.mean.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_vovnet_mean.py::test_vovnet_mean`

## Expected Behaviour / Error(s):

### On GS:

Unit test passes with Pcc 0.99

### On WH(n150):

`AssertionError: 0.0`
