## This commit contains Slice unit test for Whisper model.

1. `test_slice_whisper` method in `tests/ttnn/unit_tests/operations/test_slice.py` file will contain unit test for ttnn.conv2d.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_slice.py::test_slice_whisper`

## Expected Behaviour / Error(s):

    Expected to pass the testcase with bias.

## Details:

#### On GS(e150), WH(n300):


1. Split along third dimension in 5d tensor of shape: [8, 1500, 2, 6, 64]
    ```
    >       raise NotImplementedError
    E       NotImplementedError
    ```
