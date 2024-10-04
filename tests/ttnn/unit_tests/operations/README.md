## This commit contains ttnn.add unit test for Whisper model.

1. `test_whisper_add` method in `tests/ttnn/unit_tests/operations/test_add.py` file will contain unit test for ttnn.add.

    To run the test, use the following command : `pytest tests/ttnn/unit_tests/operations/test_add.py::test_whisper_add`

## Expected Behaviour / Error(s):

    Expected to pass the testcase with 0.99 pcc.

## Details:


1. Input shapes of tensor_a : [8, 1500 [1504], 512] and tensor_b : [1500 [1504], 512]

    #### On GS(e150):

    ```
    E       AssertionError: 0.20598982372530486
    ```

    #### On WH(n300):

    ```
    E       AssertionError: 0.17328692806300294
    ```
