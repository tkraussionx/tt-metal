# Bloom Block Analysis

## How to Run

The analysis can be performed in two ways:

1. **Random Input Test**:
   - Set the boolean `real_input_test` to False to run the test with randomly generated inputs.
   - Execute the following command:
     ```
     pytest tests/ttnn/integration_tests/bloom/test_bloom_block_analysis.py
     ```

2. **Real Input Test**:
   - To run the analysis with real input data, follow these steps to generate the necessary inputs for the Bloom block:
     - Replace the `modeling_bloom.py` file located at `build/python_env/lib/python3.8/site-packages/transformers/models/bloom/` with the modified `modeling_bloom.py` in `tt-metal/tests/ttnn/integration_tests/bloom/`.
     - By default, the analysis is performed for the 5th indexed block. To analyze other indexed blocks, set the `block_index` from `modeling_bloom.py` and `test_bloom_block_analysis.py` to the desired index (any integer in the range 0-23).
     - Run the test for the Bloom QA model using the following command:
       ```
       pytest tests/ttnn/integration_tests/bloom/test_bloom_for_question_answering.py
       ```
   - Set the boolean `real_input_test` to True.
   - Execute the following command:
     ```
     pytest tests/ttnn/integration_tests/bloom/test_bloom_block_analysis.py
     ```

## Details

The script operates similarly to an independent sub-module test, with the entry point being `bloom_block` from `ttnn_optimized_functional_bloom`.

Upon running the test, the following outputs are generated:

1. **bloom_block_analysis.csv**:
   - Contains the analysis results.
   - Location: `tests/ttnn/integration_tests/bloom/bloom_block_analysis.csv`.

2. **Plots**:
   - Scatterplots and histograms comparing torch vs TTNN tensors.
   - Location: `tests/ttnn/integration_tests/bloom/plots`.

3. **Tensor Dumps**:
   - CSV files containing dumped tensors.
   - Location: `tests/ttnn/integration_tests/bloom/tensor_csv`.
