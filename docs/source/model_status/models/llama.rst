.. _Llama:

Llama
=====

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - 7B
   * - Weights source
     - ``decapoda-research/llama-7b-hf`` form HuggingFace
   * - Weights location
     - loaded from HuggingFace
   * - Weights format
     - PyTorch weights for HuggingFace model
   * - Batch
     - 1
   * - Iterations
     - 1
   * - Validation data set
     - TODO
   * - Validation method(s)
     - TODO
   * - Performance - throughput
     -
   * - Performance - compile time
     -
   * - Accuracy
     -

Notes
-----


Demo
----
This is Llama model set up to do conditional generation.
The model expect to receive a prompt (a string of text) and will return text generated based on the prompt.
The model will return a string of text that continues the statement started in prompt.




Example of input and output:

* Input:

  I believe the meaning of life is


* Output:

  I believe the meaning of life is to find your purpose and live it.
