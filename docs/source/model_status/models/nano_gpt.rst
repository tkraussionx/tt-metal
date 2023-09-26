.. _nanoGPT:

nanoGPT
=======

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - n/a
   * - Weights source
     - ``gpt2`` form HuggingFace
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
This is nanoGPT model set up to do conditional generation.
The model expect to receive a prompt (a string of text) and will return text generated based on the prompt.
The model will return a string of text that answers the prompt.




Example of input and output:

* Input:

  Once upon a time


* Output:

  Once upon a time, she was a very popular entertainer among the nobles. She managed to make a name for herself.

  After this, she tried to be more sophisticated, and in the past few years, she successfully taught people how to not just play,
