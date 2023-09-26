.. _Bloom:

Bloom
=====

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - 560M
   * - Weights source
     - ``bigscience/bloom-560m`` form HuggingFace
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
This is Bloom model set up to do conditional generation.
The model expect to receive a prompt (a string of text) and will return text generated based on the prompt.
The model will return a string of text that continues the statement started in prompt.




Example of input and output:

* Input:

  It was a dark and stormy night


* Output:

  It was a dark and stormy night, the wind was blowing in a whirlwind, and the air was so hot that the brow of the wretches was like a bare stone. The men were in the midst of a storm, and the bare stone was a stone of fire.
