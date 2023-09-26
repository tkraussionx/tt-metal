.. _CodeGen:

CodeGen
=======

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - 350M
   * - Weights source
     - ``Salesforce/codegen-350M-mono``
   * - Weights location
     - loaded from HuggingFace
   * - Weights format
     - PyTorch weights for HuggingFace model
   * - Batch
     - 1
   * - Iterations
     - 1
   * - Validation data set
     -
   * - Validation method(s)
     -
   * - Performance - throughput
     -
   * - Performance - compile time
     -
   * - Accuracy
     -

Demo
----
This is CodeGen model set up to do program synthesis.
The model expect to receive context (a paragraph of text) that instructs it what code to generate.
The model will return the generated code.


Example of input (context and quesiton) and output (answer):

* Input:

  .. code-block:: Python

     # Implement a function that computes the square of an integer argument.


* Output:

  .. code-block:: Python

     # import libraries
     import numpy as np

     # Implement a function that computes the square of an integer argument.
     def square(x):
        return x**2
