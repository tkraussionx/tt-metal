.. _Falcon:

Falcon
======

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - 7B
   * - Weights source
     - ``tiiuae/falcon-7b-instruct`` form HuggingFace
   * - Weights location
     - weka (``/mnt/MLPerf/tt_dnn-models/Falcon/tiiuae/falcon-7b-instruct/``)
   * - Weights format
     - pre-processed weights for GS Falcon model
   * - Batch
     - TODO: 32
   * - Iterations
     - TODO: 2048
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
This is Falcon 7B model set up to do conditional generation.
The model expect to receive a prompt (a string of text) and will return text generated based on the prompt.


Example of input and output:

* Input:

  Write a poem about Valencia


* Output:

  Valencia, the city of the sun,

  A place of beauty, of fun,

  A place of culture, of art,

  Where the people are warm, and the heart.

  The city of the sun, where the sky is blue,
