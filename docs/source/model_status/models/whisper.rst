.. _Whisper:

Whisper
=======

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - Tiny
   * - Weights source
     - ``openai/whisper-tiny.en`` form HuggingFace
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
This is Whisper Tiny model set up to do automatic speech recognition.
The model expect to receive an audio clip and will retun a string with the transcript of audio.




Example of input and output for automatic speech recognition task:

* Input:

  audio clip (Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.)


* Output:

   Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
