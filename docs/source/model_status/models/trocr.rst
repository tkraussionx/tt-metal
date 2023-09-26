.. _TrOCR:

TrOCR
=====

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - base
   * - Weights source
     - ``microsoft/trocr-base-handwritten`` from HuggingFace
   * - Weights location
     - loaded from HuggingFace
   * - Weights format
     - PyTorch weights for HuggingFace
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
This is TrOCR model set up to do optical charcater recognition.
The model expect to receive an image with handwritten text and will output string whit that text.



Example of input and output:

* Input:

  .. image:: /_static/iam_ocr_image.jpg
    :width: 400
    :alt: Handwritten word "industrie"

|

* Output:

  industrie
