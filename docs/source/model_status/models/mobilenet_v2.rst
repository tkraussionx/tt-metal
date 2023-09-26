.. _MobileNet V2:

MobileNet V2
============

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - V2
   * - Weights source
     - ``google/mobilenet_v2_1.0_224`` form HuggingFace
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
This is MobileNet V2 model set up to do object detection on images.
The model expect to receive an image as an input and will return a string (usually single word) describing detected object.



Example of input and output:

* Input:

  .. image:: /_static/ILSVRC2012_val_00048736.JPEG
    :width: 400
    :alt: A child and an adult dressed in baseball uniforms fist bumping.

|

* Output:

  baseball
