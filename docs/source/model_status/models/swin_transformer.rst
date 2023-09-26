.. _Swin Transformer:

Swin Transformer
================

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - tiny
   * - Weights source
     - ``microsoft/swin-tiny-patch4-window7-224`` from HuggingFace
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
This is Swin Transformer model set up to do image classification.
The model expect to receive an image as an input and will return a string (usually single word) describing the image.



Example of input and output:

* Input:

  .. image:: /_static/ILSVRC2012_val_00048736.JPEG
    :width: 400
    :alt: A child and an adult dressed in baseball uniforms fist bumping.

|

* Output:

  baseball
