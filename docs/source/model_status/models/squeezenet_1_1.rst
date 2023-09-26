.. _SqueezeNet_1_1:

SqueezeNet_1_1
==============

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - Tiny
   * - Weights source
     - ``SqueezeNet1_1_Weights.IMAGENET1K_V1`` form TorchVision
   * - Weights location
     - loaded from TorchVision
   * - Weights format
     - PyTorch weights for TorchVision model
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
This is Squeezenet 1.1 model set up to do object detection on images.
The model expect to receive an image as an input and will return a string (usually single word) describing detected object.



Example of input and output:

* Input:

  .. image:: /_static/ILSVRC2012_val_00048736.JPEG
    :width: 400
    :alt: A child and an adult dressed in baseball uniforms fist bumping.

|

* Output:

  baseball
