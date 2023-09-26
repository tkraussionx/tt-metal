.. _Inception V4:

Inception V4
============

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - V4
   * - Weights source
     - ``inception_v4`` loaded from Pytorch Image Models
   * - Weights location
     - loaded from Pytorch Image Models
   * - Weights format
     - PyTorch weights for Pytorch Image Models model
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
This is Inception V4 model set up to do object detection on images.
The model expect to receive an image as an input and will return a string (usually single word) describing detected object.



Example of input and output:

* Input:

  .. image:: /_static/ILSVRC2012_val_00048736.JPEG
    :width: 400
    :alt: A child and an adult dressed in baseball uniforms fist bumping.

|

* Output:

  baseball
