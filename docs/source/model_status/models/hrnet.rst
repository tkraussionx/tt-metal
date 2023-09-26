.. _HRNet:

HRNet
=====

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - small
   * - Weights source
     - ``hrnet_w18_small`` from PyTorch Image Models
   * - Weights location
     - loaded from PyTorch Image Models
   * - Weights format
     - PyTorch weights for PyTorch Image Models
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
This is HRNet model set up to do image classification.
The model expect to receive an image as an input and will return a string (usually single word) describing the image.



Example of input and output:

* Input:

  .. image:: /_static/ILSVRC2012_val_00048736.JPEG
    :width: 400
    :alt: A child and an adult dressed in baseball uniforms fist bumping.

|

* Output:

  baseball
