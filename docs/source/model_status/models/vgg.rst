.. _VGG:

VGG
===

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - VGG16
   * - Weights source
     - ``VGG16_Weights.IMAGENET1K_V1`` from ``torchvision``
   * - Weights location
     - loaded from ``torchvision``
   * - Weights format
     - PyTorch weights for ``torchvision`` model
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
This is VGG model set up to do image classification.
The model expect to receive an image as an input and will return a string (usually single word) describing the image.



Example of input and output:

* Input:

  .. image:: /_static/ILSVRC2012_val_00048736.JPEG
    :width: 400
    :alt: A child and an adult dressed in baseball uniforms fist bumping.

|

* Output:

  baseball
