.. _ViT:

ViT
===

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - base
   * - Weights source
     - ``google/vit-base-patch16-224`` form HuggingFace
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
This is ViT model set up to do image classificaiton.
The model expect to receive a prompt (a string of text) and will return text generated based on the prompt.
The model will return a string of text that answers the prompt.



Example of input and output:

* Input:

  .. image:: /_static/ILSVRC2012_val_00048736.JPEG
    :width: 400
    :alt: A child and an adult dressed in baseball uniforms fist bumping.

|

* Output:

  baseball
