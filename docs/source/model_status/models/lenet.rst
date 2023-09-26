.. _LeNet:

LeNet
=====

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - n/a
   * - Weights source
     - TODO
   * - Weights location
     - weka (``/mnt/MLPerf/tt_dnn-models/LeNet/model.pt``)
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
This is LeNet model set up to do handwritten digit recognition.
The model expect to receive an image of a digit as input and it will return the digit in the image as an integer.


Example of input and output:

* Input:

  .. image:: /_static/torchvision_mnist_digit_7.jpg
    :width: 100
    :alt: White handwritten digit 7 on a black background

* Output:

  7
