.. _Blaze Pose:

Blaze Pose
==========

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     -
   * - Weights source
     - from https://github.com/zmurez/MediaPipePyTorch/tree/master
   * - Weights location
     - weka (``/mnt/MLPerf/tt_dnn-models/Blazepose/models/``)
   * - Weights format
     - PyTorch weights
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
This is Blase Pose model set up to do upper body pose estimation on images.
The model expect to receive an image with a human body depicted as an input and will return locations of keypoints of upper part of human body in the image.



Example of input and output:

* Input:

  .. image:: /_static/yoga.jpg
    :width: 400
    :alt: A woman in a yoga position.

|

* Output:

  .. image:: /_static/yoga_output.png
    :width: 400
    :alt: A woman in a yoga position with lines marking position of upper body.
