.. _Yolo V3:

Yolo V3
=======

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - base from https://github.com/ultralytics/yolov3/tree/master
   * - Weights source
     - ``ultralytics/yolov`` from PyTorch hub
   * - Weights location
     - weka (``/mnt/MLPerf/tt_dnn-models/Yolo/models/yolov3``
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
This is YOLO v3 model set up to do object detection on images.
The model expect to receive an image as an input and will return labels and bounding boxes for detected objects.



Example of input and output:

* Input:

  .. image:: /_static/dog-cycle-car.png
    :width: 400
    :alt: A dog sitting on a porch in front of a parked bicycle, with street and parked car in the background.

|

* Output:

  .. image:: /_static/dog-cycle-car_output.png
    :width: 400
    :alt: A dog sitting on a porch in front of a parked bicycle, with street and parked car in the background. Four objects in the image are sorounded by box labeled with what is inside it.
