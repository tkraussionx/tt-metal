.. _FLAN-T5 Small:

FLAN-T5 Small
=============

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - Small
   * - Weights source
     - ``google/flan-t5-small``
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
This is FLAN-T5 Small model set up to do conditional generation.
The model expect to receive a prompt (a string of text) with a question or a task to accomplish.
The model will return a string of text that answers the prompt.



Example of input and output for translation task:

* Input:

  translate English to French: Welcome to NYC


* Output:

  Accueil à NCT


Example of input and output for summarization task:

* Input:

  summarize: I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder


* Output:

  I'm wasting my time.
