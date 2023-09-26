.. _T5 Base:

T5 Base
=======

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - Base
   * - Weights source
     - ``t5-base`` form HuggingFace
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
This is T5 Base model set up to do conditional generation.
The model expect to receive a prompt (a string of text) with a question or a task to accomplish.
The model will return a string of text that answers the prompt.




Example of input and output for translation task:

* Input:

  translate English to German: The house is wonderful.


* Output:

  Das Haus ist wunderbar.


Example of input and output for summarization task:

* Input:

  summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information.


* Output:

  QuillBot's Summarizer is a quick and easy way to read documents. instead of reading through documents, you can get a short annotated summary.
