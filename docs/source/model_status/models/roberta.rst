.. _RoBERTa:

RoBERTa for masked LM
=====================

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - Base
   * - Weights source
     - ``roberta-base`` form HuggingFace
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


Demo
----
This is RoBERTa Base model set up to do masked language modeling.
The model expects to receive a prompt (a string of text) containing special token `<mask>` that the model will fill.
The model returns a string of text that fills the place of `<mask>` in input.

Example of input and output:

* Input:

  The Milky Way is a <mask> galaxy.


* Output:

  spiral



RoBERTa for question answering
==============================

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - Base
   * - Weights source
     - ``deepset/roberta-base-squad2`` form HuggingFace
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

Demo
----
This is RoBERTa Base model set up to do question answering.
The model expect to receive context (a paragraph of text) and a question that can be answered based on context.
The model will return an answer, that is a portion of input context.


Example of input (context and quesiton) and output (answer):

* Context:

  My name is Merve and I live in İstanbul.


* Question:

  Where do I live?

* Answer:

  İstanbul


RoBERTa for sequence classification
===================================

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - Base
   * - Weights source
     - ``cardiffnlp/twitter-roberta-base-emotion`` form HuggingFace
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

Demo
----
This is RoBERTa Base model set up to do sequence classification.
The model expect to receive a input (a string of text) and will determine what is the emotion expressed in it.


Example of input and output:

* Input:

  Hello, my dog is cute


* Output:

  optimism
