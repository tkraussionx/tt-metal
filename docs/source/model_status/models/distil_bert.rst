.. _DistilBERT:

DistilBERT
==========

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - base
   * - Weights source
     - ``distilbert-base-uncased-distilled-squad``
   * - Weights location
     - loaded from HuggingFace
   * - Weights format
     - PyTorch weights for HuggingFace model
   * - Batch
     - 1
   * - Iterations
     - 1
   * - Validation data set
     -
   * - Validation method(s)
     -
   * - Performance - throughput
     -
   * - Performance - compile time
     -
   * - Accuracy
     -

Demo
----
This is DistilBERT model set up to do question answering.
The model expect to receive context (a paragraph of text) and a question that can be answered based on context.
The model will return an answer, that is a portion of input context.


Example of input (context and quesiton) and output (answer):

* Context:

  The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.


* Question:

  In what country is Normandy located?

* Answer:

  France
