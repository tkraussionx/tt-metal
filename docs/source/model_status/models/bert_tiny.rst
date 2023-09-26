.. _BERT Tiny:

BERT tiny
=========

.. list-table::
   :widths: 25 50
   :header-rows: 0

   * - Model variant
     - tiny
   * - Weights source
     - ``mrm8488/bert-tiny-finetuned-squadv2``
   * - Weights location
     - model on weka (``/mnt/MLPerf/tt_dnn-models/Bert/mrm8488/bert-tiny-finetuned-squadv2/``)
   * - Weights format
     - PyTorch weights for HuggingFace model
   * - Batch
     - 1
   * - Iterations
     - 1
   * - Validation data set
     - SQUAD-v2
   * - Validation method(s)
     - Exact match from evaluate library (https://pypi.org/project/evaluate/)
   * - Performance - throughput
     -
   * - Performance - compile time
     -
   * - Accuracy
     -


Demo
----
This is BERT Tiny model finetuned on SQUAD-v2 data set and is set up to do question answering.
The model expect to receive context (a paragraph of text) and a question that can be answered based on context.
The model will return an answer, that is a portion of input context.

Model accuracy is evaluated against SQUAD-v2 validation data set and percent of answers that are exact match to one of expected ansers in SQUAD-v2 dataset is reported.


Example of input (context and quesiton) and output (answer):

* Context:

  The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.


* Question:

  In what country is Normandy located?

* Answer:

  France
