.. _TT-DNN:

TT-DNN
******

Overview
========

TT-DNN is a simplified Python interface to the compute engine of the TT-Metal.

This will be the future plan. For now, the ``tt_lib`` Python module is a
unified Python interface that provides both TT-DNN and the Tensor library.

tt-DNN API through ``tt_lib``
=============================

Enums
-----

.. autoclass:: tt_lib.tensor.BcastOpMath
    :members: ADD, SUB, MUL

.. autoclass:: tt_lib.tensor.BcastOpDim
    :members: H, W, HW

.. autoclass:: tt_lib.tensor.ReduceOpMath
    :members: SUM, MAX

.. autoclass:: tt_lib.tensor.ReduceOpDim
    :members: H, W, HW

Tensor elementwise operations
-----------------------------

.. autofunction:: tt_lib.tensor.add

.. autofunction:: tt_lib.tensor.sub

.. autofunction:: tt_lib.tensor.mul

.. autofunction:: tt_lib.tensor.gelu

.. autofunction:: tt_lib.tensor.relu

.. autofunction:: tt_lib.tensor.sigmoid

.. autofunction:: tt_lib.tensor.exp

.. autofunction:: tt_lib.tensor.recip

.. autofunction:: tt_lib.tensor.sqrt

.. autofunction:: tt_lib.tensor.log

.. autofunction:: tt_lib.tensor.tanh


Tensor matrix math operations
-----------------------------

.. autofunction:: tt_lib.tensor.matmul

.. autofunction:: tt_lib.tensor.bmm


Tensor manipulation operations
------------------------------

These operations change the tensor shape in some way, giving it new dimensions
but in general retaining the data.

.. autofunction:: tt_lib.tensor.reshape

.. autofunction:: tt_lib.tensor.transpose

.. autofunction:: tt_lib.tensor.transpose_hc

.. autofunction:: tt_lib.tensor.transpose_hc_rm

.. autofunction:: tt_lib.tensor.tilize

.. autofunction:: tt_lib.tensor.untilize


Broadcast and Reduce
--------------------

.. autofunction:: tt_lib.tensor.bcast

.. autofunction:: tt_lib.tensor.reduce



``tt_lib`` Mini-Graph Library
==============================

Fused Operations
----------------

We have a variety of common operations that require fusion of multiple
base operations together.

.. autofunction:: tt_lib.fused_ops.linear.Linear

.. autofunction:: tt_lib.fused_ops.softmax.softmax

.. autofunction:: tt_lib.fused_ops.layernorm.Layernorm

.. autofunction:: tt_lib.fused_ops.add_and_norm.AddAndNorm


Experimental Operations
=======================

Operations in this section are experimental, don't have full support, and may behave in unexpectedx ways.

.. autofunction:: tt_lib.tensor.fill_rm

.. autofunction:: tt_lib.tensor.fill_ones_rm

.. autofunction:: tt_lib.tensor.pad_h_rm

.. autofunction:: tt_lib.tensor.large_bmm

.. autofunction:: tt_lib.tensor.large_bmm_single_block

.. autofunction:: tt_lib.tensor.conv_as_large_bmm_single_core_single_block

.. autofunction:: tt_lib.tensor.conv_as_large_bmm_single_core

.. autofunction:: tt_lib.tensor.bert_large_fused_qkv_matmul

.. autofunction:: tt_lib.tensor.bert_large_ff1_matmul

.. autofunction:: tt_lib.tensor.bert_large_ff2_matmul

.. autofunction:: tt_lib.tensor.bert_large_selfout_matmul

.. autofunction:: tt_lib.tensor.bert_large_pre_softmax_bmm

.. autofunction:: tt_lib.tensor.bert_large_post_softmax_bmm
