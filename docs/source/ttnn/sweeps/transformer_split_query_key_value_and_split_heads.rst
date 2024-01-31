.. _ttnn.sweep_test_transformer_split_query_key_value_and_split_heads:

transformer_split_query_key_value_and_split_heads
====================================================================
====  ========  ===========  ============  ===============  ===========  ===========  =================  ==============================================================================================================================
  ..  status      exception    batch_size    sequence_size    num_heads    head_size  input_dtype        input_memory_config
====  ========  ===========  ============  ===============  ===========  ===========  =================  ==============================================================================================================================
   0  passed            nan             1              384            4           64  DataType.BFLOAT16  tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
   1  passed            nan             1              384            4          128  DataType.BFLOAT16  tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
   2  passed            nan             1              384           16           64  DataType.BFLOAT16  tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
   3  passed            nan             1              384           16          128  DataType.BFLOAT16  tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
   4  passed            nan             1             1024            4           64  DataType.BFLOAT16  tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
   5  passed            nan             1             1024            4          128  DataType.BFLOAT16  tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
   6  passed            nan             1             1024           16           64  DataType.BFLOAT16  tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
   7  passed            nan             1             1024           16          128  DataType.BFLOAT16  tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
====  ========  ===========  ============  ===============  ===========  ===========  =================  ==============================================================================================================================
