# Optimization Techniques


## Contents

-[Optimization Techniques](#optimization-techniques)
- [1. Convolution](#convolution)
- [2. Upsample](#upsample)
- [3. Concat](#concat)
- [4. Linear](#linear)
- [5. Common](#common)
- [6. Yolov4 optimization](#yolov4-optimization)
- [7. MobilenetV2 Optimization](#mobilenetv2-optimization)

## Convolution
The default Conv2D configurations are as follows:

```py
Conv2dConfig {
   MathFidelity math_fidelity = MathFidelity::HiFi4;
   DataType dtype = DataType::BFLOAT16;
   DataType weights_dtype = DataType::BFLOAT16;
   bool math_approx_mode_enabled = true;
   bool fp32_dest_acc_enabled = false;
   bool packer_l1_accum_enabled = false;
   string activation = "";
   uint32_t input_channels_alignment = 32;
   bool deallocate_activation = false;
   bool reallocate_halo_output = false;
   uint32_t act_block_h_override = 0;
   uint32_t act_block_w_div = 1;
   bool reshard_if_not_optimal = false;
   bool override_sharding_config = false;
   TensorMemoryLayout shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
   std::optional<CoreRangeSet> core_grid = std::nullopt;
   bool transpose_shards = true;
   Layout output_layout = Layout::TILE;
   bool enable_act_double_buffer = false;
   bool enable_split_reader = false;
   bool enable_subblock_padding = false;
}
```

To optimize we can do the following,

1. Set math_fidelity to `MathFidelity::LoFi`
```py
                conv_config = ttnn.Conv2dConfig(
                            math_fidelity=ttnn.MathFidelity.LoFi,
                            )
```

2. Set the dtype and weight_dtype to `BFLOAT8_b`
```py
                conv_config = ttnn.Conv2dConfig(
                           dtype=ttnn.bfloat8_b,
                            weights_dtype=ttnn.bfloat8_b,
                            )
```

3. Enable `deallocate_activation` if you are not using the input tensor of the conv anywhere after passing into this conv.
```py
                conv_config = ttnn.Conv2dConfig(
                        deallocate_activation=True,
                        )
```

4. Enable `reshard_if_not_optimal`, if `shard_layout = TensorMemoryLayout::HEIGHT_SHARDED` and `override_sharding_config` is false which allows Conv to pick the optimal sharding config based on “height_sharding” config and reshards activation tensor to it.
```py
                conv_config = ttnn.Conv2dConfig(
                        reshard_if_not_optimal=True,
                        )
```

5. Enable `override_sharding_config` if `shard_layout = TensorMemoryLayout::HEIGHT_SHARDED` and `reshard_if_not_optimal` is `false` and if `true`, core_grid must be provided. If enabled Conv op reshards activation tensor to it
```py
                conv_config = ttnn.Conv2dConfig(
                        override_sharding_config=True,
                        )

```
6. Configure sharding with respect to the input dimension, i.e., it is advised to use `BLOCK_SHARDED` if C ~= N*H*W, `HEIGHT_SHARDED` if  N*H*W >>> C and `WIDTH_SHARDED` if C >>> N*H*W.

Example,
if input shape is 1,128,128,32[NHWC], we can use height sharding since  N*H*W >>> C,
```py
                conv_config = ttnn.Conv2dConfig(
                        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        )
```
if input shape is 1,32,32,640[NHWC], we can use block sharding since  N*H*W ~= C,
```py
                conv_config = ttnn.Conv2dConfig(
                        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        )
```
if input shape is 1,16,16,1024[NHWC], we can use width sharding since C >>> N*H*W,
```py
                conv_config = ttnn.Conv2dConfig(
                        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        )
```
7. Provide core_grid if `override_sharding_config` is enabled. While increasing the core_grid results in a higher core count and a higher FPS, it lowers core utilization. Reducing the core_grid results in a lower core count, which raises core utilization but lowers FPS.

                from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
                shard_grid = get_shard_grid_from_num_cores(self.core_count, device)
                conv_config.core_grid = shard_grid
                conv_config.override_sharding_config = True

8. Enabling `enable_act_double_buffer`, `enable_split_reader` may increase the performance of conv

```py
                conv_config = ttnn.Conv2dConfig(
                enable_act_double_buffer=True,
                enable_split_reader=True,
                )
```
## Upsample

Implementing ttnn.upsample() with shard memory configuration may increase performance,

```py
                shardspec = ttnn.create_sharded_memory_config_(
                        x.shape,
            Core_grid = x.memory_config().shard_spec.grid,
                    ttnn.ShardStrategy.HEIGHT,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR
                )
                if x.is_sharded():
                    x = ttnn.reshard(x, shardspec)
                else:
                    x = ttnn.interleaved_to_sharded(x, shardspec)


                x = ttnn.upsample(x, (2, 2, 1), memory_config=x.memory_config())
```

## Concat

ttnn.concat() op can be optimized by implementing it with shard memory config,

```py

                sharded_memory_config = ttnn.create_sharded_memory_config(
                        [512, 128],
                        core_grid=tensor2.memory_config().shard_spec.grid,
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        use_height_and_width_as_shard_shape=True,
                    )
                output_tensor = ttnn.concat([tensor1, tensor2], dim=3, memory_config=sharded_memory_config)
```

## Linear

Use shard MM/Linear instead of interleaved MM/Linear.

Instead of interleaved MM/Linear,

```py
                tt_input_a=ttnn.linear(
                        tt_input_a,
                        tt_weight_a,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat8_b,
                    )
```

FPS while using interleaved MM/linear,

FPS (MatMul/Conv Ops only): 17903.18 <br/>
FPS (Other Device Ops): 17903.18 <br/>
FPS (All Ops): 17903.18 <br/>

Use shard linear/MM,

```py
        tt_input_a = ttnn.to_memory_config(
                        tt_input_a,
                        memory_config=ttnn.create_sharded_memory_config(
                            tt_input_a.shape,
                            core_grid=ttnn.CoreGrid(y=8, x=8),
                            strategy=ttnn.ShardStrategy.HEIGHT,
                            orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        ),
                        dtype=ttnn.bfloat8_b,
                    )


        tt_input_a = ttnn.linear(
                        tt_input_a,
                        tt_weight_a,
                        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                        dtype=ttnn.bfloat8_b,
                        core_grid=ttnn.CoreGrid(y=8, x=8),
                        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.LoFi,
                        ),
                    )
```

FPS while using shard MM/linear,

FPS (MatMul/Conv Ops only): 258464.72 <br/>
FPS (Other Device Ops): 134444.743 <br/>
FPS (All Ops): 134444.743 <br/>

Note: The FPS will be greater when we use the 8x8 whereas the utilization will be increased as we reduce the core_grid. Utilization is calculated by `(PM ideal/device kernel duration)*(108/core count)`.

## Common

1. Store the tensor in L1 memory_config instead of DRAM and use TILE_LAYOUT rather than ROW_MAJOR_LAYOUT, Wherever possible.
2. Have weights/biases of all ops in bfloat8_b type.
3. Deallocate the tensor whenever the need is over.
4. Keep math fidelity to LoFi for ops like MM,layernorm, etc, By default it will be HiFi2 math fidelity.

## Yolov4 optimization:

In yolov4 implementation there are 9 sub_modules where resblock is implemented inline in other sub_modules. So, In our implementation there are 8 sub_modules i.e, Downsample1 to Downsample5, neck, head and yolov4 sub_modules.

Looking on Downsample 1,
There are a total of 8 conv, 8 batch_norm, 8 mish ops,1 addition and concat.

1.  We are fusing the weights/bias of conv and batch_norm.

2. These are the shapes of conv inputs, [1,320,320,3],[1,320,320,32],[1,160,160,64],[1,160,160,64],[1,160,160,64],[1,160,160,32],[1,160,160,64],[1,160,160,128]. Shapes are in NHWC format.

As input_tensor shapes NxHxW>> C, We are using HEIGHT_SHARDED layout for all convs.

```py
        conv_config = ttnn.Conv2dConfig(
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            )
```

3. The default layout of math_fidelity will be `math_fidelity=ttnn.MathFidelity.HiFi4`, Change it to `math_fidelity=ttnn.MathFidelity.LoFi`

4. Enable  `deallocate_activation=True`, wherever the input passed to conv is not used further in the pipeline.

5. Use `reshard_if_not_optimal= True`, Previous conv `in_channels!=out_channels` and `stride!=1`. Using reshard_if_not_optimal may or may not increase the performance of the model.

6. Change the dtype/weight_dtype of conv_config from bfloat16 to bfloat8_b.

7.  Change  Concat implementation from interleaved to sharded.

Before,(Interleaved concat)
```py
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_left = ttnn.sharded_to_interleaved(output_tensor_left, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_left], dim=3,memory_config=ttnn.L1_MEMORY_CONFIG)
```
After,(Shard concat)
```py
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_left = ttnn.to_layout(output_tensor_left, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [512, 128],
            core_grid=output_tensor_left.memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_left], dim=3, memory_config=output_sharded_memory_config)
```

8. There is no need of mentioning the memory_config for mish or addition operation since, It returns the same memory_config that we pass as input.


## MobilenetV2 Optimization

In MobilenetV2 model there are 52 convs, 52 batchnorms, 35 relu6, 1 linear and 10 adddition operations , Here is how we implemented the model optimally,

1. Firstly, all the conv ops and the linear ops are configured with the `math_fidelity=ttnn.MathFidelity.LoFi`.

2. Since we have the feature of fusing batchnorm with conv, we fused the weights/bias of conv with the batchnorm using `fold_batch_norm2d_into_conv2d` method.

3. All the convs are configured with the `dtype` and `weights_dtype` of `bfloat8_b`.

4. For the convs whose dimension has `NxHxW >>> c` are configured with `TensorMemoryLayout::HEIGHT_SHARDED`, for those with `C ~= NxHxW` are configured with `TensorMemoryLayout::BLOCK_SHARDED`.
    For example,
        The input dimension of c1 is [1,128,128,4], here `NxHxW >>> c`. So we configured c1 with `TensorMemoryLayout::HEIGHT_SHARDED`
        The input dimension of c14 is [1,16,16,192], here `C ~= NxHxW`. So we configured c14 with `TensorMemoryLayout::BLOCK_SHARDED`

5. The convs which are configiured with `TensorMemoryLayout::HEIGHT_SHARDED` are enabled with `reshard_if_not_optimal` which allows Conv to pick the optimal sharding config based on `HEIGHT_SHARDED` config and reshards activation tensor to it. _Note: Enabling `reshard_if_not_optimal` may or may not increase the performance of the model_.

6. Enable `deallocate_activation=True`, wherever the input passed to conv is not used further in the pipeline.

7. In addition to this, linear op is implemented with `compute_kernel_config` and `memory_config` as follows for optimal performance,
    ```py
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=True,
                        fp32_dest_acc_en=False,
                        packer_l1_acc=False,
                    )
            output_tensor = ttnn.linear(output_tensor, self.l1_weight, bias=self.l1_bias,
                        dtype=ttnn.bfloat8_b,  memory_config = ttnn.L1_MEMORY_CONFIG, compute_kernel_config=compute_kernel_config)```

8. Since ReLU6 and Addition operations returns the input memory_config, configuring ReLU6 and Addition ops are not needed.
