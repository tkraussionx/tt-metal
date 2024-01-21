#!/bin/bash

# Ensure that the TT_METAL_HOME variable is set
if [ -z "$TT_METAL_HOME" ]; then
    echo "TT_METAL_HOME is not set"
    exit 1
fi

# Check if pytest argument list is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 [pytest parameters]"
    exit 1
fi

# Assemble pytest string for calculate_flops.py parameter extraction
pytest="pytest $TT_METAL_HOME/tests/tt_eager/python_api_testing/unit_testing/test_optimized_conv_v2.py::test_optimized_conv_v2$1"

# Extract conv parameters from pytest string
param_string=$(echo "$pytest" | grep -oP "\[.*?\]" | tr -dc '0-9-')
param_string=${param_string%-}
IFS='-' read -ra PARAMS <<< "$param_string"
batch_size=${PARAMS[3]}
output_channels=${PARAMS[4]}
input_channels=${PARAMS[5]}
input_height=${PARAMS[6]}
input_width=${PARAMS[7]}
conv_filter_height=${PARAMS[8]}
conv_filter_width=${PARAMS[9]}
conv_stride_h=${PARAMS[10]}
conv_stride_w=${PARAMS[11]}
conv_pad_h=${PARAMS[12]}
conv_pad_w=${PARAMS[13]}
# Below are hardcoded maxpool parameters for now.
# Make sure these match the MaxPool parameters defined in test_optimized_conv_v2.py,
# namely lines 354~356
maxpool_filter_height=2
maxpool_filter_width=2
maxpool_stride_h=2
maxpool_stride_w=2
maxpool_pad_h=0
maxpool_pad_w=0

# Perform pytest, store full terminal output in output.txt
$pytest > output.txt

# Parse output.txt for kernel timings (in nanoseconds)
DEV_CONV=$(awk '/Operation tt::tt_metal::OptimizedConv/ {print $(NF-1)}' output.txt)
HOST_CONV=$(awk '/HOST_Conv/ {print $(NF-1)}' output.txt)
HOST_MAXPOOL=$(awk '/HOST_Maxpool/ {print $(NF-1)}' output.txt)

# Dump kernel timings for later use if you want
echo "DEV_CONV: $DEV_CONV nanoseconds" > timings.txt
echo "HOST_CONV: $HOST_CONV nanoseconds" >> timings.txt
echo "HOST_MAXPOOL: $HOST_MAXPOOL nanoseconds" >> timings.txt

# Calculate | FLOPS | GFLOPS | TFLOPS | based on above timings
# Pass in the following arguments into the Python calculator:
args=(
    # Conv parameters
    $batch_size $output_channels $input_channels $input_height $input_width $conv_filter_height $conv_filter_width $conv_stride_h $conv_stride_w $conv_pad_h $conv_pad_w

    # Maxpool parameters
    $maxpool_filter_height $maxpool_filter_width $maxpool_stride_h $maxpool_stride_w $maxpool_pad_h $maxpool_pad_w

    # Kernel timings. Feel free to interchange DEV and HOST for different stencil configurations, as long as variables are defined above
    $DEV_CONV $HOST_MAXPOOL
)

python calculate_flops.py "${args[@]}"
