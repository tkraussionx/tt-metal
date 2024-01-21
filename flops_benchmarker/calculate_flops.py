import sys


def calculate_conv_output_size(input_size, filter_size, stride, pad):
    return ((input_size - filter_size + 2 * pad) // stride) + 1


def calculate_maxpool_output_size(input_size, filter_size, stride):
    # For max pooling with kernel = (2, 2) and stride = (2, 2)
    # Hardcoded for now.
    return (input_size - filter_size) // stride + 1


def calculate_flops_conv(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
):
    output_height = calculate_conv_output_size(input_height, filter_height, stride_h, pad_h)
    output_width = calculate_conv_output_size(input_width, filter_width, stride_w, pad_w)

    flops_per_example = (
        output_height * output_width * output_channels * input_channels * filter_height * filter_width * 2
    )
    total_flops = flops_per_example * batch_size

    return total_flops, output_height, output_width


def calculate_flops_maxpool(input_channels, input_height, input_width, kernel_size, stride):
    output_height = calculate_maxpool_output_size(input_height, kernel_size, stride)
    output_width = calculate_maxpool_output_size(input_width, kernel_size, stride)

    flops_per_example = output_height * output_width * input_channels * kernel_size * kernel_size
    total_flops = flops_per_example

    return total_flops, output_height, output_width


# Prints a nice table of FLOPS, GFLOPS, TFLOPS, and Tensor sizes
def compute_performance_metrics(conv_params, maxpool_params, conv_time_ns, maxpool_time_ns):
    conv_flops, conv_output_height, conv_output_width = calculate_flops_conv(*conv_params)

    conv_time_s = conv_time_ns / 1e9
    maxpool_time_s = maxpool_time_ns / 1e9

    conv_flops_per_s = round(conv_flops / conv_time_s) if conv_time_s > 0 else 0

    maxpool_flops, maxpool_output_height, maxpool_output_width = calculate_flops_maxpool(
        conv_params[1],  # input_channels for maxpool (output channels from conv)
        conv_output_height,
        conv_output_width,
        *maxpool_params[:2],  # kernel_size, stride
    )
    maxpool_flops_per_s = round(maxpool_flops / maxpool_time_s) if maxpool_time_s > 0 else 0

    conv_gflops = conv_flops_per_s / 1e9
    conv_tflops = conv_flops_per_s / 1e12
    maxpool_gflops = maxpool_flops_per_s / 1e9
    maxpool_tflops = maxpool_flops_per_s / 1e12

    print(f"{'Metric'}\t{'Convolution'}\t{'MaxPool'}")
    print("-" * 50)
    print(f"{'FLOPS'}\t{conv_flops_per_s}\t{maxpool_flops_per_s}")
    print(f"{'GFLOPS'}\t{conv_gflops}\t{maxpool_gflops}")
    print(f"{'TFLOPS'}\t{conv_tflops}\t{maxpool_tflops}")
    print(f"{'Output Height'}\t{conv_output_height}\t{maxpool_output_height}")
    print(f"{'Output Width'}\t{conv_output_width}\t{maxpool_output_width}")

def main():
    print(f"Received {len(sys.argv)-1} arguments: {sys.argv[1:]}")  # Debugging line
    if len(sys.argv) != 20:
        print("Incorrect number of arguments.")
        sys.exit(1)

    batch_size = int(sys.argv[1])
    output_channels = int(sys.argv[2])
    input_channels = int(sys.argv[3])
    input_height = int(sys.argv[4])
    input_width = int(sys.argv[5])
    conv_filter_height = int(sys.argv[6])
    conv_filter_width = int(sys.argv[7])
    conv_stride_h = int(sys.argv[8])
    conv_stride_w = int(sys.argv[9])
    conv_pad_h = int(sys.argv[10])
    conv_pad_w = int(sys.argv[11])
    maxpool_filter_height = int(sys.argv[12])
    maxpool_filter_width = int(sys.argv[13])
    maxpool_stride_h = int(sys.argv[14])
    maxpool_stride_w = int(sys.argv[15])
    maxpool_pad_h = int(sys.argv[16])
    maxpool_pad_w = int(sys.argv[17])
    conv_time_ns = int(sys.argv[18])
    maxpool_time_ns = int(sys.argv[19])

    conv_params = (
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        conv_filter_height,
        conv_filter_width,
        conv_stride_h,
        conv_stride_w,
        conv_pad_h,
        conv_pad_w,
    )
    maxpool_params = (
        maxpool_filter_height,
        maxpool_filter_width,
        maxpool_stride_h,
        maxpool_stride_w,
        maxpool_pad_h,
        maxpool_pad_w,
    )

    compute_performance_metrics(conv_params, maxpool_params, conv_time_ns, maxpool_time_ns)


if __name__ == "__main__":
    main()
    
