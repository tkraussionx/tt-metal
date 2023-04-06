

import math

# Example call:
# _, _, H_out, W_out = size_after_residual(x.size(), out_channels, 3, stride, padding, dilation)

def size_after_residual(size, out_channels, kernel_size, stride, padding, dilation):
    """Calculate the size of the output of the residual function
    """
    N, C_in, H_in, W_in = size

    H_out = math.floor(
        (H_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    W_out = math.floor(
        (W_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    return N, out_channels, H_out, W_out
