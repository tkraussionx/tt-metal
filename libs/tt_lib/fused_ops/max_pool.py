import tt_lib as ttl

from typing import Union, List

def format_tensor(x, target_layout, device, pad_value=0.0):
    if x.layout() == target_layout:
        return x
    if x.layout() == ttl.tensor.Layout.ROW_MAJOR and target_layout == ttl.tensor.Layout.TILE:
        x_padded_shape = ttl.tensor.pad_to_tile_shape(x.shape(), False, False, True, True)
        return ttl.tensor.format_input_tensor(x, device, x_padded_shape, pad_value, target_layout)
    elif x.layout() == ttl.tensor.Layout.TILE and target_layout == ttl.tensor.Layout.ROW_MAJOR:
        return ttl.tensor.format_output_tensor(x, x.shape_without_padding(), device, target_layout)
    else:
        assert False

def run_max_pool_on_device_wrapper(device, kernel_size, stride, padding, channels_last=False):
    def max_pool_2d(x, x_actual_shape):
        if channels_last:
            #x = format_tensor(x, ttl.tensor.Layout.TILE, device)
            print("shape and layout before 1st transpose")
            print(x.shape())
            print(x.layout())
            x = ttl.tensor.transpose(x)
            print("shape and layout after 1st transpose")
            print(x.shape())
            print(x.layout())
            x = ttl.tensor.transpose_hc(x)
            print("shape and layout after 2nd transpose")
            print(x.shape())
            print(x.layout())
            x = x.reshape(x_actual_shape[0], x_actual_shape[3], x_actual_shape[1], x_actual_shape[2])
            print("shape before maxpool")
            print(x.shape())
        out = ttl.tensor.max_pool2d(x, kernel_size, kernel_size, stride, stride, padding, padding)
        max_pool_out_shape = out.shape()
        print("max pool output shape")
        print(max_pool_out_shape)
        if channels_last:
            out = out.reshape(max_pool_out_shape[0], max_pool_out_shape[1], 1, max_pool_out_shape[2]*max_pool_out_shape[3])
            out = format_tensor(out, ttl.tensor.Layout.TILE, device)
            print("shape and layout before 1st transpose")
            print(out.shape())
            print(out.layout())
            out = ttl.tensor.transpose_hc(out)
            print("shape and layout after 1st transpose")
            print(out.shape())
            print(out.layout())
            out = ttl.tensor.unpad(out, [0,0,0,0], [out.shape()[0]-1, 0, out.shape()[2]-1, out.shape()[3]-1])
            print("shape and layout after unpad")
            print(out.shape())
            print(out.layout())
            out = ttl.tensor.transpose(out)
            print("shape and layout after 2nd transpose")
            print(out.shape())
            print(out.layout())
            # out = ttl.tensor.unpad(out, [0,0,0,0], [max_pool_out_shape[0]-1, max_pool_out_shape[2]-1, max_pool_out_shape[3]-1, max_pool_out_shape[1]-1])
            # print("shape and layout after unpad")
            # print(out.shape())
            # print(out.layout())
        return out

    return max_pool_2d
