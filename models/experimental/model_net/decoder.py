import ttnn


def remove_non_valid_pixels(x, crop_size, off=0):
    assert crop_size % 2 == 0, "Crop size must be even"
    crop_size = crop_size // 2
    return x[:, crop_size + off : x.shape[1] - crop_size, crop_size + off : x.shape[2] - crop_size, :]


def crop_and_concat(inputs, shortcut_input, crop_size):
    if crop_size > 0:
        shortcut_input = remove_non_valid_pixels(shortcut_input, crop_size)

    output = ttnn.concat([inputs, shortcut_input], dim=3)
    return output


def decoder(input_a, input_b, force_crop, device):
    # Instead of Conv2DTranspose the below is done
    output_tensor = ttnn.upsample(input_a, scale_factor=(2, 2, 1))
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.pad(
        output_tensor,
        [1, output_tensor.shape[1] + 2, output_tensor.shape[2] + 2, output_tensor.shape[3]],
        [0, 0, 0, 0],
        0,
    )
    output_tensor = ttnn.to_device(output_tensor, device=device)
    output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
    output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = ttnn.to_device(output_tensor, device=device)

    if input_a.shape[3] > 64:
        output_tensor = output_tensor[:, :, :, :64]
    # Conv2DTranspose ends

    if force_crop[0] > 0:
        output_tensor = remove_non_valid_pixels(output_tensor, force_crop[0])

    output = crop_and_concat(output_tensor, input_b, force_crop[1])
    return output


def last_decoder(input_a, force_crop, device):
    # Instead of Conv2DTranspose the below is done
    output_tensor = ttnn.upsample(input_a, scale_factor=(2, 2, 1))
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.pad(
        output_tensor,
        [1, output_tensor.shape[1] + 2, output_tensor.shape[2] + 2, output_tensor.shape[3]],
        [0, 0, 0, 0],
        0,
    )
    output_tensor = ttnn.to_device(output_tensor, device=device)
    output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
    output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = ttnn.to_device(output_tensor, device=device)

    if input_a.shape[3] > 2:
        output_tensor = output_tensor[:, :, :, :2]
    # Conv2DTranspose ends

    if force_crop > 0:
        output_tensor = remove_non_valid_pixels(output_tensor, force_crop)

    return output_tensor
