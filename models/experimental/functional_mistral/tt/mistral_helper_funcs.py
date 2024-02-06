# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm


def _reshape_for_broadcast(freqs_cis, x_shape, x_dim):
    ndim = x_dim
    assert 1 < ndim
    assert freqs_cis.shape == (x_shape[1], x_shape[-1]), (
        freqs_cis.shape,
        (x_shape[1], x_shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x_shape)]
    return freqs_cis.view(*shape)


def get_freqs_cis(freqs_cis, query_shape, key_shape, device, mem_config):
    freqs_cis = _reshape_for_broadcast(freqs_cis, query_shape, 4)

    freq_real = torch_to_tt_tensor_rm(freqs_cis.real, device)
    freq_img = torch_to_tt_tensor_rm(freqs_cis.imag, device)
    freqs_cis = tt_lib.tensor.complex_tensor(freq_real, freq_img)

    freq_real.deallocate()
    freq_img.deallocate()

    t_one_xq = tt_lib.tensor.ones(query_shape, output_mem_config=mem_config)
    t_one_xq = tt_to_torch_tensor(t_one_xq)
    t_one_xq = ttnn.to_device(
        ttnn.to_layout(ttnn.from_torch(t_one_xq, dtype=ttnn.bfloat16), layout=ttnn.TILE_LAYOUT), device
    )

    freqs_real = freqs_cis.real
    freqs_real = tt_to_torch_tensor(freqs_real)
    freqs_real = ttnn.to_device(
        ttnn.to_layout(ttnn.from_torch(freqs_real, dtype=ttnn.bfloat16), layout=ttnn.TILE_LAYOUT), device
    )

    freqs_imag = freqs_cis.imag
    freqs_imag = tt_to_torch_tensor(freqs_imag)
    freqs_imag = ttnn.to_device(
        ttnn.to_layout(ttnn.from_torch(freqs_imag, dtype=ttnn.bfloat16), layout=ttnn.TILE_LAYOUT), device
    )

    freqs_real = ttnn.pad(
        freqs_real, padding=((0, t_one_xq.shape[-0] - freqs_real.shape[0]), (0, 0), (0, 0), (0, 0)), value=1
    )
    bcast_freq_re_xq = t_one_xq * freqs_real
    freqs_imag = ttnn.pad(
        freqs_imag, padding=((0, t_one_xq.shape[-0] - freqs_imag.shape[0]), (0, 0), (0, 0), (0, 0)), value=1
    )

    bcast_freq_im_xq = t_one_xq * freqs_imag

    bcast_freq_re_xq = ttnn.to_torch(bcast_freq_re_xq)
    bcast_freq_re_xq = torch_to_tt_tensor_rm(bcast_freq_re_xq, device)
    bcast_freq_im_xq = ttnn.to_torch(bcast_freq_im_xq)
    bcast_freq_im_xq = torch_to_tt_tensor_rm(bcast_freq_im_xq, device)

    bcast_freq_xq = tt_lib.tensor.complex_tensor(bcast_freq_re_xq, bcast_freq_im_xq)

    bcast_freq_re_xq.deallocate()
    bcast_freq_im_xq.deallocate()

    t_one_xk = tt_lib.tensor.ones(key_shape, output_mem_config=mem_config)
    t_one_xk = tt_to_torch_tensor(t_one_xk)
    t_one_xk = ttnn.to_device(
        ttnn.to_layout(ttnn.from_torch(t_one_xk, dtype=ttnn.bfloat16), layout=ttnn.TILE_LAYOUT), device
    )

    bcast_freq_re_xk = t_one_xk * freqs_real
    bcast_freq_im_xk = t_one_xk * freqs_imag

    bcast_freq_re_xk = ttnn.to_torch(bcast_freq_re_xk)
    bcast_freq_re_xk = torch_to_tt_tensor_rm(bcast_freq_re_xk, device)
    bcast_freq_im_xk = ttnn.to_torch(bcast_freq_im_xk)
    bcast_freq_im_xk = torch_to_tt_tensor_rm(bcast_freq_im_xk, device)

    bcast_freq_xk = tt_lib.tensor.complex_tensor(bcast_freq_re_xk, bcast_freq_im_xk)

    bcast_freq_re_xk.deallocate()
    bcast_freq_im_xk.deallocate()
    freqs_cis.deallocate()

    return bcast_freq_xq, bcast_freq_xk
