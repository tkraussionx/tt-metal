from pathlib import Path
import sys

import torch
import pytest
import tt_lib
import numpy
from models.utility_functions import comp_pcc
from tests.models.resnet.metalResnetBlock50 import compute_conv_output_shape, resnet50_1x1_conv_as_matmul, resnet50_optimized_conv, _nearest_32, format_tensor
import time
# hardcoding matmul config for 1x1 convs
# key: mm act height, mm act width, mm weight width
hardcoded_matmul_config_conv = {
    1 :
    {
        (3136, 64, 64) : {"compute_with_storage_grid_size" : (2,2),
                                "in0_block_w" : 2,
                                "out_subblock_h" : 1,
                                "out_subblock_w": 1,
                                "per_core_M": 49,
                                "per_core_N": 1,
                            },

        (3136, 64, 256) : {"compute_with_storage_grid_size" : (4,2),
                                "in0_block_w" : 2,
                                "out_subblock_h" : 1,
                                "out_subblock_w": 1,
                                "per_core_M": 49,
                                "per_core_N": 2,
                            },
        (3136, 256, 64) : {"compute_with_storage_grid_size" : (2,7),
                        "in0_block_w" : 8,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 14,
                        "per_core_N": 1,
                    },
        (3136, 256, 128) : {"compute_with_storage_grid_size" : (4,7),
                        "in0_block_w" : 8,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 14,
                        "per_core_N": 1,
                    },
        (800, 128, 512) : {"compute_with_storage_grid_size" : (4,2),
                        "in0_block_w" : 4,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 13,
                        "per_core_N": 4,
                    },
        (800, 512, 128) : {"compute_with_storage_grid_size" : (4,4),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 7,
                        "per_core_N": 1,
                    },
        (800, 512, 256) : {"compute_with_storage_grid_size" : (8,4),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 7,
                        "per_core_N": 1,
                    },
        (224, 256, 1024) : {"compute_with_storage_grid_size" : (8,7),
                        "in0_block_w" : 8,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 1,
                        "per_core_N": 4,
                    },
        (224, 1024, 256) : {"compute_with_storage_grid_size" : (8,7),
                        "in0_block_w" : 32,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 1,
                        "per_core_N": 1,
                    },
        (224, 1024, 512) : {"compute_with_storage_grid_size" : (8,7),
                        "in0_block_w" : 32,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 1,
                        "per_core_N": 2,
                    },
        (64, 512, 2048) : {"compute_with_storage_grid_size" : (8,2),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 1,
                        "per_core_N": 8,
                    },
        (64, 2048, 512) : {"compute_with_storage_grid_size" : (8,2),
                        "in0_block_w" : 64,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 1,
                        "per_core_N": 2,
                    },
    },
    2 : {
        (6272, 64, 64) : {"compute_with_storage_grid_size" : (2,4),
                                "in0_block_w" : 2,
                                "out_subblock_h" : 1,
                                "out_subblock_w": 1,
                                "per_core_M": 49,
                                "per_core_N": 1,
                            },

        (6272, 64, 256) : {"compute_with_storage_grid_size" : (4,4),
                                "in0_block_w" : 2,
                                "out_subblock_h" : 1,
                                "out_subblock_w": 1,
                                "per_core_M": 49,
                                "per_core_N": 2,
                            },
        (6272, 256, 64) : {"compute_with_storage_grid_size" : (2,9), # (x,y)
                        "in0_block_w" : 8,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 22, # across y
                        "per_core_N": 1, # across x
                    },
        (6272, 256, 128) : {"compute_with_storage_grid_size" : (4,9),
                        "in0_block_w" : 8,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 22,
                        "per_core_N": 1,
                    },
        (1568, 128, 512) : {"compute_with_storage_grid_size" : (4,4),
                        "in0_block_w" : 4,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 13,
                        "per_core_N": 4,
                    },
        (1568, 512, 128) : {"compute_with_storage_grid_size" : (4,9),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 6,
                        "per_core_N": 1,
                    },
        (1568, 512, 256) : {"compute_with_storage_grid_size" : (8,9),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 6,
                        "per_core_N": 1,
                    },
        (416, 256, 1024) : {"compute_with_storage_grid_size" : (8,7),
                        "in0_block_w" : 8,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 2,
                        "per_core_N": 4,
                    },
        (416, 1024, 256) : {"compute_with_storage_grid_size" : (8,7),
                        "in0_block_w" : 32,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 2,
                        "per_core_N": 1,
                    },
        (416, 1024, 512) : {"compute_with_storage_grid_size" : (8,7),
                        "in0_block_w" : 32,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 2,
                        "per_core_N": 2,
                    },
        (128, 512, 2048) : {"compute_with_storage_grid_size" : (8,4),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 1,
                        "per_core_N": 8,
                    },
        (128, 2048, 512) : {"compute_with_storage_grid_size" : (8,4),
                        "in0_block_w" : 64,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 1,
                        "per_core_N": 2,
                    },
    },
    8 :
    {
        (25088, 64, 64) : {"compute_with_storage_grid_size" : (2,8),
                                "in0_block_w" : 1,
                                "out_subblock_h" : 1,
                                "out_subblock_w": 1,
                                "per_core_M": 98,
                                "per_core_N": 1,
                            },

        (25088, 64, 256) : {"compute_with_storage_grid_size" : (8,8),
                                "in0_block_w" : 1,
                                "out_subblock_h" : 1,
                                "out_subblock_w": 1,
                                "per_core_M": 98,
                                "per_core_N": 1,
                            },
        (25088, 256, 64) : {"compute_with_storage_grid_size" : (2,8),
                        "in0_block_w" : 1,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 98,
                        "per_core_N": 1,
                    },
        (25088, 256, 128) : {"compute_with_storage_grid_size" : (4,8),
                        "in0_block_w" : 1,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 98,
                        "per_core_N": 1,
                    },
        (6272, 128, 512) : {"compute_with_storage_grid_size" : (4,9),
                        "in0_block_w" : 2,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 22,
                        "per_core_N": 4,
                    },
        (6272, 512, 128) : {"compute_with_storage_grid_size" : (4,9),
                        "in0_block_w" : 2,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 22,
                        "per_core_N": 1,
                    },
        (6272, 512, 256) : {"compute_with_storage_grid_size" : (8,9),
                        "in0_block_w" : 2,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 22,
                        "per_core_N": 1,
                    },
        (1568, 256, 1024) : {"compute_with_storage_grid_size" : (8,9),
                        "in0_block_w" : 4,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 6,
                        "per_core_N": 4,
                    },
        (1568, 1024, 256) : {"compute_with_storage_grid_size" : (8,9),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 6,
                        "per_core_N": 1,
                    },
        (1568, 1024, 512) : {"compute_with_storage_grid_size" : (8,9),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 6,
                        "per_core_N": 2,
                    },
        (416, 512, 2048) : {"compute_with_storage_grid_size" : (8,7),
                        "in0_block_w" : 16,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 2,
                        "per_core_N": 8,
                    },
        (416, 2048, 512) : {"compute_with_storage_grid_size" : (8,7),
                        "in0_block_w" : 32,
                        "out_subblock_h" : 1,
                        "out_subblock_w": 1,
                        "per_core_M": 2,
                        "per_core_N": 2,
                    },
    },
}

# (act_matrix_height, weight_matrix_width) : (act_block_height, weight_block_width, out_subblock_h, out_subblock_w, out_block_height, (num_cores_x, num_cores_y), per_core_out_matrix_height, per_core_out_matrix_width)
hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv = {
    1 : {
        (3136, 64) : [64, 64, 64, 64, 64, (7,7), 64, 64],
        (800, 128) : [32, 128, 32, 64, 32, (5,5), 32, 128],
        (224, 256) : [32, 128, 32, 128, 32, (1,7), 32, 256],
        (64, 512) : [32, 64, 32, 64, 32, (1, 2), 32, 512],
        (256, 512) : [32, 32, 32, 32, 96, (3,2), 96, 256],
        #(224, 512) : [32, 64, 32, 64, 64, (2,2), 128, 256],
    },
    2  : {
        (6272, 64) : [128, 64, 128, 64, 128, (7,7), 128, 64],
        (1568, 128) : [32, 128, 32, 64, 32, (7,7), 32, 128],
        (416, 256) : [64, 128, 64, 128, 64, (7,1), 64, 256],
        (128, 512) : [32, 64, 32, 64, 32, (1,4), 32, 512],
    },
    8 : {
        (25088, 64) : [128, 64, 128, 64, 128, (7,7), 512, 64],
        (6272, 128) : [64, 128, 64, 64, 64, (7,7), 128, 128],
        (1568, 256) : [224, 32, 32, 32, 224, (7,8), 224, 32],
        (416, 512) : [32, 32, 32, 32, 64, (7,8), 64, 64], # passes but non determinism
        #(416, 512) : [32, 64, 32, 32, 96, (5,8), 96, 64],
        (224, 512) : [32, 32, 32, 32, 64, (4,8), 64, 64], # passes determistic
        (288, 512) : [32, 32, 32, 32, 64, (5,8), 64, 64], # passes determistic
    },
}

@pytest.mark.parametrize("N", (8,))
@pytest.mark.parametrize(
    "K, C, H, W, R, S, stride_h, stride_w, pad_h, pad_w",
    (
        # # # 3x3 convs in rn50 (not complete list)
        # (64, 64, 56, 56, 3, 3, 1, 1, 1, 1),
        # (256, 256, 28, 28, 3, 3, 2, 2, 1, 1),
        # (256, 256, 14, 14, 3, 3, 1, 1, 1, 1),
        # (512, 512, 14, 14, 3, 3, 2, 2, 1, 1),
        (512, 512, 7, 7, 3, 3, 1, 1, 1, 1),
        # (512, 512, 16, 16, 3, 3, 1, 1, 1, 1),
        #(512, 512, 5, 5, 3, 3, 1, 1, 1, 1),
        #(512, 512, 6, 6, 3, 3, 1, 1, 1, 1),

        # # downsample convs in rn50 (not complete list)
        # (128, 128, 56, 56, 1, 1, 2, 2, 0, 0),
        # (256, 256, 28, 28, 3, 3, 2, 2, 1, 1),

    )
)
def test_resnet50_conv(use_program_cache, device, N,K,C,H,W,R,S,stride_h,stride_w,pad_h,pad_w):
    TILE_HEIGHT = 32
    TILE_WIDTH = 32
    TILE_VOLUME = TILE_HEIGHT * TILE_WIDTH
    assert C % 32 == 0
    assert K % 32 == 0
    num_iterations = 10
    torch.manual_seed(0)
    memory_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)
    conv_input_shape = [N, C, H, W]
    conv_weight_shape = [K, C, R, S]
    conv_bias_shape = [1, 1, 1, K]
    conv_input_pyt = torch.randn(conv_input_shape, dtype=torch.bfloat16).float()
    conv_input_pyt_nhwc = torch.permute(conv_input_pyt, (0, 2, 3, 1))
    conv_input_shape_nhwc = conv_input_pyt_nhwc.shape
    conv_weight_pyt = torch.randn(conv_weight_shape, dtype=torch.bfloat16).float()
    conv_bias_pyt = torch.randn(conv_bias_shape, dtype=torch.bfloat16).float()
    out_golden = torch.nn.functional.conv2d(conv_input_pyt, conv_weight_pyt,
                                            #bias=conv_bias_pyt.reshape(-1),
                                            stride=(stride_h, stride_w), padding=(pad_h, pad_w))

    is_1x1_conv = R == 1 and S == 1 and stride_h == 1 and stride_w == 1 and pad_h == 0 and pad_w == 0

    conv_params = [K, C, R, S, stride_h, stride_w, pad_h, pad_w, 1, 1]
    conv_output_shape = compute_conv_output_shape(conv_params, conv_input_shape_nhwc)
    print("Conv output shape - ", conv_output_shape)
    conv_as_mm_act_height = conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2]
    conv_as_mm_padded_act_height = _nearest_32(conv_as_mm_act_height)

    assert (conv_as_mm_padded_act_height, K) in hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv[N]
    [act_block_h_datums, weight_block_w_datums, out_subblock_h_datums, out_subblock_w_datums, out_block_h_datums, grid_size, per_core_out_matrix_h, per_core_weight_matrix_w] = hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv[N][(conv_as_mm_padded_act_height, K)]
    assert per_core_out_matrix_h % 32 == 0
    per_core_out_matrix_h_ntiles = (int) (per_core_out_matrix_h / 32)
    per_core_weight_matrix_w_ntiles = (int) (per_core_weight_matrix_w / 32)

    conv = resnet50_optimized_conv(conv_weight_pyt.reshape(-1).tolist(),
                        conv_params,
                        device,
                        [act_block_h_datums, C*S], [C*S, weight_block_w_datums],
                        [out_subblock_h_datums, out_subblock_w_datums], out_block_h_datums,
                        grid_size, per_core_out_matrix_h_ntiles, per_core_weight_matrix_w_ntiles,
                        conv_bias_pyt.reshape(-1).tolist(),
                        )

    conv_input_on_device = tt_lib.tensor.Tensor(
                        conv_input_pyt_nhwc.reshape(-1).tolist(),
                        conv_input_pyt_nhwc.shape,
                        tt_lib.tensor.DataType.BFLOAT16,
                        tt_lib.tensor.Layout.ROW_MAJOR).to(device, memory_config)

    outputs_on_device = []
    for i in range(3):
        outputs_on_device.append(conv(conv_input_on_device))

    for i in range(2):
        o_host1 = outputs_on_device[i].cpu().to_torch()
        o_host2 = outputs_on_device[i+1].cpu().to_torch()
        o1_shape = list(o_host1.size())
        o2_shape = list(o_host2.size())
        assert o1_shape == o2_shape
        assert o1_shape == [1, 1, conv_as_mm_padded_act_height, K]
        assert numpy.prod(o1_shape) % TILE_VOLUME == 0
        if not torch.equal(o_host1, o_host2):
            mismatch_found_on_unpadded_location = False
            # reshape to tile shape - 1, # of tiles, 32, 32
            num_tiles = (int) (numpy.prod(o1_shape) / TILE_VOLUME)
            assert K % TILE_WIDTH == 0
            num_tiles_row = (int) (K / TILE_WIDTH)
            o_host1_reshaped = o_host1.reshape(1, num_tiles, 32, 32)
            o_host2_reshaped = o_host2.reshape(1, num_tiles, 32, 32)
            for tile_id in range(num_tiles):
                for h_t in range(TILE_HEIGHT):
                    for w_t in range(TILE_WIDTH):
                        tile_id_h = (int) (tile_id / num_tiles_row)
                        tile_id_w = (int) (tile_id % num_tiles_row)
                        out_matrix_h = (tile_id_h * TILE_HEIGHT) + h_t
                        out_matrix_w = (tile_id_w * TILE_WIDTH) + w_t
                        assert out_matrix_h < conv_as_mm_padded_act_height
                        assert out_matrix_w < K
                        #print("ou")
                        if (out_matrix_h < conv_as_mm_act_height):
                            # compare
                            if o_host1_reshaped[0][tile_id][h_t][w_t] != o_host2_reshaped[0][tile_id][h_t][w_t]:
                                mismatch_found_on_unpadded_location = True
                                print(f"Mismatch at tile_id_h={tile_id_h},tile_id_w={tile_id_w},h_t={h_t},w_t{w_t}, o1={o_host1_reshaped[0][tile_id][h_t][w_t]}, o2={o_host2_reshaped[0][tile_id][h_t][w_t]}")
            assert not mismatch_found_on_unpadded_location


    output_on_device = outputs_on_device[0]
    outputs_on_device = []
    # convert tiled output to RM
    assert(output_on_device.layout() == tt_lib.tensor.Layout.TILE)

    prev_output_on_device_on_host = None
    for i in range(3):
        outputs_on_device.append(format_tensor(output_on_device, tt_lib.tensor.Layout.ROW_MAJOR, device, memory_config))
    for i in range(2):
        o_host1 = outputs_on_device[i].cpu().to_torch()
        o_host2 = outputs_on_device[i+1].cpu().to_torch()
        o1_shape = list(o_host1.size())
        o2_shape = list(o_host2.size())
        assert o1_shape == o2_shape
        assert o1_shape == [1, 1, conv_as_mm_act_height, K]
        if not torch.equal(o_host1, o_host2):
            for n in range(1):
                for c in range(1):
                    for h in range(conv_as_mm_act_height):
                        for w in range(K):
                            if o_host1[n][c][h][w] != o_host2[n][c][h][w]:
                                print(f"Mismatch at {n},{c},{h},{w}, o1={o_host1[n][c][h][w]}, o2={o_host2[n][c][h][w]}")
            assert False

    output_on_device = outputs_on_device[0]
    output_on_device = output_on_device.reshape(conv_output_shape[0], conv_output_shape[1], conv_output_shape[2], conv_output_shape[3])
    # Copy to host and Compare against pytorch
    out = output_on_device.cpu()
    assert out.layout() == tt_lib.tensor.Layout.ROW_MAJOR

    out_result = out.to_torch()
    # NHWC to NCHW
    out_result = torch.transpose(out_result, 2, 3)
    out_result = torch.transpose(out_result, 1, 2)

    # Compare against golden
    assert out_result.shape == out_golden.shape
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc
