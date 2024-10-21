config = {
    "nc": 80,
    "depth_multiple": 1.0,
    "width_multiple": 1.0,
    "anchors": [
        [12, 16, 19, 36, 40, 28],
        [36, 75, 76, 55, 72, 146],
        [142, 110, 192, 243, 459, 401],
    ],
    "backbone": [
        [-1, 1, "Conv", [32, 3, 1]],  # conv1
        [-1, 1, "Conv", [64, 3, 2]],  # conv2
        [-1, 1, "Conv", [64, 3, 1]],  # conv3
        [-1, 1, "Conv", [128, 3, 2]],  # conv4
        [-1, 1, "Conv", [64, 1, 1]],  # conv5
        [-2, 1, "Conv", [64, 1, 1]],  # conv6
        [-1, 1, "Conv", [64, 3, 1]],  # conv7
        [-1, 1, "Conv", [64, 3, 1]],  # conv8
        [-1, 1, "Conv", [64, 3, 1]],  # conv9
        [-1, 1, "Conv", [64, 3, 1]],  # conv10
        [[-1, -3, -5, -6], 1, "Concat", [1]],
        [-1, 1, "Conv", [256, 1, 1]],  # conv11
        [-1, 1, "MP", []],
        [-1, 1, "Conv", [128, 1, 1]],  # conv12
        [-3, 1, "Conv", [128, 1, 1]],  # conv13
        [-1, 1, "Conv", [128, 3, 2]],  # conv14
        [[-1, -3], 1, "Concat", [1]],
        [-1, 1, "Conv", [128, 1, 1]],  # conv15
        [-2, 1, "Conv", [128, 1, 1]],  # conv16
        [-1, 1, "Conv", [128, 3, 1]],  # conv17
        [-1, 1, "Conv", [128, 3, 1]],  # conv18
        [-1, 1, "Conv", [128, 3, 1]],  # conv19
        [-1, 1, "Conv", [128, 3, 1]],  # conv20
        [[-1, -3, -5, -6], 1, "Concat", [1]],
        [-1, 1, "Conv", [512, 1, 1]],  # conv21
        [-1, 1, "MP", []],
        [-1, 1, "Conv", [256, 1, 1]],  # conv22
        [-3, 1, "Conv", [256, 1, 1]],  # conv23
        [-1, 1, "Conv", [256, 3, 2]],  # conv24
        [[-1, -3], 1, "Concat", [1]],
        [-1, 1, "Conv", [256, 1, 1]],  # conv25
        [-2, 1, "Conv", [256, 1, 1]],  # conv26
        [-1, 1, "Conv", [256, 3, 1]],  # conv27
        [-1, 1, "Conv", [256, 3, 1]],  # conv28
        [-1, 1, "Conv", [256, 3, 1]],  # conv29
        [-1, 1, "Conv", [256, 3, 1]],  # conv30
        [[-1, -3, -5, -6], 1, "Concat", [1]],
        [-1, 1, "Conv", [1024, 1, 1]],  # conv31
        [-1, 1, "MP", []],
        [-1, 1, "Conv", [512, 1, 1]],  # conv32
        [-3, 1, "Conv", [512, 1, 1]],  # conv33
        [-1, 1, "Conv", [512, 3, 2]],  # conv34
        [[-1, -3], 1, "Concat", [1]],
        [-1, 1, "Conv", [256, 1, 1]],  # conv35
        [-2, 1, "Conv", [256, 1, 1]],  # conv36
        [-1, 1, "Conv", [256, 3, 1]],  # conv37
        [-1, 1, "Conv", [256, 3, 1]],  # conv38
        [-1, 1, "Conv", [256, 3, 1]],  # conv39
        [-1, 1, "Conv", [256, 3, 1]],  # conv40
        [[-1, -3, -5, -6], 1, "Concat", [1]],
        [-1, 1, "Conv", [1024, 1, 1]],  # conv41
    ],
    "head": [
        [-1, 1, "SPPCSPC", [512]],
        [-1, 1, "Conv", [256, 1, 1]],  # conv42
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [37, 1, "Conv", [256, 1, 1]],  # conv43
        [[-1, -2], 1, "Concat", [1]],
        [-1, 1, "Conv", [256, 1, 1]],  # conv44
        [-2, 1, "Conv", [256, 1, 1]],  # conv45
        [-1, 1, "Conv", [128, 3, 1]],  # conv46
        [-1, 1, "Conv", [128, 3, 1]],  # conv47
        [-1, 1, "Conv", [128, 3, 1]],  # conv48
        [-1, 1, "Conv", [128, 3, 1]],  # conv49
        [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],
        [-1, 1, "Conv", [256, 1, 1]],  # conv50
        [-1, 1, "Conv", [128, 1, 1]],  # conv51
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [24, 1, "Conv", [128, 1, 1]],  # conv52
        [[-1, -2], 1, "Concat", [1]],
        [-1, 1, "Conv", [128, 1, 1]],  # conv53
        [-2, 1, "Conv", [128, 1, 1]],  # conv54
        [-1, 1, "Conv", [64, 3, 1]],  # conv55
        [-1, 1, "Conv", [64, 3, 1]],  # conv56
        [-1, 1, "Conv", [64, 3, 1]],  # conv57
        [-1, 1, "Conv", [64, 3, 1]],  # conv58
        [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],
        [-1, 1, "Conv", [128, 1, 1]],  # conv59
        [-1, 1, "MP", []],
        [-1, 1, "Conv", [128, 1, 1]],  # conv60
        [-3, 1, "Conv", [128, 1, 1]],  # conv61
        [-1, 1, "Conv", [128, 3, 2]],  # conv62
        [[-1, -3, 63], 1, "Concat", [1]],
        [-1, 1, "Conv", [256, 1, 1]],  # conv63
        [-2, 1, "Conv", [256, 1, 1]],  # conv64
        [-1, 1, "Conv", [128, 3, 1]],  # conv65
        [-1, 1, "Conv", [128, 3, 1]],  # conv66
        [-1, 1, "Conv", [128, 3, 1]],  # conv67
        [-1, 1, "Conv", [128, 3, 1]],  # conv68
        [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],
        [-1, 1, "Conv", [256, 1, 1]],  # conv69
        [-1, 1, "MP", []],
        [-1, 1, "Conv", [256, 1, 1]],  # conv70
        [-3, 1, "Conv", [256, 1, 1]],  # conv71
        [-1, 1, "Conv", [256, 3, 2]],  # conv72
        [[-1, -3, 51], 1, "Concat", [1]],
        [-1, 1, "Conv", [512, 1, 1]],  # conv73
        [-2, 1, "Conv", [512, 1, 1]],  # conv74
        [-1, 1, "Conv", [256, 3, 1]],  # conv75
        [-1, 1, "Conv", [256, 3, 1]],  # conv76
        [-1, 1, "Conv", [256, 3, 1]],  # conv77
        [-1, 1, "Conv", [256, 3, 1]],  # conv78
        [[-1, -2, -3, -4, -5, -6], 1, "Concat", [1]],
        [-1, 1, "Conv", [512, 1, 1]],  # conv79
        [75, 1, "RepConv", [256, 3, 1]],
        [88, 1, "RepConv", [512, 3, 1]],
        [101, 1, "RepConv", [1024, 3, 1]],
        [[102, 103, 104], 1, "Detect", ["nc", "anchors"]],
    ],
}
