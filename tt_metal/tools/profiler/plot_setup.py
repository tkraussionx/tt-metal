class test_base:
    timerAnalysisBase = {
        "FW start": {
            "type": "risc",
            "start": {"risc": "BRISC", "timerID": 0},
            "end": {"risc": "BRISC", "timerID": 1},
        },
        "BRISC kernel start -> BRISC kernel end": {
            "type": "risc",
            "start": {"risc": "BRISC", "timerID": 2},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC kernel start -> NCRISC kernel end": {
            "type": "risc",
            "start": {"risc": "NCRISC", "timerID": 2},
            "end": {"risc": "NCRISC", "timerID": 3},
        },
        "compute~": {
            "type": "core",
            "start": {"risc": "NCRISC", "timerID": 2},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "Device start -> Device end": {
            "type": "device",
            "start": {"core""risc": "NCRISC", "timerID": 1},
            "end": {"risc": "NCRISC", "timerID": 4},
        },
    }

    riscsData = {
        'BRISC': {
            "color":"light:g"
        },
        'NCRISC': {
            "color":"light:r"
        },
        'TENSIX': {
            "color":"light:gray"
        }

    }

    riscs = ["BRISC","NCRISC","TENSIX"]

    timerIDLabels = [
        (0, "Start"),
        (1, "Firmware Start"),
        (2, "Data Movement Kernel start"),
        (3, "Data Movement Kernel End"),
        (4, "Firmware End"),
    ]

class test_matmul_multi_core_multi_dram(test_base):
    timerAnalysis = {
        "NC_start -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "2"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "compute~": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "6"},
            "end": {"risc": "BRISC", "timerID": "5"},
        },
        "B_end": {"type": "single", "risc": "BRISC", "timerID": "4"},
    }


class test_matmul_multi_core_multi_dram_in0_mcast(test_base):
    timerAnalysis = {
        "NC_start_sender -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "10"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_start_reciever -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "7"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
    }


class test_matmul_multi_core_multi_dram_in1_mcast(test_base):
    timerAnalysis = {
        "NC_start_sender -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "20"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_start_reciever -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "16"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
    }

class test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast(test_base):
    timerAnalysis = {
        "NC_in0_s_in1_r -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "24"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_in0_s_in1_s -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "29"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_in0_r_in1_r -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "34"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_in0_r_in1_s -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "39"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
    }
