from abc import ABC, abstractmethod
from models.utility_functions import comp_pcc, torch2tt_tensor, tt2torch_tensor, get_devices_for_t3000
import torch
from loguru import logger
import ttnn


class DidtTestBase(ABC):
    def __init__(
        self,
        num_devices,
        all_devices,
        seq_len,
        inner_dim,
        weights_n,
        per_core_M,
        per_core_N,
        in_block_w,
        out_subblock_h,
        out_subblock_w,
        loop_count,
        determinism_check_enabled,
        determinism_check_iterations,
    ):
        self.in0_mem_config = None
        self.in1_mem_config = None
        self.out_mem_config = None
        self.in0_dtype = None
        self.in1_dtype = None
        self.out_dtype = None
        self.num_devices = num_devices
        self.all_devices = all_devices
        self.seq_len = seq_len
        self.inner_dim = inner_dim
        self.weights_n = weights_n
        self.per_core_M = per_core_M
        self.per_core_N = per_core_N
        self.in_block_w = in_block_w
        self.out_subblock_h = out_subblock_h
        self.out_subblock_w = out_subblock_w
        self.loop_count = loop_count
        self.determinism_check_enabled = determinism_check_enabled
        self.determinism_check_iterations = determinism_check_iterations

    @abstractmethod
    def set_in0_mem_config(self):
        pass

    @abstractmethod
    def set_in1_mem_config(self):
        pass

    @abstractmethod
    def set_out_mem_config(self):
        pass

    @abstractmethod
    def set_data_formats(self):
        pass

    @abstractmethod
    def set_program_config(self):
        pass

    @abstractmethod
    def set_compute_config(self):
        pass

    def test_didt(self):
        torch.manual_seed(1234)

        devices = []
        if self.num_devices == 8:
            devices = get_devices_for_t3000(self.all_devices, self.num_devices)
        else:
            devices = self.all_devices

        #  logger.info(f"Running on {num_devices} devices")

        self.set_in0_mem_config()
        self.set_in1_mem_config()
        self.set_out_mem_config()
        self.set_data_formats()

        a_shape = [1, 1, self.seq_len, self.inner_dim]
        b_shape = [1, 1, self.inner_dim, self.weights_n]

        num_activation_tensors = 1
        if self.determinism_check_enabled:
            # If we are running determinism checks, we want to switch activation tensors
            # every time we complete an iteration of a determinism check, to confirm that
            # device is producing new results, and not just reusing an already existing buffer
            num_activation_tensors = 10

        A = []
        for act in range(num_activation_tensors):
            A.append(torch.randn(a_shape))
        B = torch.randn(b_shape)

        a_t = [[None for _ in range(self.num_devices)] for _ in range(num_activation_tensors)]
        b_t = []

        for device_idx in range(self.num_devices):
            for act in range(num_activation_tensors):
                a_t[act][device_idx] = torch2tt_tensor(
                    A[act], devices[device_idx], ttnn.Layout.TILE, self.in0_mem_config, self.in0_dtype
                )
            b_t.append(torch2tt_tensor(B, devices[device_idx], ttnn.Layout.TILE, self.in1_mem_config, self.in1_dtype))

        self.set_program_config()
        self.set_compute_config()

        num_nd_outputs = [0] * num_devices
        reference_out = []
