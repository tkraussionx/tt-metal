import time
import torch

from loguru import logger

import utility_functions_new
from utility_functions_new import Profiler

def deprecated(func, *args, **kwargs):
    def helper():
        logger.warning("this file is deprecated and will be remove soon; use utility_functions_new.py")
        return func(args, kwargs)
    return helper

@deprecated
def is_close(a, b, rtol=1e-2, atol=1e-2, max_mag=2.0, max_mag_fraction=0.02):
    """
    A variant of np.isclose with logging.
    """
    return utility_functions_new.is_close(a, b, rtol, atol, max_mag, max_mag_fraction)


@deprecated
def print_diff_tt_pyt(a, b, annotation=""):
    return utility_functions_new.print_diff_tt_pyt(a, b, annotation)


@deprecated
def get_oom_of_float(float_lst):
    """
    Given a list of floats, returns a list of the order or magnitudes
    of the floats. Useful when you want to make sure that even if your
    tt outputs don't match pytorch all that well, they are at least
    on the same order of magnitude
    """
    return utility_functions_new.get_oom_of_float(float_lst)


@deprecated
def ttP(x, count=4, offset=0, stride=1):
    utility_functions_new.ttP(x, count, offset, stride)

@deprecated
def enable_compile_cache():
    """
    Enables persistent compiled kernel caching - disables recompiling the kernels for the duration of running process if built/kernels/.../hash directory with kernel binaries is present.
    """
    utility_functions_new.enable_compile_cache()

@deprecated
def disable_compile_cache():
    """
    Disables persistent compiled kernel caching. This is the default state.
    """
    utility_functions_new.disable_compile_cache()

@deprecated
def get_compile_cache_enabled():
    """
    Returns the current state of persistent compile cache on/off switch.
    """
    return utility_functions_new.get_compile_cache_enabled()

@deprecated
def enable_compilation_reports():
    """
    Enables generating reports of compilation statistics in .reports/tt_metal dir
    """
    return utility_functions_new.enable_compilation_reports()

@deprecated
def disable_compilation_reports():
    """
    Disables generating reports of compilation statistics
    """
    return utility_functions_new.disable_compilation_reports()

@deprecated
def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    return utility_functions_new.comp_allclose(golden, calculated, rtol, atol)


@deprecated
def comp_pcc(golden, calculated, pcc=0.99):
    return utility_functions_new.comp_pcc(golden, calculated, pcc)


@deprecated
def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    return utility_functions_new.comp_allclose_and_pcc(golden, calculated, rtol, atol, pcc)

@deprecated
def torch2tt_tensor(py_tensor: torch.Tensor, tt_device, tt_layout=ttl.tensor.Layout.TILE, tt_memory_config=ttl.tensor.MemoryConfig(True, -1)):
    return utility_functions_new.py_tensor, tt_device, tt_layout, tt_memory_config)

@deprecated
def tt2torch_tensor(tt_tensor, tt_host=None):
    return utility_functions_new.tt2torch_tensor(tt_tensor, tt_host)

@deprecated
def pad_by_zero(x: torch.Tensor, device):
    return utility_functions_new.pad_by_zero(x, device)

@deprecated
def unpad_from_zero(x, desired_shape, host):
    return utility_functions_new.unpad_from_zero(x, desired_shape, host)

@deprecated
class Profiler():
    def __init__(self):
        self.start_times = dict()
        self.times = dict()
        self.disabled = False

    def enable(self):
        self.disabled = False

    def disable(self):
        self.disabled = True

    def start(self, key):
        if self.disabled:
            return

        self.start_times[key] = time.time()

    def end(self, key, PERF_CNT=1):
        if self.disabled:
            return

        if key not in self.start_times:
            return

        diff = time.time() - self.start_times[key]

        if key not in self.times:
            self.times[key] = []

        self.times[key].append(diff / PERF_CNT)

    def get(self, key):
        if key not in self.times:
            return 0

        return sum(self.times[key]) / len(self.times[key])

    def print(self):

        for key in self.times:
            average = self.get(key)
            print(f"{key}: {average:.3f}s")


@deprecated
def tt_to_torch_tensor(tt_tensor, host):
    return utility_functions_new.tt_to_torch_tensor(tt_tensor, host)

@deprecated
def torch_to_tt_tensor_rm(py_tensor, device, shape=None, put_on_device=True):
    return torch_to_tt_tensor_rm(py_tensor, device, shape, put_on_device)

@deprecated
def torch_to_tt_tensor(py_tensor, device):
    return utility_functions_new.torch_to_tt_tensor(py_tensor, device)
