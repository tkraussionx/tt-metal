import os
import pathlib

import torch

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc, comp_allclose
)


def main():
    pcc = 0.998

    hf_files = sorted(list(pathlib.Path("tmp").glob("*.hf_*")))
    tt_files = sorted(list(pathlib.Path("tmp").glob("*.tt_*")))

    assert len(hf_files) == len(tt_files)

    all_pass = True
    for hf_file, tt_file in zip(hf_files, tt_files):
        hf_tensor = torch.load(hf_file)
        tt_tensor = torch.load(tt_file)

        hf_tensor = hf_tensor.to(tt_tensor.dtype)

        if hf_tensor.ndim == 3 and tt_tensor.ndim == 4:
            assert tt_tensor.shape[0] == 1
            tt_tensor = tt_tensor[0]

        if hf_tensor.ndim == 2 and tt_tensor.ndim == 2:
            if hf_tensor.shape[1] < tt_tensor.shape[1]:
                tt_tensor = tt_tensor[:, :hf_tensor.shape[1]]

        if hf_tensor.ndim == 3 and tt_tensor.ndim == 3:
            if hf_tensor.shape[1] < tt_tensor.shape[1]:
                tt_tensor = tt_tensor[:, :hf_tensor.shape[1]]

        if hf_tensor.ndim == 4 and tt_tensor.ndim == 4:
            if hf_tensor.shape[0] < tt_tensor.shape[0]:
                tt_tensor = tt_tensor[:, :, :hf_tensor.shape[0]]
            if hf_tensor.shape[2] < tt_tensor.shape[2]:
                tt_tensor = tt_tensor[:, :, :hf_tensor.shape[2]]
            if hf_tensor.shape[3] < tt_tensor.shape[3]:
                tt_tensor = tt_tensor[:, :,  :, :hf_tensor.shape[3]]

        # if hf_tensor.shape != tt_tensor.shape:
        #     continue

        assert hf_tensor.shape == tt_tensor.shape, f"{hf_file}: {hf_tensor.shape} != {tt_tensor.shape}"
        assert hf_tensor.dtype == tt_tensor.dtype, f"{hf_file}: {hf_tensor.dtype} != {tt_tensor.dtype}"

        does_pass, result = comp_allclose(hf_tensor, tt_tensor, rtol=1e-1, atol=1e-1)
        all_pass &= does_pass
        # if float(result.split()[-1]) < pcc:
        print(f"{str(hf_file):40} PASS: {does_pass:<5} INFO: {result}")

    assert all_pass

if __name__ == "__main__":
    main()
