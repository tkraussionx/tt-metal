import torch

import pathlib

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)

def main():
    pcc = 0.99

    hf_files = sorted(list(pathlib.Path("tmp").glob("*.hf_*")))
    tt_files = sorted(list(pathlib.Path("tmp").glob("*.tt_*")))
    assert len(hf_files) == len(tt_files)
    print(hf_files)
    print(tt_files)

    for hf_file, tt_file in zip(hf_files, tt_files):
        hf_tensor = torch.load(hf_file)
        tt_tensor = torch.load(tt_file)

        hf_tensor = hf_tensor.to(tt_tensor.dtype)

        if hf_tensor.ndim == 3 and tt_tensor.ndim == 4:
            assert tt_tensor.shape[0] == 1
            tt_tensor = tt_tensor[0]

        if "attention_input.hf" in str(hf_file):
            tt_tensor = tt_tensor[:, :hf_tensor.shape[1]]


        assert hf_tensor.shape == tt_tensor.shape, f"{hf_file}: {hf_tensor.shape} != {tt_tensor.shape}"
        assert hf_tensor.dtype == tt_tensor.dtype, f"{hf_file}: {hf_tensor.dtype} != {tt_tensor.dtype}"

        does_pass, result = comp_pcc(hf_tensor, tt_tensor, pcc)
        assert does_pass, f"{hf_file}: \n{hf_tensor} != {tt_tensor}"
        print(f"{hf_file}: {result}")

if __name__ == "__main__":
    main()
