import torch
import numpy as np
from loguru import logger
from functools import wraps


def multi_output_handler(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        assert len(args) >= 2
        pytorch_out = args[0]
        tt_lib_out = args[1]
        if isinstance(pytorch_out, (list, tuple)):
            assert len(pytorch_out) == len(
                tt_lib_out
            ), "Number of outputs from tt and pt does not match"
            result = True
            output = []
            for i in range(len(pytorch_out)):
                res, out = func(pytorch_out[i], tt_lib_out[i], *args[2:], **kwargs)
                result &= res
                output.append(out)
            output = "|".join(output)
            return result, output
        else:
            result, output = func(*args, **kwargs)
            return result, output

    return wrap


def get_atol_rtol_pcc(golden, calculated):
    # Calculate atol and rtol
    cal_atol = torch.max(torch.abs(golden - calculated)).item()
    cal_rtol = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()

    # Calculate PCC
    def get_pcc(golden, calculated):
        # Both tensors are nan
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            logger.warning("Both tensors are 'nan'")
            return 1.0

        # One tensor is all nan, the other is not
        if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            logger.error("One tensor is all nan, the other is not.")
            return 0.0

        # One tensor is all zero, the other is not
        if torch.any(golden.bool()) != torch.any(calculated.bool()):
            logger.warning("One tensor is all zero")
            return 0.0

        # if torch.any(torch.isinf(golden)) or torch.any(torch.isinf(calculated)):
        #    raise RuntimeError(f"Tensor overflow to infinity: \n{golden}\n{calculated}")

        # if torch.any(torch.isneginf(golden)) or torch.any(torch.isneginf(calculated)):
        #    raise RuntimeError(f"Tensor overflow to negative infinity: \n{golden}\n{calculated}")

        else:
            # For now, mask all infs and nans so that we check the rest... TODO
            golden = golden.clone()
            golden[
                torch.logical_or(
                    torch.isnan(golden),
                    torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
                )
            ] = 0
            calculated = calculated.clone()
            calculated[
                torch.logical_or(
                    torch.isnan(calculated),
                    torch.logical_or(
                        torch.isinf(calculated), torch.isneginf(calculated)
                    ),
                )
            ] = 0

            if torch.equal(golden, calculated):
                return 1.0

            if golden.dtype == torch.bfloat16:
                golden = golden.type(torch.float32)
                calculated = calculated.type(torch.float32)
            cal_pcc = np.min(
                np.ma.corrcoef(
                    np.ma.masked_invalid(
                        torch.squeeze(golden).detach().numpy()
                    ).flatten(),
                    np.ma.masked_invalid(
                        torch.squeeze(calculated).detach().numpy()
                    ).flatten(),
                )
            )

            if isinstance(cal_pcc, np.ma.core.MaskedConstant):
                return 1.0

            return cal_pcc

    cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


@multi_output_handler
def comp_equal(golden, calculated):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    return torch.equal(golden, calculated), output_str


@multi_output_handler
def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, _, output_str = get_atol_rtol_pcc(golden, calculated)
    return torch.allclose(golden, calculated, rtol, atol, True), output_str


@multi_output_handler
def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)
    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    return cal_pcc >= pcc, output_str


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    _, _, cal_pcc, output_str = get_atol_rtol_pcc(golden, calculated)
    passing = True
    passing &= torch.allclose(golden, calculated, rtol, atol, True)
    passing &= cal_pcc >= pcc
    return passing, output_str
