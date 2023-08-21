import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as torch_scaled_dot_product_attention
# from models.utility_functions import comp_allclose
from loguru import logger
import numpy as np

def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    atol_delta = torch.max(torch.abs(golden - calculated)).item()
    rtol_delta = torch.max(
        torch.abs(golden - calculated) / torch.abs(calculated)
    ).item()
    return (
        torch.allclose(golden, calculated, rtol, atol, True),
        f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}",
    )

def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, f"PCC: {1.0}"

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, f"PCC: {0.0}"

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, f"PCC: {0.0}"

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
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, f"PCC: {1.0}"

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, f"PCC: {1.0}"

    return cal_pcc >= pcc, f"PCC: {cal_pcc}"

class TT_functional:
    def scaled_dot_product_attention(
        Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False
    ):
        DTYPE = Q.dtype
        L, S = Q.size(-2), K.size(-2)
        # print("QKV:", Q.dtpye, K.dtype, V.dtype)

        def make_mask(L, S, DTYPE):
            attn_mask = torch.ones(L, S, dtype=DTYPE).tril(diagonal=0).to(K.device)
            inverted_mask = 1.0 - attn_mask
            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(DTYPE).min
            )

        assert (
            is_causal or attn_mask is not None
        ), "attn_mask must be provided if is_causal is False"
        assert (
            not is_causal or attn_mask is None
        ), "attn_mask must be None if is_causal is True"

        if attn_mask is None or is_causal:
            attn_mask = make_mask(L, S, DTYPE)

        # attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(Q.size(-1), dtype=DTYPE))) + attn_mask, dim=-1)
        # attn_weight = torch.dropout(attn_weight, dropout_p, train)
        ATT = (
            Q
            @ K.transpose(-2, -1)
            / torch.tensor(Q.size(-1) ** (1 / 2), dtype=DTYPE).to(K.device)
        )
        attn_weight = F.softmax(ATT + attn_mask, dim=-1, dtype=DTYPE)
        attn_weight = nn.Dropout(p=dropout_p)(attn_weight)
        # print(attn_weight.shape, attn_weight.dtype, V.shape, V.dtype)
        return attn_weight @ V


def test_falcon_scaled_dot_product_attention(
    query_layer_  = torch.randn(torch.Size([1, 71, 128, 64])),
    key_layer_ = torch.randn(torch.Size([1, 1, 128, 64])),
    value_layer_ = torch.randn(torch.Size([1, 1, 128, 64]))
    ):

    torch_out = torch_scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True)

    tt_torch_out = TT_functional.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True)

    does_pass, pcc_message = comp_pcc(torch_out, tt_torch_out, pcc =  0.98)

    logger.info(comp_allclose(torch_out, tt_torch_out))
    logger.info(pcc_message)


    if does_pass:
        logger.info("scaled_dot_product_attention passed!")
    else:
        logger.warning("scaled_dot_product_attention failed!")

    assert does_pass
