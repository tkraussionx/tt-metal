import tt_lib

def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0) -> tt_lib.tensor.Tensor:

    # print(f"input shape {input.shape()}")
    # print(f"batch1 shape {batch1.shape}")
    # print(f"batch2 shape {batch2.shape}")

    if beta != 1.0:
        input = tt_lib.tensor.mul(beta, input)

    tmp = tt_lib.tensor.bmm(batch1, batch2)

    if alpha != 1.0:
        tmp = tt_lib.tensor.mul(alpha, tmp)

    result = tt_lib.tensor.add(input, tmp)

    return result
