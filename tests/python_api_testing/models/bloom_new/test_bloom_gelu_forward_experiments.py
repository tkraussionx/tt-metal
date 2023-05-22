from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib
import random

from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.bloom_new.bloom_utils as bloom_utils
import python_api_testing.models.bloom_new.bloom_gelu_forward as bloom_gelu_forward


def gen_input(a,b,c,d,range1,range2):
    test_in = (range1-range2)*torch.rand(1, 1, 32, 32) + range2
    return test_in

def run_bloom_gelu_forward_test(device):

    # Prepare input
    torch.manual_seed(0)


    print('NEW TEST---------------------------------APPROX GELU TTMETAL NATIVE')

    # test 1
    test_in = gen_input(1,1,32,32,-1,1)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = tt_lib.tensor.gelu(tt_test_in)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")

    assert does_pass

    # test 2
    test_in = gen_input(1,1,32,32,-10,10)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = tt_lib.tensor.gelu(tt_test_in)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)


    # test 3
    test_in = gen_input(1,1,32,32,-1000,1000)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = tt_lib.tensor.gelu(tt_test_in)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")

    assert does_pass




    print('NEW TEST---------------------------------APPROX GELU TTMETAL NATIVE LARGE')

    # test 1
    test_in = gen_input(1,1,65536,65536,-1,1)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = tt_lib.tensor.gelu(tt_test_in)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")

    assert does_pass

    # test 2
    test_in = gen_input(1,1,65536, 65536, -10 ,10)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = tt_lib.tensor.gelu(tt_test_in)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)


    # test 3
    test_in = gen_input(1,1,65536,65536,-1000,1000)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = tt_lib.tensor.gelu(tt_test_in)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")

    assert does_pass




    print('NEW TEST---------------------------------APPROX TT_LIB')

    # test 1
    test_in = gen_input(1,1,32,32,-1,1)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")

    assert does_pass

    # test 2
    test_in = gen_input(1,1,32,32,-10,10)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)


    # test 3
    test_in = gen_input(1,1,32,32,-1000,1000)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")

    assert does_pass



    print('NEW TEST---------------------------------APPROX TT_LIB LARGE')

    # test 1
    test_in = gen_input(1,1,65536,65536,-1,1)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")

    assert does_pass

    # test 2
    test_in = gen_input(1,1,65536,65536,-10,10)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)


    # test 3
    test_in = gen_input(1,1,65536, 65536, -1000,1000)

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


    tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")


    print("TT_METAL APPROX-------------------")
    for i in range(20):
        dim = random.randint(1,65536)
        print('DIM:------------------------')
        print(dim)
        # test 1
        test_in = gen_input(1,1,dim,dim,-1,1)

        pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
        tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


        tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
        tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

        does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

        print(comp_allclose(pt_out, tt_out_converted))
        print(pcc_message)

        if does_pass:
            logger.info("bloom_gelu_forward: Passed!")
        else:
            logger.warning("bloom_gelu_forward: Failed!")

        assert does_pass


    print("TT_METAL APPROX-------------------")
    for i in range(20):
        dim = random.randint(1,65536)
        print('DIM:------------------------')
        print(dim)
        # test 1
        test_in = gen_input(1,1,dim,dim,-1000,1000)

        pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
        tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


        tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
        tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

        does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

        print(comp_allclose(pt_out, tt_out_converted))
        print(pcc_message)

        if does_pass:
            logger.info("bloom_gelu_forward: Passed!")
        else:
            logger.warning("bloom_gelu_forward: Failed!")

        assert does_pass



    print("TT_METAL CORE----------------")
    for i in range(10):
        dim = random.randint(1,65536)
        print('DIM:------------------------')
        print(dim)
        # test 1
        test_in = gen_input(1,1,dim,dim,-1,1)

        pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
        tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


        tt_out = tt_lib.tensor.gelu(tt_test_in)
        tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

        does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

        print(comp_allclose(pt_out, tt_out_converted))
        print(pcc_message)

        if does_pass:
            logger.info("bloom_gelu_forward: Passed!")
        else:
            logger.warning("bloom_gelu_forward: Failed!")

        assert does_pass


    print("TT_METAL CORE----------------")
    for i in range(10):
        dim = random.randint(1,65536)
        print('DIM:------------------------')
        print(dim)
        # test 1
        test_in = gen_input(1,1,dim,dim,-1000,1000)

        pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
        tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)


        tt_out = tt_lib.tensor.gelu(tt_test_in)
        tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

        does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

        print(comp_allclose(pt_out, tt_out_converted))
        print(pcc_message)

        if does_pass:
            logger.info("bloom_gelu_forward: Passed!")
        else:
            logger.warning("bloom_gelu_forward: Failed!")

        assert does_pass



def test_bloom_gelu_forward():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_bloom_gelu_forward_test(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_gelu_forward()
