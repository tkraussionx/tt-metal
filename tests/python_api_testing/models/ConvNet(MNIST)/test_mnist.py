from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from libs import tt_lib as ttl

from mnist import *

def test_mnist_convnet_inference():
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device

        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        #######

        torch_ConvNet, state_dict = load_torch()
        test_dataset, test_loader = prep_data()

        tt_convnet = TtConvNet(device, host, state_dict)
        correctness = 0
        for image, labels in test_loader:
            img = image.to('cpu')

            torch_output = torch_ConvNet(img)
            _, torch_predicted = torch.max(torch_output.data, -1)

            tt_output = tt_convnet(img)

            _, tt_predicted = torch.max(tt_output.data, -1)
            print(tt_output.shape, torch_output.shape)
            correctness += sum(tt_predicted.flatten() == torch_predicted.flatten())

            # print(f"Torch Predicted: {torch_predicted} \n   TT Predicted: {tt_predicted} \n        Labels: {labels[0]}")
            break
    print(correctness)
    assert correctness == batch_size
    ttl.device.CloseDevice(device)

test_mnist_convnet_inference()
