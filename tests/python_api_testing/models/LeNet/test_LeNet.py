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

from LeNet5 import *


def test_LeNet_inference():
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device

        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        #######

        torch_LeNet, state_dict = load_torch_LeNet()
        test_dataset, test_loader = prep_data()

        TTLeNet = TtLeNet5(num_classes, device, host, state_dict)
        correctness = 0

        for image, labels in test_loader:

            img = image.to('cpu')
            # img = image[:2, :, :, :].to('cpu')
            torch_output = torch_LeNet(img).unsqueeze(1).unsqueeze(1)
            _, torch_predicted = torch.max(torch_output.data, -1)

            tt_output = TTLeNet(img)

            _, tt_predicted = torch.max(tt_output.data, -1)
            correctness += sum(tt_predicted.flatten() == torch_predicted.flatten())
            break
            # print(tt_predicted.flatten() == torch_predicted.flatten(), correctness)
            # print(tt_output.shape, " tt")
            # print(torch_output.shape, "torch")
            # print(comp_allclose_and_pcc(tt_output, torch_output))
            # print(f"Torch Predicted: {torch_predicted} \n   TT Predicted: {tt_predicted} \n        Labels: {labels[ind]}")

        assert correctness == 16

    ttl.device.CloseDevice(device)
