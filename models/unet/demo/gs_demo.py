import os
import pytest
import torch
from torch import nn
import numpy as np
import random
from loguru import logger
from PIL import Image
from pathlib import Path

import tt_lib

from torch.utils.data import Dataset

from models.unet.tt.unet_model import TtUnet
from tests.python_api_testing.models.unet.reference.unet_model import UNet

from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
)


class BasicDataset(Dataset):
    def __init__(
        self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ""
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        logging.info(f"Creating dataset with {len(self.ids)} examples")
        logging.info("Scanning mask files to determine unique values")
        with Pool() as p:
            unique = list(
                tqdm(
                    p.imap(
                        partial(
                            unique_mask_values,
                            mask_dir=self.mask_dir,
                            mask_suffix=self.mask_suffix,
                        ),
                        self.ids,
                    ),
                    total=len(self.ids),
                )
            )

        self.mask_values = list(
            sorted(np.unique(np.concatenate(unique), axis=0).tolist())
        )
        logging.info(f"Unique mask values: {self.mask_values}")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC
        )
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert (
            img.size == mask.size
        ), f"Image and mask {name} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = torch2tt_tensor(img, device)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode="bilinear"
        )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros(
            (mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8
        )
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def run_tt_unet_demo(device):
    random.seed(42)
    torch.manual_seed(42)

    n_channels = 3
    n_classes = 2

    # load Unet model ================================================
    reference_model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
        map_location="cpu",
    )
    reference_model.load_state_dict(checkpoint)
    reference_model.eval()
    state_dict = reference_model.state_dict()

    # get TtUnet module ========================================
    gs_module = TtUnet(device, state_dict, n_channels, n_classes, False)

    in_files = ["models/unet/bmw.jpeg", "models/unet/honda.jpg"]
    out_files = [
        "models/unet/bmw_out.jpeg",
        "models/unet/honda_out.jpg",
    ]

    for i, filename in enumerate(in_files):
        print(f"Predicting image {filename} ...")
        img = Image.open(filename)

        mask = predict_img(
            net=gs_module,
            full_img=img,
            device=device,
        )

        out_filename = out_files[i]
        result = mask_to_image(mask, mask_values)
        result.save(out_filename)
        print(f"Mask saved to {out_filename}")


def test_gs_demo():
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_tt_unet_demo(device)

    tt_lib.device.CloseDevice(device)
