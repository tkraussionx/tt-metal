import os
import torch


def download_imagenet_classes(path):
    # check if file exists
    classes_file = os.path.isfile(os.path.join(path, "imagenet_classes.txt"))
    if not classes_file:
        torch.hub.download_url_to_file(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            os.path.join(path, "imagenet_classes.txt"),
        )


def download_image(path, image="dog.jpg"):
    if image == "dog.jpg":
        torch.hub.download_url_to_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            os.path.join(path, "dog.jpg"),
        )
    else:
        torch.hub.download_url_to_file(
            "https://upload.wikimedia.org/wikipedia/commons/1/11/Viperine_water_snake_%28Natrix_maura%29.jpg",
            os.path.join(path, "water_sneak.jpg"),
        )
