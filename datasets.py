import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.data import random_split

TRAIN_FRAC, TEST_FRAC = 0.8, 0.2


def make_emnist(split="balanced"):
    dataset = EMNIST(
        os.path.join(os.path.dirname(__file__), "data"),
        split=split,
        download=True,
        train=True,
        transform=transforms.ToTensor()
    )
    train_len = round(len(dataset) * TRAIN_FRAC)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])


    return train_set, val_set


# TODO: CIFAR100