import os

import torch
import numpy as np
import torch.nn.functional as F
from bindsnet.encoding import PoissonEncoder
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EMNIST
from torchvision.transforms.transforms import Lambda, ToTensor, Compose
from PIL import Image

TRAIN_FRAC, TEST_FRAC = 0.8, 0.2


class CachingEMNIST(EMNIST):
    def __init__(self, *args, **kwargs):
        self.patch_kwargs = kwargs.pop("patch_kwargs")
        super().__init__(*args, **kwargs)
        self.save_folder = os.path.join(self.root, "EMNIST", "encoded")
        os.makedirs(self.save_folder, exist_ok=True)
        self.kwarg_string = "_".join(
            f"{k}={v}".replace(" ", "") for k, v in self.patch_kwargs.items()
        )

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "EMNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "EMNIST", "processed")

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        os.makedirs(self.save_folder, exist_ok=True)
        path_to_save = os.path.join(self.save_folder, f"{index}_{self.kwarg_string}.pt")
        if os.path.exists(path_to_save):
            img = torch.load(path_to_save)
        else:
            img = Image.fromarray(img.numpy(), mode="L")

            if self.transform is not None:
                img = self.transform(img)
            torch.save(img, path_to_save)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def make_emnist(split="balanced", **patch_kwargs):
    patch_encoder = PatchEncoder(**patch_kwargs)
    dataset = CachingEMNIST(
        os.path.join(os.path.dirname(__file__), "data"),
        split=split,
        download=True,
        train=True,
        transform=Compose([ToTensor(), Lambda(patch_encoder)]),
        patch_kwargs=patch_kwargs,
    )
    train_len = round(len(dataset) * TRAIN_FRAC)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    return train_set, val_set


# TODO: CIFAR100


class PatchEncoder:
    def __init__(
        self,
        patch_shape,
        time_per_patch,
        use_4_position=True,
        patch_intensity=512,
        position_intensity=512,
        dt=1,
    ):
        self.patch_shape = patch_shape
        self.time_per_patch = time_per_patch
        self.use_4_position = use_4_position
        self.patch_intensity = patch_intensity
        self.position_intensity = position_intensity
        self.dt = dt

    def __call__(self, x):
        """Make patches

        x: 3D tensor of size (channels, height, width), normalized to [0,1]
        """

        # for each image (C, H, W), we compute its im2col with patches of shape patch_shape
        # then, encode each patch with Poisson
        PADDING = 0
        STRIDE = 1

        channels = x.shape[0]
        # if channels != self.n_input_channels:
        #     raise ValueError(
        #         f"Number of channels in the image ({channels}) doesn't match n_input_channels {self.n_input_channels}"
        #     )
        encoder = PoissonEncoder(time=self.time_per_patch, dt=self.dt)

        patch_h, patch_w = self.patch_shape
        input_size = channels * patch_h * patch_w

        # the images are in matplotlib format (channels, rows, columns)
        # but patches are apparently made as (x,y), which would be (columns, rows)
        # thus, transpose the patches to correspond to (rows, columns)
        x_t = x.transpose(1, 2)
        # dilation not implemented
        patches = F.unfold(
            x_t.unsqueeze(0), kernel_size=self.patch_shape, padding=PADDING, stride=STRIDE
        )
        patches = patches.squeeze(0)
        n_patches = patches.shape[-1]
        patches = patches.transpose(0, 1)  # (n_patches, input_size)

        patch_positions = get_im2col_indices(
            x.shape, patch_h, patch_w, padding=PADDING, stride=STRIDE
        )
        # normalize positions to [0,1]
        patch_positions /= torch.max(patch_positions, axis=0, keepdim=True)[0]  # (n_patches, 2)
        if self.use_4_position:
            inv_patch_positions = 1 - patch_positions
            patch_positions = torch.cat(
                [patch_positions, inv_patch_positions], dim=1
            )  # (n_patches, 4)
        assert len(patch_positions) == n_patches

        # encode each patch separately
        encoded_patches = torch.stack([encoder(p * self.patch_intensity) for p in patches])
        # encoded_patches: (n_patches, time_per_patch, input_size)
        # reshape to proper shape
        encoded_patches = encoded_patches.view(n_patches, self.time_per_patch, input_size)

        # encode positions
        encoded_positions = torch.stack(
            [encoder(p * self.position_intensity) for p in patch_positions]
        )
        # encoded_positions: (n_patches, time, 2|4)

        return dict(
            patches=patches,
            encoded_patches=encoded_patches,
            positions=patch_positions,
            encoded_positions=encoded_positions,
            image=x,
        )


# adapted from
# https://fdsmlhn.github.io/2017/11/02/Understanding%20im2col%20implementation%20in%20Python(numpy%20fancy%20indexing)/
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    return torch.FloatTensor([*zip(i[0], j[0])])
