import os
import click
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import make_emnist
from network import Prototype1
from visualization import (
    visualize_image,
    visualize_patches,
    visualize_spikes,
    visualize_filter_weights,
)
from bindsnet.network import load


@click.command()
@click.option("-b", "--batch_size", default=1, show_default=True)  # B=1: 600hr. B=32: 170hr
@click.option("-e", "--n_epochs", default=1, show_default=True)
@click.option(
    "--n_train",
    default=None,
    show_default=True,
    type=int,
    help="Number of train examples to use per epoch.",
)
@click.option(
    "--n_val",
    default=5,
    show_default=True,
    type=int,
    help="Number of examples to visualize after training.",
)
@click.option("--time_per_patch", default=10, show_default=True, help="Timesteps per patch")
@click.option(
    "-l",
    "--load-filename",
    default=None,
    type=str,
    help="Filename to load a model from",
)
@click.option("--n_filters", default=32, show_default=True)
@click.option("--n_l1_features", default=64, show_default=True)
@click.option("--n_l2_features", default=64, show_default=True)
@click.option(
    "--vis_val_images",
    default=True,
    show_default=True,
    is_flag=True,
    help="Show images for each image during validation visualization",
)
@click.option(
    "--vis_val_spikes",
    default=True,
    show_default=True,
    is_flag=True,
    help="Show spikes for each image during validation visualization",
)
@click.option(
    "--vis_filters",
    default=True,
    show_default=True,
    is_flag=True,
    help="Visualize filter weights during validation visualization",
)
@click.option(
    "--vis_val_patches",
    default=False,
    show_default=True,
    is_flag=True,
    help="Show image patches for each image during validation visualization",
)
@click.option(
    "--save/--no_save",
    default=True,
    show_default=True,
    is_flag=True,
    help="Save model after training",
)
@click.option(
    "--train/--no_train",
    "do_train",
    default=True,
    show_default=True,
    help="Run the train loop",
)
def main(**kwargs):
    kwargs["time_per_patch"] = 10
    kwargs["use_4_position"] = True
    kwargs["patch_shape"] = (5, 5)

    seed = 1
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    else:
        torch.manual_seed(seed)
        device = "cpu"

    kwargs["device"] = device

    num_workers = os.cpu_count() - 1
    torch.set_num_threads(num_workers)
    print("Running on Device = ", device)

    train_data, val_data = make_emnist(
        split="balanced",
        patch_shape=kwargs["patch_shape"],
        time_per_patch=kwargs["time_per_patch"],
        use_4_position=kwargs["use_4_position"],
        patch_intensity=128,
        position_intensity=128,
    )
    # each batch is (batch_size, n_patches, time_per_patch, n_channels*patch_shape)
    if kwargs["load_filename"]:
        network = load(kwargs["load_filename"], map_location=device)
    else:
        network = Prototype1(
            n_input_channels=1,
            patch_shape=kwargs["patch_shape"],
            n_filters=kwargs["n_filters"],
            n_l1_features=kwargs["n_l1_features"],
            n_l2_features=kwargs["n_l2_features"],
        )
    network = network.to(device)

    ##### TRAIN #####
    # the dataloaders have to be out here for pickling for some reason
    if kwargs["do_train"]:
        train_dataloader = DataLoader(
            train_data,
            batch_size=kwargs["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device == "cuda",
        )
        train_network(network, train_dataloader, kwargs)

    #### VALIDATION AND VISUALIZATION #####
    print("Validation")
    val_dataloader = DataLoader(
        val_data, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=device == "cuda"
    )
    validate_network(network, val_dataloader, kwargs)


def train_network(network, train_dataloader, kwargs):
    network.train(True)
    for epoch in range(kwargs["n_epochs"]):
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Get next input sample.
            if kwargs["n_train"] is not None and step >= kwargs["n_train"] / kwargs["batch_size"]:
                break
            x, y = batch  # x: (batch, channels, height, width), y: (batch,)
            if kwargs["device"] == "cuda":
                x = {k: v.cuda() for k, v in x.items()}
            # Run the network on the input.
            network.run(x, time_per_patch=kwargs["time_per_patch"], monitor_spikes=False)
            # visualize_image(x["image"])
            # visualize_spikes(monitors, x)
            # plt.show()

    if kwargs["save"]:
        save_dir = os.path.join(os.path.dirname(__file__), "saves")
        os.makedirs(save_dir, exist_ok=True)
        network.save(os.path.join(save_dir, datetime.now().strftime("%d_%m_%y-%H_%M_%S.model")))


def validate_network(network, val_dataloader, kwargs):
    network.train(False)
    if kwargs["vis_filters"]:
        visualize_filter_weights(network)
        plt.show()
    for step, batch in enumerate(tqdm(val_dataloader)):
        # Get next input sample.
        if step >= kwargs["n_val"]:
            break
        x, y = batch  # x: (batch, channels, height, width), y: (batch,)
        if kwargs["device"] == "cuda":
            x = {k: v.cuda() for k, v in x.items()}
        # Run the network on the input.
        layer_monitors = network.run(
            x, time_per_patch=kwargs["time_per_patch"], monitor_spikes=True
        )
        if kwargs["vis_val_images"]:
            visualize_image(x["image"])
        if kwargs["vis_val_spikes"]:
            visualize_spikes(layer_monitors, x)
        if kwargs["vis_val_patches"]:
            visualize_patches(x, kwargs["patch_shape"])
        plt.show()


if __name__ == "__main__":
    main()