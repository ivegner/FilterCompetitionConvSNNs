import os
import click
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import make_emnist
from network import Prototype1
from visualization import visualize_image, visualize_patches, visualize_spikes


@click.command()
@click.option("-b", "--batch_size", default=1, show_default=True)
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
    "--vis_val_patches",
    default=False,
    show_default=True,
    is_flag=True,
    help="Show image patches for each image during validation visualization",
)
def main(
    batch_size,  # B=1: 600hr for dataset. B=32: 170hr for dataset
    n_epochs,
    n_train,
    n_val,
    time_per_patch,
    n_filters,
    n_l1_features,
    n_l2_features,
    vis_val_images,
    vis_val_spikes,
    vis_val_patches,
):
    time_per_patch = 10
    use_4_position = True
    patch_shape = (5, 5)

    gpu = True
    seed = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        device = "cpu"
        if gpu:
            gpu = False

    num_workers = os.cpu_count() - 1
    torch.set_num_threads(num_workers)
    print("Running on Device = ", device)

    train_data, val_data = make_emnist(
        split="balanced",
        patch_shape=patch_shape,
        time_per_patch=time_per_patch,
        use_4_position=use_4_position,
        patch_intensity=128,
        position_intensity=128,
    )
    # each batch is (batch_size, n_patches, time_per_patch, n_channels*patch_shape)

    network = Prototype1(
        n_input_channels=1,
        patch_shape=patch_shape,
        n_filters=n_filters,
        n_l1_features=n_l1_features,
        n_l2_features=n_l2_features,
    )
    if gpu:
        network = network.to(device)

    # the dataloaders have to be out here for pickling for some reason
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=gpu
    )
    for epoch in range(n_epochs):
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Get next input sample.
            if n_train is not None and step >= n_train / batch_size:
                break
            x, y = batch  # x: (batch, channels, height, width), y: (batch,)
            if gpu:
                x = {k: v.cuda() for k, v in x.items()}
            # Run the network on the input.
            network.run(x, time_per_patch=time_per_patch, monitor_spikes=False)
            # visualize_image(x["image"])
            # visualize_spikes(monitors, x)
            plt.show()

    save_dir = os.path.join(os.path.dirname(__file__), "saves")
    os.makedirs(save_dir, exist_ok=True)
    network.save(os.path.join(save_dir, datetime.now().strftime("%d_%m_%y-%H_%M_%S.model")))

    print("Validation")
    val_dataloader = DataLoader(
        val_data, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(val_dataloader)):
        # Get next input sample.
        if step >= n_val:
            break
        x, y = batch  # x: (batch, channels, height, width), y: (batch,)
        if gpu:
            x = {k: v.cuda() for k, v in x.items()}
        # Run the network on the input.
        layer_monitors = network.run(x, time_per_patch=time_per_patch, monitor_spikes=True)
        if vis_val_images:
            visualize_image(x["image"])
        if vis_val_spikes:
            visualize_spikes(layer_monitors, x)
        if vis_val_patches:
            visualize_patches(x, patch_shape)
        plt.show()


if __name__ == "__main__":
    main()