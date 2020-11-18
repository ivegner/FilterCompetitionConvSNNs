import matplotlib.pyplot as plt
import os
from bindsnet.analysis.plotting import plot_spikes
import numpy as np
from torchvision.utils import make_grid
from matplotlib.cm import get_cmap
import torch

plt.ioff()


def visualize_spikes(monitors, x):
    positions = x["positions"].cpu()  # (batch_size, n_patches, 2|4)
    batch_size, n_patches = positions.size(0), positions.size(1)
    if batch_size != 1:
        print(f"Batch size >1 detected ({batch_size}), only visualizing first batch")
    positions = positions[0]

    spikes = {}

    for layer, mon in monitors.items():
        s = mon.get("s")
        n_timesteps, n_dim = s.size(0), s.size(-1)
        # s.shape: [time_per_patch*n_samples, batch_size*n_patches, n_dim]

        s = s.view(n_timesteps, batch_size, -1, n_dim).transpose(0, 1).view(batch_size, -1, n_dim)
        # s.shape: [batch_size, time, n_dim]
        spikes[layer] = s[0].cpu()

    ims, axes = plot_spikes(spikes, figsize=(16, 9))

    time = spikes["l2"].size(0)
    time_per_patch = time / n_patches

    patches_per_row = int(np.sqrt(n_patches))
    position_strings = [f"({x.numpy()[0]:.2f}, {x.numpy()[1]:.2f})" for x in positions]
    ticks = list(zip(np.arange(0, time, time_per_patch), position_strings))[::patches_per_row]
    tick_idxs, tick_labels = zip(*ticks)
    for ax in axes:
        ax.set_xticks(tick_idxs)
        ax.set_xticks(np.arange(0, time, time_per_patch), minor=True)
        ax.set_xticklabels(tick_labels, fontdict={"fontsize": 7})
        ax.set_xlim(left=0, right=time)

    vis_dir_path = os.path.join(os.path.dirname(__file__), "vis")
    os.makedirs(vis_dir_path, exist_ok=True)
    plt.savefig(os.path.join(vis_dir_path, "spikes.png"))


def visualize_image(image):
    # image: (batch, channels, height, width)
    assert len(image.shape) == 4
    if image.shape[0] != 1:
        print(f"Batch size >1 detected ({image.shape[0]}), only visualizing first image")
    image = image[0].transpose(0, 2).cpu()
    fig, ax = plt.subplots()
    ax.imshow(image)


def visualize_patches(x, patch_shape):
    positions = x["positions"].cpu()  # (batch_size, n_patches, 2|4)
    patches = x["patches"].cpu()  # (batch_size, n_patches, input_size)
    batch_size, n_patches, input_size = patches.shape

    if batch_size != 1:
        print(f"Batch size >1 detected ({batch_size}), only visualizing first image")
    positions = positions[0]
    patches = patches[0].view(n_patches, -1, *patch_shape).permute(0, 2, 3, 1)  # channels last
    n_patch_rows = int(np.sqrt(n_patches))
    n_patch_cols = int(np.ceil(n_patches / n_patch_rows))
    fig, axes = plt.subplots(n_patch_rows, n_patch_cols, figsize=(14, 14))
    for i, patch in enumerate(patches):
        row = i // n_patch_rows
        col = i - row * n_patch_cols
        ax = axes[row, col]
        ax.imshow(patch)
        pos = positions[i]
        ax.set_title(f"({pos.numpy()[0]:.2f}, {pos.numpy()[1]:.2f})", fontdict={"fontsize": 7})
        ax.set_axis_off()

    fig.tight_layout()


def visualize_filter_weights(network):
    filter_weights = network.connections[("input", "filter")].w  # (c*h*w, n_filters)
    patch_shape = network.patch_shape
    n_filters = filter_weights.size(1)
    filter_weights = filter_weights.view(-1, *patch_shape, n_filters)
    filter_weights = filter_weights.permute(3, 0, 1, 2).cpu()  # (n, c, h, w)
    if filter_weights.size(1) == 1:
        # one channel, cmap it
        cmap = get_cmap("bwr")

        # working around weird bug in cmap -- first image comes out blank...
        filter_weights = torch.cat(
            [torch.zeros_like(filter_weights[0]).unsqueeze(0), filter_weights]
        )
        filter_weights = cmap(filter_weights.squeeze(1))[1:]
        filter_weights = filter_weights[:, :, :, :3]  # just rgb, no a
        filter_weights = torch.Tensor(filter_weights).permute(0, 3, 1, 2)

    # grid = make_grid(filter_weights, normalize=True, range=(-1, 1))
    grid = make_grid(filter_weights)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(grid.permute(1, 2, 0).cpu())
