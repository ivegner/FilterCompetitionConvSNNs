import matplotlib.pyplot as plt
import os
from bindsnet.analysis.plotting import plot_spikes
import numpy as np

plt.ioff()


def visualize_spikes(network, x):
    positions = x["positions"]  # (batch_size, n_patches, 2|4)
    batch_size, n_patches = positions.size(0), positions.size(1)
    if batch_size != 1:
        print(f"Batch size >1 detected ({batch_size}), only visualizing first batch")
    positions = positions[0]


    rec = network.monitor.get()
    spikes = {k: rec[k]["s"] for k in network.layers}

    for layer, s in spikes.items():
        n_timesteps, n_dim = s.size(0), s.size(-1)
        # s.shape: [time_per_patch*n_samples, batch_size*n_patches, n_dim]

        s = s.view(n_timesteps, batch_size, -1, n_dim).transpose(0, 1).view(batch_size, -1, n_dim)
        # s.shape: [batch_size, time, n_dim]
        spikes[layer] = s[0]

    ims, axes = plot_spikes(spikes, figsize=(16,9))

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
    image = image[0].transpose(0, 2)
    fig, ax = plt.subplots()
    ax.imshow(image)

def visualize_patches(x, patch_shape):
    positions = x["positions"]  # (batch_size, n_patches, 2|4)
    patches = x["patches"]  # (batch_size, n_patches, input_size)
    batch_size, n_patches, input_size = patches.shape

    if batch_size != 1:
        print(f"Batch size >1 detected ({batch_size}), only visualizing first image")
    positions = positions[0]
    patches = patches[0].view(n_patches, -1, *patch_shape).permute(0,2,3,1) # channels last
    n_patch_rows = int(np.sqrt(n_patches))
    n_patch_cols = int(np.ceil(n_patches / n_patch_rows))
    fig, axes = plt.subplots(n_patch_rows, n_patch_cols, figsize=(14,14))
    for i, patch in enumerate(patches):
        row = i // n_patch_rows
        col = (i - row*n_patch_cols)
        ax = axes[row, col]
        ax.imshow(patch)
        pos = positions[i]
        ax.set_title(f"({pos.numpy()[0]:.2f}, {pos.numpy()[1]:.2f})", fontdict={"fontsize": 7})
        ax.set_axis_off()

    fig.tight_layout()
