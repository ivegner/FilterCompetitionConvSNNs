import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import make_emnist
from network import Prototype1
from visualization import visualize_spikes

if __name__ == "__main__":
    time_per_patch = 10
    n_epochs = 10
    patch_shape = (3, 3)
    n_filters = 32
    n_l1_features = 64
    n_l2_features = 64
    n_train = 10
    batch_size = 1 # TODO: look into batching

    train_data, val_data = make_emnist(split="balanced")
    # each batch is (time_per_sample, 1, 28, 28)

    network = Prototype1(
        n_input_channels=1,
        patch_shape=patch_shape,
        n_filters=n_filters,
        n_l1_features=n_l1_features,
        n_l2_features=n_l2_features,
    )

    # the dataloaders have to be out here for pickling for some reason
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() - 1
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() - 1
    )

    for epoch in range(10):
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Get next input sample.
            if step > n_train/batch_size:
                break
            x, y = batch  # x: (batch, channels, height, width), y: (batch,)

            # Run the network on the input.
            network.run(x, time_per_patch=time_per_patch)

        visualize_spikes(network)
        network.reset_state_variables()
