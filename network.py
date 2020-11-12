import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import NetworkMonitor
import torch.nn.functional as F
from bindsnet.encoding import PoissonEncoder


class Prototype1(Network):
    def __init__(
        self, n_input_channels, patch_shape, n_filters, n_l1_features, n_l2_features, dt=1, use_4_position=True
    ):
        """

        Prototype Network

        **Args:**
        :param int `n_input_channels`: Number of input channels
        :param int `patch_shape`: (h, w) shape of convolutional patches to generate, in pixels.
            Analogous to filter shape of conventional CNNs.
        :param int `n_filters`: Number of filter neurons.
        :param int `n_l1_features`: Number of neurons in the L1 feature discovery layer.
        :param int `n_l2_features`: Number of neurons in the L2 feature discovery layer.

        **Kwargs:**
        :param int `dt`: Ms per tick, default is 1
        :param bool `use_4_position`: Whether to use 4 position neurons instead of 2. Default is True.

        """
        super().__init__(dt=dt)
        input_size = np.prod([n_input_channels, *patch_shape])
        input_layer = Input(input_size, traces=True)
        position_layer = Input(4 if use_4_position else 2, traces=True)
        filter_layer = LIFNodes(n_filters, traces=True)
        feature_l1 = LIFNodes(n_l1_features, traces=True)
        feature_l2 = LIFNodes(n_l2_features, traces=True)

        # TODO: normalize connections
        # TODO: Inhibitory connections
        input_filter_connection = Connection(input_layer, filter_layer)
        position_filter_connection = Connection(position_layer, filter_layer)
        filter_l1_connection = Connection(filter_layer, feature_l1)
        l1_l2_connection = Connection(feature_l1, feature_l2)

        self.add_layer(input_layer, name="input")
        self.add_layer(position_layer, name="position")
        self.add_layer(filter_layer, name="filter")
        self.add_layer(feature_l1, name="l1")
        self.add_layer(feature_l2, name="l2")
        self.add_connection(input_filter_connection, source="input", target="filter")
        self.add_connection(position_filter_connection, source="position", target="filter")
        self.add_connection(filter_l1_connection, source="filter", target="l1")
        self.add_connection(l1_l2_connection, source="l1", target="l2")
        # connect L2 to filters too maybe?

        self.monitor = NetworkMonitor(self)
        self.add_monitor(self.monitor, "monitor")

        self.use_4_position = use_4_position
        self.n_input_channels = n_input_channels
        self.patch_shape = patch_shape
        self.dt = dt

    def run(self, x, time_per_patch):
        # x: (batch, channels, height, width)
        encoded_x = self._make_patches(x, time_per_patch=time_per_patch)
        # x["encoded_patches"]: (batch_size, n_patches, time_per_patch, input_size)
        # x["encoded_positions"]: (batch_size, n_patches, time_per_patch, 2|4)
        for patch_idx in range(encoded_x["encoded_patches"].size(1)):
            _raw_patch = encoded_x["patches"][:, patch_idx]
            _raw_position = encoded_x["positions"][:, patch_idx]
            patch = encoded_x["encoded_patches"][:, patch_idx].transpose(0, 1)  # batch on dim 1
            position = encoded_x["encoded_positions"][:, patch_idx].transpose(0, 1)
            super().run({"input": patch, "position": position}, time=time_per_patch)

    def _make_patches(self, x, time_per_patch, patch_intensity=512, position_intensity=512):
        """Make patches

        x: 4D tensor of size (batch, channels, height, width), normalized to [0,1]
        """
        # for each image (C, H, W), we compute its im2col with patches of shape patch_shape
        # then, encode each patch with Poisson
        PADDING = 0
        STRIDE = 1

        batch_size = x.shape[0]
        channels = x.shape[1]
        if channels != self.n_input_channels:
            raise ValueError(
                f"Number of channels in the image ({channels}) doesn't match n_input_channels {self.n_input_channels}"
            )
        encoder = PoissonEncoder(time=time_per_patch, dt=self.dt)

        patch_h, patch_w = self.patch_shape
        input_size = self.n_input_channels * patch_h * patch_w

        # dilation not implemented
        patches = F.unfold(x, kernel_size=self.patch_shape, padding=PADDING, stride=STRIDE)
        n_patches = patches.shape[-1]
        patches = patches.permute(0, 2, 1).view(-1, input_size)  # (B * n_patches, input_size)

        patch_positions = get_im2col_indices(
            x.shape, patch_h, patch_w, padding=PADDING, stride=STRIDE
        )
        # normalize positions to [0,1]
        patch_positions /= torch.max(patch_positions, axis=0, keepdim=True)[0]  # (n_patches, 2)
        if self.use_4_position:
            inv_patch_positions = 1 - patch_positions
            patch_positions = torch.cat([patch_positions, inv_patch_positions], dim=1) # (n_patches, 4)
        assert len(patch_positions) == n_patches

        # encode each patch separately
        encoded_patches = torch.stack([encoder(p * patch_intensity) for p in patches])
        # encoded_patches: (B * n_patches, time_per_patch, input_size)
        # reshape to proper shape
        encoded_patches = encoded_patches.view(batch_size, n_patches, time_per_patch, input_size)

        # encode positions
        encoded_positions = torch.stack([encoder(p * position_intensity) for p in patch_positions])
        # encoded_positions: (n_patches, time, 2|4)
        # add batch dimension
        encoded_positions = encoded_positions.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return dict(
            patches=patches.view(batch_size, n_patches, input_size),
            encoded_patches=encoded_patches,
            positions=patch_positions.unsqueeze(0).expand(batch_size, -1, -1),
            encoded_positions=encoded_positions,
        )


# adapted from
# https://fdsmlhn.github.io/2017/11/02/Understanding%20im2col%20implementation%20in%20Python(numpy%20fancy%20indexing)/
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
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