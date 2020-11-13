import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import NetworkMonitor, Monitor
from bindsnet.learning import PostPre


class Prototype1(Network):
    def __init__(
        self,
        n_input_channels,
        patch_shape,
        n_filters,
        n_l1_features,
        n_l2_features,
        dt=1,
        use_4_position=True,
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
        INH_WEIGHT = 0.1
        input_size = np.prod([n_input_channels, *patch_shape])
        input_layer = Input(input_size, traces=True)
        position_layer = Input(4 if use_4_position else 2, traces=True)
        filter_layer = LIFNodes(n_filters, traces=True)
        feature_l1 = LIFNodes(n_l1_features, traces=True)
        self.L2 = L2(n_l1_features, n_l2_features, dt=dt)

        # TODO: normalize connections
        input_filter_connection = Connection(input_layer, filter_layer, update_rule=PostPre)
        filter_l1_connection = Connection(filter_layer, feature_l1, update_rule=PostPre)
        position_l1_connection = Connection(position_layer, feature_l1, update_rule=PostPre)

        # Inhibitory connections
        inh_filter_connection = Connection(
            filter_layer, filter_layer, w=INH_WEIGHT * (torch.eye(filter_layer.n) - 1)
        )
        inh_l1_connection = Connection(
            feature_l1, feature_l1, w=INH_WEIGHT * (torch.eye(feature_l1.n) - 1)
        )

        self.add_layer(input_layer, name="input")
        self.add_layer(position_layer, name="position")
        self.add_layer(filter_layer, name="filter")
        self.add_layer(feature_l1, name="l1")
        self.add_connection(input_filter_connection, source="input", target="filter")
        self.add_connection(filter_l1_connection, source="filter", target="l1")
        self.add_connection(position_l1_connection, source="position", target="l1")

        self.add_connection(inh_filter_connection, source="filter", target="filter")
        self.add_connection(inh_l1_connection, source="l1", target="l1")

        # monitor whole network
        self.monitor = NetworkMonitor(self)
        self.add_monitor(self.monitor, name="monitor")

        # monitor just L1 and reset every batch
        self._batch_l1_monitor = Monitor(feature_l1, ["s"])
        self.add_monitor(self._batch_l1_monitor, name="l1_monitor")

        self.use_4_position = use_4_position
        self.n_input_channels = n_input_channels
        self.patch_shape = patch_shape
        self.dt = dt

    def run(self, encoded_x, time_per_patch):
        # x: {"encoded_patches", "encoded_positions", "patches", "positions"}
        # x["encoded_patches"]: (batch_size, n_patches, time_per_patch, input_size)
        # x["encoded_positions"]: (batch_size, n_patches, time_per_patch, 2|4)

        batch_size, n_patches = encoded_x["encoded_patches"].shape[:2]
        # Flatten. Effectively treat each patch as a batch
        patches = encoded_x["encoded_patches"].view(batch_size * n_patches, time_per_patch, -1)
        positions = encoded_x["encoded_positions"].view(batch_size * n_patches, time_per_patch, -1)

        super().run(
            {"input": patches.permute(1, 0, 2), "position": positions.permute(1, 0, 2)},
            time=time_per_patch,
        )
        l1_out = self._batch_l1_monitor.get("s")
        l1_dim = l1_out.shape[-1]
        # l1_out.shape: [time_per_patch, batch_size*n_patches, l1_dim]
        l1_out = (
            l1_out.view(time_per_patch, batch_size, n_patches, l1_dim) # reshape into original
            .transpose(0, 1) # swap batch and time
            .view(batch_size, n_patches * time_per_patch, l1_dim) # unite time and patches
            .transpose(0, 1) # time first, batches second
        )
        l2_out = self.L2.run(l1_out)

        # # for patch_idx in range(encoded_x["encoded_patches"].size(1)):
        #     _raw_patch = encoded_x["patches"][:, patch_idx]
        #     _raw_position = encoded_x["positions"][:, patch_idx]
        #     patch = encoded_x["encoded_patches"][:, patch_idx].transpose(0, 1)  # batch on dim 1
        #     position = encoded_x["encoded_positions"][:, patch_idx].transpose(0, 1)
        #     super().run({"input": patch, "position": position}, time=time_per_patch)
        #     self._patch_reset() # reset voltages in all layers except L2
        self._sample_reset()  # reset all voltages

    # def _patch_reset(self):
    #     """Reset all variables except L2"""
    #     for layer in self.layers:
    #         if layer != "l2":
    #             self.layers[layer].reset_state_variables()

    #     for connection in self.connections:
    #         if "l2" not in connection:
    #             self.connections[connection].reset_state_variables()

    def _sample_reset(self):
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        # reset monitor for next batch
        self._batch_l1_monitor.reset_state_variables()


class L2(Network):
    def __init__(
        self,
        n_l1_features,
        n_l2_features,
        dt=1,
    ):
        """

        The L2 part of the network

        :param int `n_l1_features`: Number of neurons in the L1 feature discovery layer.
        :param int `n_l2_features`: Number of neurons in the L2 feature discovery layer.

        **Kwargs:**
        :param int `dt`: Ms per tick, default is 1
        :param bool `use_4_position`: Whether to use 4 position neurons instead of 2. Default is True.

        """
        super().__init__(dt=dt)
        INH_WEIGHT = 0.1
        feature_l1 = Input(n_l1_features, traces=True)
        feature_l2 = LIFNodes(n_l2_features, traces=True)

        # TODO: normalize connections
        l1_l2_connection = Connection(feature_l1, feature_l2, update_rule=PostPre)

        # Inhibitory connections
        inh_l2_connection = Connection(
            feature_l2, feature_l2, w=INH_WEIGHT * (torch.eye(feature_l2.n) - 1)
        )
        self.add_layer(feature_l1, name="l1")
        self.add_layer(feature_l2, name="l2")
        self.add_connection(l1_l2_connection, source="l1", target="l2")
        self.add_connection(inh_l2_connection, source="l2", target="l2")

        self.monitor = NetworkMonitor(self)
        self.add_monitor(self.monitor, name="monitor")

        self.dt = dt

    def run(self, l1_outputs):
        l1_outputs = l1_outputs.float()
        # Experimental hack: filter by nonzero timesteps
        # nonzero_l1_timesteps = l1_outputs.nonzero()[:, 0]
        # l1_outputs = l1_outputs[nonzero_l1_timesteps]
        super().run({"l1": l1_outputs.float()}, time=l1_outputs.size(0))
        self._sample_reset()  # reset all voltages

    def _sample_reset(self):
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())