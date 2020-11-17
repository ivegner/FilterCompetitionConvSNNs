from operator import pos
import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, AdaptiveLIFNodes, Nodes
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
        nu=1e-4
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
        INH_WEIGHT = 1
        THRESH = -63
        NORM_FILTER = 5
        NORM_L1 = 15
        NORM_L2 = 100
        ADAPTIVE = False
        input_size = np.prod([n_input_channels, *patch_shape])
        position_size = 4 if use_4_position else 2

        input_layer = Input(input_size, traces=True)
        position_layer = Input(position_size, traces=True)
        filter_layer = ClampingNodes(n_filters, traces=True, adaptive=ADAPTIVE, thresh=THRESH)
        feature_l1 = ClampingNodes(n_l1_features, traces=True, adaptive=ADAPTIVE, thresh=THRESH)
        feature_l2 = ClampingNodes(
            n_l2_features, traces=True, adaptive=ADAPTIVE, thresh=THRESH, tc_decay=1e10
        )

        input_filter_connection = Connection(
            input_layer, filter_layer, update_rule=PostPre, norm=NORM_FILTER, nu=nu
        )
        # norm by amount of inputs
        filter_l1_connection = Connection(
            filter_layer,
            feature_l1,
            update_rule=PostPre,
            norm=NORM_L1 * (n_filters / (n_filters + position_size)),
            nu=nu,
        )
        position_l1_connection = Connection(
            position_layer,
            feature_l1,
            update_rule=PostPre,
            norm=NORM_L1 * (position_size / (n_filters + position_size)),
            nu=nu,
        )
        l1_l2_connection = Connection(
            feature_l1, feature_l2, update_rule=PostPre, norm=NORM_L2, nu=nu
        )

        # Inhibitory connections
        inh_filter_connection = Connection(
            filter_layer, filter_layer, w=INH_WEIGHT * (torch.eye(filter_layer.n) - 1)
        )
        inh_l1_connection = Connection(
            feature_l1, feature_l1, w=INH_WEIGHT * (torch.eye(feature_l1.n) - 1)
        )
        inh_l2_connection = Connection(
            feature_l2, feature_l2, w=INH_WEIGHT * (torch.eye(feature_l2.n) - 1)
        )

        self.add_layer(input_layer, name="input")
        self.add_layer(position_layer, name="position")
        self.add_layer(filter_layer, name="filter")
        self.add_layer(feature_l1, name="l1")
        self.add_layer(feature_l2, name="l2")
        self.add_connection(input_filter_connection, source="input", target="filter")
        self.add_connection(filter_l1_connection, source="filter", target="l1")
        self.add_connection(position_l1_connection, source="position", target="l1")
        self.add_connection(l1_l2_connection, source="l1", target="l2")

        self.add_connection(inh_filter_connection, source="filter", target="filter")
        self.add_connection(inh_l1_connection, source="l1", target="l1")
        self.add_connection(inh_l2_connection, source="l2", target="l2")

        self.use_4_position = use_4_position
        self.n_input_channels = n_input_channels
        self.patch_shape = patch_shape
        self.dt = dt

    def run(self, encoded_x, time_per_patch, monitor_spikes=False):
        # x: {"encoded_patches", "encoded_positions", "patches", "positions"}
        # x["encoded_patches"]: (batch_size, n_patches, time_per_patch, input_size)
        # x["encoded_positions"]: (batch_size, n_patches, time_per_patch, 2|4)
        batch_size, n_patches = encoded_x["encoded_patches"].shape[:2]

        monitors = None
        if monitor_spikes:
            # Create monitors for each batch separately, if requested
            # skipped at train due to overhead
            monitors = {}
            for layer in self.layers:
                monitor = Monitor(self.layers[layer], state_vars=["s"])
                self.add_monitor(monitor, name=layer)
                monitors[layer] = monitor

        for patch_idx in range(n_patches):
            print(patch_idx, f"{patch_idx/n_patches:.2f}", end="\r")
            _raw_patch = encoded_x["patches"][:, patch_idx]
            _raw_position = encoded_x["positions"][:, patch_idx]
            patch = encoded_x["encoded_patches"][:, patch_idx].transpose(0, 1)  # batch on dim 1
            position = encoded_x["encoded_positions"][:, patch_idx].transpose(0, 1)
            super().run({"input": patch, "position": position}, time=time_per_patch)
            self._patch_reset()  # reset all voltages
        self._batch_reset()  # reset all voltages including L2, and clear monitors

        return monitors

    def _patch_reset(self):
        """Reset all variables except L2"""
        for layer in self.layers:
            if layer != "l2":
                self.layers[layer].reset_state_variables()

        for connection in self.connections:
            if "l2" not in connection:
                self.connections[connection].reset_state_variables()

    def _batch_reset(self):
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        self.monitors = {}  # clear list of monitors each batch

    # def to(self, *args, **kwargs):
    #     # mmm, fixing library bugs
    #     for k, rec in self.monitor.recording.items():
    #         for v in rec:
    #             self.monitor.recording[k][v] = self.monitor.recording[k][v].to(
    #                 *args, **kwargs
    #             )
    #     return super().to(*args, **kwargs)

    # def reset_state_variables(self):
    #     """Reset all state variables. Replaces buggy library version"""
    #     self._sample_reset()
    #     for k, rec in self.monitor.recording.items():
    #         for v in rec:
    #             self.monitor.recording[k][v] = torch.Tensor().to(self.monitor.recording[k][v].device)


class ClampingNodes(AdaptiveLIFNodes):
    """
    Node layer that clamps spikes per timestep to only those within `clamp_epsilon` mV of the
    maximum-voltage neuron. Can be adaptive or not.
    """

    def __init__(self, *args, adaptive=False, clamp_epsilon=0.0, **kwargs):
        """

        Node layer that clamps spikes per timestep to only those within `clamp_epsilon*100`% of the
            maximum-voltage neuron. Can be adaptive or not.

        **Keyword arguments:**
        :param bool `adaptive`: Whether to make the neurons adaptive (see AdaptiveLIFNodes), defaults to False
        :param float `clamp_epsilon`: neurons within `clamp_epsilon*100`% of the maximum-voltage neuron will spike
            (if they are above the threshold themselves).
            Defaults to 0.0 (only the highest voltage spikes). A value of 1.0 produces behavior identical to
            the original (Adaptive)LIFNodes.
        """
        super().__init__(*args, **kwargs)
        self.adaptive = adaptive
        self.clamp_eps = clamp_epsilon

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.adaptive and self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * x

        # Decrement refractory counters.
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)

        potential_s = self.v - (self.thresh + self.theta)
        # self.s = self.s.masked_fill(potential_s < 0, 0).bool()
        max_v_surplus = torch.max(potential_s)
        if max_v_surplus < 0.0:
            max_v_surplus = 0.0
        self.s = potential_s >= (1 - self.clamp_eps) * max_v_surplus
        # max_idx = torch.argmax(potential_s)
        # self.s.zero_()
        # self.s = self.s.bool()
        # self.s[0, max_idx] = (potential_s[0, max_idx] >= 1.)

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.adaptive and self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)

        # voltage clipping to lowerbound
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        Nodes.forward(self, x)
