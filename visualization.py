import matplotlib.pyplot as plt
from bindsnet.analysis.plotting import plot_spikes
plt.ioff()

def visualize_spikes(network, batch_size):
    rec = network.monitor.get()
    rec_l2 = network.L2.monitor.get()
    spikes = {k: rec[k]["s"] for k in network.layers}
    spikes["l2"] = rec_l2["l2"]["s"]
    for layer, s in spikes.items():
        n_timesteps, n_dim = s.size(0), s.size(-1)
        # s.shape: [time_per_patch*n_samples, batch_size*n_patches, n_dim]

        s = s.view(n_timesteps, batch_size, -1, n_dim).transpose(0,1).view(batch_size, -1, n_dim)
        # s.shape: [batch_size, time, n_dim]
        if s.size(0) != 1:
            print(f"Batch size >1 detected ({s.size(0)}), only visualizing first batch")
        spikes[layer] = s[0]
    plot_spikes(spikes)
    plt.savefig("./vis/spikes.png")
    plt.show()