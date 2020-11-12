import matplotlib.pyplot as plt
from bindsnet.analysis.plotting import plot_spikes
plt.ioff()

def visualize_spikes(network):
    rec = network.monitor.get()
    spikes = {}
    for layer in ["input", "position", "filter", "l1", "l2"]:
        s = rec[layer]["s"]
        if s.size(1) != 1:
            print(f"Batch size >1 detected ({s.size(1)}), only visualizing first batch")
        spikes[layer] = s[:, 0] # first batch
    plot_spikes(spikes)
    plt.show()