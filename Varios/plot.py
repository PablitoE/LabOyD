import matplotlib.pyplot as plt
import numpy as np


def boxplot_by_bins(x, y, bins=12, ax=None, decimals=2, *args, **kwargs):
    bin_edges = np.linspace(x.min(), x.max(), bins + 1)
    bin_centers = np.round((bin_edges[1:] + bin_edges[:-1]) / 2, decimals=decimals)

    idx_groups = np.digitize(x, bin_edges)
    groups = [y[idx_groups == i] for i in range(1, len(bin_edges))]

    if ax is not None:
        ax: plt.Axes
        return ax.boxplot(groups, positions=bin_centers, *args, **kwargs)
    return plt.boxplot(groups, positions=bin_centers, *args, **kwargs)
