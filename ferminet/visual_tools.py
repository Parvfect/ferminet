

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg


import matplotlib.pyplot as plt
import numpy as np
import math


def plot_electron_histograms(pos, bins=50, max_cols=3, figsize=(12, 12)):
    """
    Plot 2D histograms of electron positions from QMC simulation.
    Each electron gets its own subplot, arranged in a grid with square axes.
    
    Parameters
    ----------
    pos : np.ndarray
        Shape (batch_size, n_electrons*3).
        Coordinates stored as (x, y, z) for each electron.
    bins : int
        Number of histogram bins per axis.
    max_cols : int
        Maximum number of columns in the subplot grid.
    figsize : tuple
        Size of the matplotlib figure.
    """

    if pos.shape[0] == 1:
        pos = pos.squeeze(0)  # shape: (batch_size, n_electrons*3)

    batch_size, total_dim = pos.shape
    n_electrons = total_dim // 3

    ncols = min(n_electrons, max_cols)
    nrows = math.ceil(n_electrons / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # flatten in case of 2D array

    for i in range(n_electrons):
        ax = axes[i]
        # extract x,y for electron i
        x = pos[:, 3*i]
        y = pos[:, 3*i + 1]

        h = ax.hist2d(x, y, bins=50, cmap="plasma")
        fig.colorbar(h[3], ax=ax, label="Counts")
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Electron {i+1}")
        ax.set_aspect("equal")  # make plots square

    # Hide any unused axes
    for j in range(n_electrons, len(axes)):
        axes[j].set_visible(False)

    fig.set_canvas(FigureCanvasAgg(fig))
    fig.canvas.draw()
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    else:
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        arr = np.asarray(buf).reshape(h, w, 4)[..., :3]  # drop alpha

    return arr




def save_scatter_plot(pos, prob_density):
    """Call from within step function?"""

    pos = pos[0]
    prob_density = prob_density[0]
    n_elec = int(pos.shape[1] / 3)
    electron_pos = [
        pos[:, 3*i:3*(i+1)] for i in range(n_elec)]

    # Normalize probabilities to [0, 1] for coloring
    norm = colors.Normalize(
        vmin=prob_density.min(), vmax=prob_density.max())


    def truncate_colormap(cmap, minval=0.3, maxval=1.0, n=256):
        new_cmap = cm.colors.LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    cmaps = [
        truncate_colormap(plt.cm.Reds, 0.3, 1.0),
        truncate_colormap(plt.cm.Blues, 0.3, 1.0),
        truncate_colormap(plt.cm.Greens, 0.3, 1.0),
        truncate_colormap(plt.cm.Purples, 0.3, 1.0),
        truncate_colormap(plt.cm.Oranges, 0.3, 1.0),
        plt.cm.cividis, plt.cm.plasma, plt.cm.viridis
    ]

    # Pick distinct colormaps (repeat if more electrons than colormaps)
    """
    cmaps = [
        plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges,
        plt.cm.cividis, plt.cm.plasma, plt.cm.viridis, plt.cm.inferno, plt.cm.copper
    ]
    """

    plot_dict = {
        "pos": pos,
        "prob_density": prob_density
    }

    # Some different marker shapes (will cycle if > len(markers))
    markers = ['o', '^', 's', 'D', 'v', '>', '<', 'p', 'h', '*']

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    scatters = []
    for i in range(n_elec):
        cmap = cmaps[i % len(cmaps)]
        marker = markers[i % len(markers)]
        
        sc = ax.scatter(
            electron_pos[i][:,0], electron_pos[i][:,1], electron_pos[i][:,2],
            c=cmap(norm(prob_density)),
            s=5, alpha=0.6, marker=marker, label=f"{i}"
        )
        scatters.append(sc)

    # --- Add nuclei from geometry ---
    #atoms = data.atoms[0][0]  # should be shape (n_atoms, 3)
    #charges = data.charges[0][0]


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Electron density (colored by |Ψ|²)")
    ax.legend(markerscale=3)

    # Shared colorbar (take from first scatter)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("|Ψ|² (relative scale)")

    """
    fig.set_canvas(FigureCanvasAgg(fig))
    fig.canvas.draw()
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    else:
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        arr = np.asarray(buf).reshape(h, w, 4)[..., :3]  # drop alpha
    """

    wandb.log({"scatter": wandb.Image(fig)})
    return []
