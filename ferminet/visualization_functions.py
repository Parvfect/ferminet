



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

n_elec = 6
electron_pos = [pos[:, 3*i:3*(i+1)] for i in range(n_elec)]

# Normalize probabilities to [0, 1] for coloring
norm = colors.Normalize(vmin=prob_density_.min(), vmax=prob_density_.max())


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

# Some different marker shapes (will cycle if > len(markers))
markers = ['o', '^', 's', 'D', 'v', '>', '<', 'p', 'h', '*']

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

scatters = []
for i in range(n_elec):
    cmap = cmaps[i % len(cmaps)]
    marker = markers[i % len(markers)]
    if i == 6:
        sc = ax.scatter(
        electron_pos[i][:,0], electron_pos[i][:,1], electron_pos[i][:,2],
        c=cmap(norm(prob_density_)),
        s=5, alpha=0.6, marker='o', label=f"e+"
    )
    elif (i+1) % 3 == 0:
        sc = ax.scatter(
        electron_pos[i][:,0], electron_pos[i][:,1], electron_pos[i][:,2],
        c=cmap(norm(prob_density_)),
        s=5, alpha=0.6, marker='*', label=f"Li-2s-e"
    )
    else:

        sc = ax.scatter(
            electron_pos[i][:,0], electron_pos[i][:,1], electron_pos[i][:,2],
            c=cmap(norm(prob_density_)),
            s=5, alpha=0.6, marker=marker, label=f"Li-1s-e"
        )
    scatters.append(sc)

# --- Add nuclei from geometry ---
atoms = data.atoms[0][0]  # should be shape (n_atoms, 3)
charges = data.charges[0][0]


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Li2-e+ electron density (colored by |Ψ|²)")
ax.legend(markerscale=3)

# Shared colorbar (take from first scatter)
sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("|Ψ|² (relative scale)")

plt.show()
