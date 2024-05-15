import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from network.params import parameters


class MplColorHelper:
    def __init__(self, cmap_name: str, start_val: float = 0., stop_val: float = 1.):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val: np.ndarray):
        return self.scalarMap.to_rgba(val)

if __name__ == '__main__':

    # values for the voxels
    my_colormap = MplColorHelper('Blues')
    my_alpha = 0.9

    # axis definitions
    axes = [9, 15, 14]
    filled = np.zeros(axes, dtype=bool)

    # make colors
    colors = np.zeros(axes + [4])
    values = [0, 0.4, 0.8, 1.0]

    gradient_fill = np.random.choice(values, axes[-1], p=(0.65, 0.2, 0.1, 0.05))
    color_fill = my_colormap.get_rgb(val=gradient_fill)
    color_fill[:, 3] = my_alpha

    # set colors in voxel
    filled[4, 9] = True
    colors[4, 9, :, :] = color_fill
    # upscale the above voxel image, leaving gaps
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(filled, facecolors=colors, edgecolors=(0, 0, 0, 0.2))
    ax.voxels(np.ones_like(filled), facecolors=(1, 1, 1, 0.02), edgecolors=(0, 0, 0, 0.02))
    ax.set_axis_off()
    ax.view_init(elev=30, azim=45, roll=0)
    plt.savefig('StrD1_cube.png', bbox_inches='tight', pad_inches=0)
