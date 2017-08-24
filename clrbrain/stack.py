# Stack manipulations
# Author: David Young, 2017
"""Imports and exports stacks in various formats
"""

import os
import glob

import numpy as np
from skimage import transform
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import animation

from clrbrain import cli

def animated_gif(path):
    files = sorted(glob.glob(os.path.join(path, "*")))
    print(files)
    num_files = len(files)
    if num_files < 1:
        return None
    parent_path = os.path.dirname(files[0])
    name = os.path.dirname(files[0])
    tmp_path = os.path.join(parent_path, name + "_tmp.npy")
    image3d = None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    plotted_imgs = []
    i = 0
    for f in files:
        print("importing {}".format(f))
        img = io.imread(f)
        img = transform.rescale(img, 0.2, mode="reflect", multichannel=False)
        image3d = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=img.dtype, 
            shape=(1, len(files), *img.shape))
        image3d[i] = img
        plotted_imgs.append([ax.imshow(img)])
        ax.set_title("")
    
    # export to animated GIF
    out_path = os.path.join(parent_path, "_animation.gif")
    print(len(plotted_imgs))
    anim = animation.ArtistAnimation(fig, plotted_imgs, interval=100, repeat_delay=0, blit=False)
    #anim.save(out_path)
    print("saved animation file to {}".format(out_path))
    plt.show()
    return image3d

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
    cli.main(True)
    animated_gif(cli.filename)
