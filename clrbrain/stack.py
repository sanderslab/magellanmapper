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
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plotted_imgs = []
    img_size = None
    i = 0
    for f in files:
        print("importing {}".format(f))
        img = io.imread(f)
        img = transform.rescale(img, 0.2, mode="reflect", multichannel=False)
        if img_size is None:
            img_size = img.shape
        plotted_imgs.append([ax.imshow(img)])
        #ax.set_title("")
    
    # export to animated GIF
    out_path = os.path.join(path, name + "_animation.gif")
    print(len(plotted_imgs))
    plt.tight_layout(pad=0.0)
    fig_size = fig.get_size_inches()
    fig.set_size_inches(img_size[1] / fig.dpi, img_size[0] / fig.dpi)
    anim = animation.ArtistAnimation(fig, plotted_imgs, interval=100, repeat_delay=0, blit=False)
    try:
        anim.save(out_path, writer="imagemagick")
    except ValueError as e:
        print(e)
        print("No animation writer available for Matplotlib")
    print("saved animation file to {}".format(out_path))
    plt.show()

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
    cli.main(True)
    animated_gif(cli.filename)
