# Stack manipulations
# Author: David Young, 2017
"""Imports and exports stacks in various formats
"""

import os
import glob
import multiprocessing as mp

import numpy as np
from skimage import transform
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import animation

from clrbrain import cli
from clrbrain import plot_2d

def _import_img(i, path):
    print("importing {}".format(path))
    img = io.imread(path)
    img = transform.rescale(img, 0.1, mode="reflect", multichannel=False)
    return i, img, img.shape

def animated_gif(path):
    """Builds an animated GIF from a stack of images in a directory.
    
    Writes the animated file to the parent directory of path.
    
    Params:
        path: Path to the image directory. All images from this directory
            will be imported in Python sorted order.
    """
    # ascending order of all files in the directory
    files = sorted(glob.glob(os.path.join(path, "*")))#[::10]
    print(files)
    num_files = len(files)
    if num_files < 1:
        return None
    
    # set paths
    parent_path = os.path.dirname(files[0])
    name = os.path.dirname(files[0])
    tmp_path = os.path.join(parent_path, name + "_tmp.npy")
    
    # Matplotlib figure for building the animation
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # import the images as Matplotlib artists via multiprocessing
    plotted_imgs = [None for i in range(num_files)]
    img_size = [] # will keep last image's size
    i = 0
    pool = mp.Pool()
    pool_results = []
    for f in files:
        pool_results.append(pool.apply_async(_import_img, args=(i, f)))
        i += 1
    for result in pool_results:
        i, img, img_size = result.get()
        plotted_imgs[i] = [ax.imshow(
            img, cmap=plot_2d.CMAP_GRBK, vmin=0, vmax=0.1)]
    pool.close()
    pool.join()
    
    # need to compress layout to fit image only
    #print(len(plotted_imgs))
    plt.tight_layout(pad=0.0) # leaves some space for some reason
    fig_size = fig.get_size_inches()
    fig.set_size_inches(img_size[1] / fig.dpi, img_size[0] / fig.dpi)
    
    # export to animated GIF
    out_path = os.path.join(path, name + "_animation.gif")
    anim = animation.ArtistAnimation(
        fig, plotted_imgs, interval=100, repeat_delay=0, blit=False)
    try:
        anim.save(out_path, writer="imagemagick")
    except ValueError as e:
        print(e)
        print("No animation writer available for Matplotlib")
    print("saved animation file to {}".format(out_path))
    #plt.show()

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
    cli.main(True)
    animated_gif(cli.filename)
