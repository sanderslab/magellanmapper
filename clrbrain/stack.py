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
from clrbrain import importer

def _import_img(i, path):
    print("importing {}".format(path))
    img = io.imread(path)
    img = transform.rescale(img, 0.1, mode="reflect", multichannel=False)
    return i, img, img.shape

def _process_plane(i, plane):
    print("processing plane {}".format(i))
    img = transform.rescale(plane, 0.1, mode="reflect", multichannel=False)
    return i, img, img.shape

def _build_animated_gif(images, out_path, process_fnc):
    """Builds an animated GIF from a stack of images in a directory.
    
    Writes the animated file to the parent directory of path.
    
    Params:
        path: Path to the image directory. All images from this directory
            will be imported in Python sorted order.
    """
    # ascending order of all files in the directory
    num_images = len(images)
    if num_images < 1:
        return None
    
    # Matplotlib figure for building the animation
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # import the images as Matplotlib artists via multiprocessing
    plotted_imgs = [None for i in range(num_images)]
    img_size = [] # will keep last image's size
    i = 0
    pool = mp.Pool()
    pool_results = []
    for image in images:
        pool_results.append(pool.apply_async(process_fnc, args=(i, image)))
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
    anim = animation.ArtistAnimation(
        fig, plotted_imgs, interval=100, repeat_delay=0, blit=False)
    try:
        anim.save(out_path, writer="imagemagick")
    except ValueError as e:
        print(e)
        print("No animation writer available for Matplotlib")
    print("saved animation file to {}".format(out_path))
    #plt.show()

def _animated_gif_dir(path):
    """Builds an animated GIF from a stack of images in a directory.
    
    Writes the animated file to the parent directory of path.
    
    Params:
        path: Path to the image directory. All images from this directory
            will be imported in Python sorted order.
    """
    # ascending order of all files in the directory
    files = sorted(glob.glob(os.path.join(path, "*")))[::500]
    print(files)
    parent_path = os.path.dirname(path)
    name = os.path.basename(path)
    out_path = os.path.join(parent_path, name + "_animation.gif")
    _build_animated_gif(files, out_path, _import_img)

def _animated_gif_npy(path, series):
    """Builds an animated GIF from a stack of images in a directory.
    
    Writes the animated file to the parent directory of path.
    
    Params:
        path: Path to the image directory. All images from this directory
            will be imported in Python sorted order.
    """
    # ascending order of all files in the directory
    image5d = importer.read_file(path, series)
    planes = image5d[0, ::500]
    parent_path = os.path.dirname(path)
    name = os.path.basename(path)
    i = name.rfind(".")
    if i != -1:
        name = name[:i]
    out_path = os.path.join(parent_path, name + "_animation.gif")
    _build_animated_gif(planes, out_path, _process_plane)

def animated_gif(path, series=0):
    if os.path.isdir(path):
        _animated_gif_dir(path)
    else:
        _animated_gif_npy(path, series)

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
    cli.main(True)
    animated_gif(cli.filename)
