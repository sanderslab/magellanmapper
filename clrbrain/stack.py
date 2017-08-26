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

def _import_img(i, path, rescale):
    print("importing {}".format(path))
    img = io.imread(path)
    img = transform.rescale(img, rescale, mode="reflect", multichannel=False)
    return i, img

def _process_plane(i, plane, rescale):
    print("processing plane {}".format(i))
    img = transform.rescale(plane, rescale, mode="reflect", multichannel=False)
    return i, img

def _build_animated_gif(images, out_path, process_fnc, rescale):
    """Builds an animated GIF from a stack of images.
    
    Params:
        images: Array of images, either as files or Numpy array planes.
        out_path: Output path.
        process_fnc: Function to process each image through multiprocessing, 
            where the function should take an index and image and return the 
            index and processed plane.
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
    img_size = None
    i = 0
    pool = mp.Pool()
    pool_results = []
    for image in images:
        pool_results.append(pool.apply_async(
            process_fnc, args=(i, image, rescale)))
        i += 1
    for result in pool_results:
        i, img = result.get()
        if img_size is None:
            img_size = img.shape
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

def animated_gif(path, series=0, interval=1, rescale=0.1):
    """Builds an animated GIF from a stack of images in a directory or a
    .npy file.
    
    Writes the animated file to the parent directory of path.
    
    Params:
        path: Path to the image directory. All images from this directory
            will be imported in Python sorted order.
        series: Stack to build for multiseries files; defaults to 0.
    """
    parent_path = os.path.dirname(path)
    name = os.path.basename(path)
    planes = None
    fnc = None
    if os.path.isdir(path):
        planes = sorted(glob.glob(os.path.join(path, "*")))[::interval]
        print(planes)
        fnc = _import_img
    else:
        image5d = importer.read_file(path, series)
        planes = image5d[0, ::interval]
        i = name.rfind(".")
        if i != -1:
            name = name[:i]
        fnc = _process_plane
    out_path = os.path.join(parent_path, name + "_animation.gif")
    _build_animated_gif(planes, out_path, fnc, rescale)

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
    cli.main(True)
    animated_gif(cli.filename, 0, 10, 0.05)
