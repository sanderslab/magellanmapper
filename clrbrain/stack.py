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

def _build_animated_gif(images, out_path, process_fnc, rescale, aspect=None, 
                        origin=None, delay=None):
    """Builds an animated GIF from a stack of images.
    
    Args:
        images: Array of images, either as files or Numpy array planes.
        out_path: Output path.
        process_fnc: Function to process each image through multiprocessing, 
            where the function should take an index and image and return the 
            index and processed plane.
        delay: Delay between image display in ms. If None, the delay will 
            defaul to 100ms.
    """
    # ascending order of all files in the directory
    num_images = len(images)
    print("images.shape: {}".format(images.shape))
    if num_images < 1:
        return None
    
    # Matplotlib figure for building the animation
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # rescaled images will be converted from integer to float, so 
    # vmax will need to be rescaled to 0-1 range
    vmax = plot_2d.vmax_overview
    max_range = 0
    if rescale and np.issubdtype(images.dtype, np.integer):
        max_range = np.iinfo(images.dtype).max
    if max_range != 0:
        vmax = vmax / max_range
    
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
            img, cmap=plot_2d.CMAP_GRBK, vmin=0, vmax=vmax, aspect=aspect, 
            origin=origin)]
    pool.close()
    pool.join()
    
    # need to compress layout to fit image only
    #print(len(plotted_imgs))
    plt.tight_layout(pad=0.0) # leaves some space for some reason
    fig_size = fig.get_size_inches()
    if aspect is None:
        aspect = 1
    #print("img_size: {}".format(img_size))
    img_size_dpi = np.divide(img_size, fig.dpi) # convert to inches
    if aspect > 1:#img_size[0] < img_size[1]:
        fig.set_size_inches(img_size_dpi[1], img_size_dpi[0] * aspect)
    else:
        # multiply both sides by 1 / aspect => number > 1 to enlarge
        fig.set_size_inches(img_size_dpi[1] / aspect, img_size_dpi[0])
    
    # export to animated GIF
    if delay is None:
        delay = 100
    anim = animation.ArtistAnimation(
        fig, plotted_imgs, interval=delay, repeat_delay=0, blit=False)
    try:
        anim.save(out_path, writer="imagemagick")
    except ValueError as e:
        print(e)
        print("No animation writer available for Matplotlib")
    print("saved animation file to {}".format(out_path))
    #plt.show()

def animated_gif(path, series=0, interval=None, rescale=None, delay=None):
    """Builds an animated GIF from a stack of images in a directory or an
    .npy file.
    
    Writes the animated file to the parent directory of path.
    
    Args:
        path: Path to the image directory or saved Numpy array. If the path is 
            a directory, all images from this directory will be imported in 
            Python sorted order. If the path is a saved Numpy array (eg .npy 
            file), animations will be built by plane, using the plane 
            orientation set in :const:`plot_2d.plane`.
        series: Stack to build for multiseries files; defaults to 0.
        interval: Every nth image will be incorporated into the animation; 
            defaults to None, in which case 1 will be used.
        rescale: Rescaling factor for each image, performed on a plane-by-plane 
            basis; defaults to None, in which case 1.0 will be used.
        delay: Delay between image display in ms.
    """
    parent_path = os.path.dirname(path)
    name = os.path.basename(path)
    if interval is None:
        interval = 1
    if rescale is None:
        rescale = 1.0
    planes = None
    aspect = None
    origin = None
    fnc = None
    if os.path.isdir(path):
        # builds animations from all files in a directory
        planes = sorted(glob.glob(os.path.join(path, "*")))[::interval]
        print(planes)
        fnc = _import_img
    else:
        image5d = importer.read_file(path, series)
        planes, aspect, origin = plot_2d.extract_plane(
            image5d, slice(None, None, interval), plane=plot_2d.plane, 
            channel=cli.channel)
        out_name = name.replace(".czi", "_").rstrip("_")
        fnc = _process_plane
    ext = plot_2d.savefig
    if ext is None:
        ext = "gif"
    out_path = os.path.join(parent_path, out_name + "_animation." + ext)
    _build_animated_gif(planes, out_path, fnc, rescale, aspect=aspect, 
                        origin=origin, delay=delay)

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
    cli.main(True)
