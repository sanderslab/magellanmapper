# Stack manipulations
# Author: David Young, 2017, 2018
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
from scipy import ndimage

from clrbrain import config
from clrbrain import plot_2d
from clrbrain import plot_3d
from clrbrain import plot_support
from clrbrain import importer

def _import_img(i, path, rescale, multichannel):
    print("importing {}".format(path))
    img = io.imread(path)
    img = transform.rescale(
        img, rescale, mode="reflect", multichannel=multichannel, 
        preserve_range=True, anti_aliasing=True)
    return i, img

def _process_plane(i, plane, rescale, multichannel, rotate=None):
    print("processing plane {}".format(i))
    img = transform.rescale(
        plane, rescale, mode="reflect", multichannel=multichannel, 
        preserve_range=True, anti_aliasing=True)
    if rotate:
        # rotate, filling background with edge color
        #img = img[10:-100, 10:-80] # manually crop out any border
        cval = np.mean(img[0, 0])
        img = ndimage.rotate(img, rotate, cval=cval)
        # additional cropping for oblique bottom edge after rotation
        #img = img[:-50]
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
    # small border appears around images on occasion with Matplotlib 2 settings
    with plt.style.context("classic"):
        
        # ascending order of all files in the directory
        #images = images[5:-2] # manually remove border planes
        num_images = len(images)
        print("images.shape: {}".format(images.shape))
        if num_images < 1:
            return None
        
        # Matplotlib figure for building the animation
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        plot_support.hide_axes(ax)
        
        # import the images as Matplotlib artists via multiprocessing
        plotted_imgs = [None for i in range(num_images)]
        img_size = None
        multichannel = images[0].ndim >= 3
        print("channel: {}".format(config.channel))
        i = 0
        pool = mp.Pool()
        pool_results = []
        for image in images:
            # add rotation argument if necessary
            pool_results.append(pool.apply_async(
                process_fnc, args=(i, image, rescale, multichannel)))
            i += 1
        colormaps = config.process_settings["channel_colors"]
        for result in pool_results:
            i, img = result.get()
            if img_size is None:
                img_size = img.shape
            # tweak max values to change saturation
            vmax = config.vmax_overview
            #vmax = np.multiply(vmax, (0.9, 1.0))
            plotted_imgs[i] = plot_support.imshow_multichannel(
                ax, img, config.channel, colormaps, aspect, 1, 
                vmin=plot_3d.near_min, vmax=vmax, 
                origin=origin)
        pool.close()
        pool.join()
        
        fit_frame_to_image(fig, img_size, aspect)
        
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

def fit_frame_to_image(fig, shape, aspect):
    # compress layout to fit image only
    fig.tight_layout(pad=0.0) # leaves some space for some reason
    if aspect is None:
        aspect = 1
    img_size_inches = np.divide(shape, fig.dpi) # convert to inches
    print("image shape: {}, img_size_inches: {}"
          .format(shape, img_size_inches))
    if aspect > 1:
        fig.set_size_inches(img_size_inches[1], img_size_inches[0] * aspect)
    else:
        # multiply both sides by 1 / aspect => number > 1 to enlarge
        fig.set_size_inches(img_size_inches[1] / aspect, img_size_inches[0])
    print("fig size: {}".format(fig.get_size_inches()))

def animated_gif(path, series=0, interval=None, rescale=None, delay=None):
    """Builds an animated GIF from a stack of images in a directory or an
    .npy file.
    
    Writes the animated file to the parent directory of path.
    
    Args:
        path: Path to the image directory or saved Numpy array. If the path is 
            a directory, all images from this directory will be imported in 
            Python sorted order. If the path is a saved Numpy array (eg .npy 
            file), animations will be built by plane, using the plane 
            orientation set in :const:`config.plane`.
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
            image5d, slice(None, None, interval), plane=config.plane)
        out_name = name.replace(".czi", "_").rstrip("_")
        fnc = _process_plane
    ext = plot_2d.savefig
    if ext is None:
        ext = "gif"
    out_path = os.path.join(parent_path, out_name + "_animation." + ext)
    _build_animated_gif(planes, out_path, fnc, rescale, aspect=aspect, 
                        origin=origin, delay=delay)

def save_plane(image5d, offset, roi_size=None, name=None):
    """Extracts a single 2D plane and saves to file.
    
    Args:
        image5d: The full image stack.
        offset: Tuple of x,y,z coordinates of the ROI. The plane will be 
            extracted from the z coordinate. If ``roi_size`` is not None, 
            ``offset`` x,y values will be used for the ROI offset within 
            the plane.
        roi_size: List of x,y,z dimensions of the ROI; default to None, in 
            which case the entire plane will be extracted.
        name: Name of the resulting file, without the extension; default to 
            None, in which case a standard name will be given.
    """
    plane_n = offset[2]
    img2d, aspect, origin = plot_2d.extract_plane(
        image5d, plane_n, plane=config.plane)
    if roi_size is not None:
        img2d = img2d[
            offset[1]:offset[1]+roi_size[1], offset[0]:offset[0]+roi_size[0]]
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    plot_support.hide_axes(ax)
    colormaps = config.process_settings["channel_colors"]
    # use lower max threshold since overview vmax often skewed by 
    # artifacts over whole image; also use no interpolation for cleanest image
    plot_support.imshow_multichannel(
        ax, img2d, config.channel, colormaps, aspect, 1, vmin=plot_3d.near_min, 
        vmax=config.vmax_overview*0.8, 
        origin=origin, interpolation="none")
    fit_frame_to_image(fig, img2d.shape, aspect)
    if not name:
        name = "SavedPlane_z{}".format(plane_n)
    filename = name + "." + plot_2d.savefig
    print("extracting plane as {}".format(filename))
    fig.savefig(filename)

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
