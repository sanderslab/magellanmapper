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

from clrbrain import colormaps
from clrbrain import config
from clrbrain import plot_2d
from clrbrain import plot_support
from clrbrain import importer

def _import_img(i, paths, labels_img, rescale, multichannel):
    # simply import and rescale an image; labels_img is unused, included 
    # only for consistency
    path = paths[i]
    print("importing {}".format(path))
    img = io.imread(path)
    img = transform.rescale(
        img, rescale, mode="reflect", multichannel=multichannel, 
        preserve_range=True, anti_aliasing=True)
    return i, img

def _process_plane(i, img3d, labels_img, rescale, multichannel, rotate=None):
    # process a plane from within an image
    print("processing plane {}".format(i))
    img = transform.rescale(
        img3d[i], rescale, mode="reflect", multichannel=multichannel, 
        preserve_range=True, anti_aliasing=True)
    imgs = [img]
    if labels_img is not None:
        label = transform.rescale(
            labels_img[i], rescale, mode="reflect", multichannel=False, 
            preserve_range=True, anti_aliasing=False, order=0)
        imgs.append(label)
    if rotate:
        # rotate, filling background with edge color
        for i, img in enumerate(imgs):
            #img = img[10:-100, 10:-80] # manually crop out any border
            cval = np.mean(img[0, 0])
            img = ndimage.rotate(img, rotate, cval=cval)
            # additional cropping for oblique bottom edge after rotation
            #img = img[:-50]
            imgs[i] = img
    return i, imgs

def _build_stack(images, out_path, process_fnc, rescale, aspect=None, 
                        origin=None, delay=None, labels_img=None, 
                        cmap_labels=None):
    """Builds an animated GIF from a stack of images.
    
    Args:
        images: Array of images, either as files or Numpy array planes.
        out_path: Output path.
        process_fnc: Function to process each image through multiprocessing, 
            where the function should take an index and image and return the 
            index and processed plane.
        delay: Delay between image display in ms. If None, the delay will 
            defaul to 100ms.
        labels_img: Labels image to overlay; defaults to None.
        cmap_labels: Colormap for labels image; defaults to None.
    """
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
    plotted_imgs = [None] * num_images
    img_size = None
    multichannel = images[0].ndim >= 3
    print("channel: {}".format(config.channel))
    pool = mp.Pool()
    pool_results = []
    for i in range(len(images)):
        # add rotation argument if necessary
        pool_results.append(pool.apply_async(
            process_fnc, 
            args=(i, images, labels_img, rescale, multichannel)))
    cmaps = config.process_settings["channel_colors"]
    imgs_ordered = [None] * len(pool_results)
    for result in pool_results:
        i, imgs = result.get()
        if img_size is None:
            img_size = imgs[0].shape
        # tweak max values to change saturation
        vmax = config.vmax_overview
        #vmax = np.multiply(vmax, (0.9, 1.0))
        
        # multiple artists can be shown at each frame by collecting 
        # each group of artists in a list; overlay_images returns 
        # a nested list containing a list for each image, which in turn 
        # contains a list of artists for each channel
        ax_imgs = plot_support.overlay_images(
            ax, aspect, origin, imgs, [config.channel, 0], 
            [cmaps, cmap_labels], config.alphas, 
            [config.near_min, None], [vmax, None])
        plotted_imgs[i] = np.array(ax_imgs).flatten()
    pool.close()
    pool.join()
    
    fit_frame_to_image(fig, img_size, aspect)
    
    if num_images > 1:
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
    else:
        # single plane figure export
        print("extracting plane as {}".format(out_path))
        fig.savefig(out_path)

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

def animated_gif(image5d, path, offset=None, roi_size=None, slice_vals=None, 
                 rescale=None, delay=None, labels_img=None, single=False):
    """Builds an animated GIF from a stack of images in a directory or an
    .npy file.
    
    Writes the animated file to the parent directory of path.
    
    Args:
        image5d: Images as a 4/5D Numpy array (t,z,y,x[c]).
        path: Path to the image directory or saved Numpy array. If the path is 
            a directory, all images from this directory will be imported in 
            Python sorted order. If the path is a saved Numpy array (eg .npy 
            file), animations will be built by plane, using the plane 
            orientation set in :const:`config.plane`.
        offset: Tuple of offset given in user order (x, y, z); defaults to 
            None. Requires ``roi_size`` to not be None.
        roi_size: Size of the region of interest in user order (x, y, z); 
            defaults to None. Requires ``offset`` to not be None.
        slice_vals: List from which to contstruct a slice object to 
            extract only a portion of the image. Defaults to None, which 
            will give the whole image. If ``offset`` and ``roi_size`` are 
            also given, ``slice_vals`` will only be used for its interval term.
        rescale: Rescaling factor for each image, performed on a plane-by-plane 
            basis; defaults to None, in which case 1.0 will be used.
        delay: Delay between image display in ms.
        labels_img: Labels image as a Numpy z,y,x array; defaults to None.
    """
    parent_path = os.path.dirname(path)
    name = os.path.basename(path)
    
    # build z slice, which will be applied to the transposed image; 
    # reduce image to 1 plane if in single mode
    if offset is not None and roi_size is not None:
        # ROI offset and size take precedence over slice vals except 
        # for use of the interval term
        interval = None
        if slice_vals is not None and len(slice_vals) > 2:
            interval = slice_vals[2]
        size = 1 if single else roi_size[2]
        img_sl = slice(offset[2], offset[2] + size, interval)
    elif slice_vals is not None:
        # build directly from slice vals unless single mode
        if single:
            img_sl = slice(slice_vals[0], slice_vals[0] + 1)
        else:
            img_sl = slice(*slice_vals)
    else:
        # default to take the whole image
        img_sl = slice(None, None)
    if rescale is None:
        rescale = 1.0
    planes = None
    aspect = None
    origin = None
    cmap_labels = None
    fnc = None
    extracted_planes = []
    if os.path.isdir(path):
        # builds animations from all files in a directory
        planes = sorted(glob.glob(os.path.join(path, "*")))[::interval]
        print(planes)
        fnc = _import_img
        extracted_planes.append(planes)
    else:
        # load images from path and extract ROI based on slice parameters
        imgs = [image5d]
        if labels_img is not None:
            imgs.append(labels_img[None])
            show_background = config.alphas[0] == 0
            cmap_labels = colormaps.get_labels_discrete_colormap(
                labels_img, show_background=show_background)
        for img in imgs:
            planes, aspect, origin = plot_2d.extract_plane(
                img, img_sl, plane=config.plane)
            if offset is not None and roi_size is not None:
                # take ROI of x/y vals from transposed image
                planes = planes[
                    :, offset[1]:offset[1]+roi_size[1], 
                    offset[0]:offset[0]+roi_size[0]]
            extracted_planes.append(planes)
            
        out_name = name.replace(".czi", "_").rstrip("_")
        fnc = _process_plane
    
    # name file based on animation vs single plane extraction
    ext = config.savefig
    if single:
        out_name += "_plane_{}{}.{}".format(
            plot_support.get_plane_axis(config.plane), img_sl.start, ext)
    else:
        if ext is None: ext = "gif"
        out_name += "_animation." + ext
    out_path = os.path.join(parent_path, out_name)
    
    # export planes
    planes_labels = None if len(extracted_planes) < 2 else extracted_planes[1]
    _build_stack(extracted_planes[0], out_path, fnc, rescale, aspect=aspect, 
                 origin=origin, delay=delay, 
                 labels_img=planes_labels, cmap_labels=cmap_labels)

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
    cmaps = config.process_settings["channel_colors"]
    # use lower max threshold since overview vmax often skewed by 
    # artifacts over whole image; also use no interpolation for cleanest image
    plot_support.imshow_multichannel(
        ax, img2d, config.channel, cmaps, aspect, 1, vmin=config.near_min, 
        vmax=config.vmax_overview*0.8, 
        origin=origin, interpolation="none")
    fit_frame_to_image(fig, img2d.shape, aspect)
    if not name:
        name = "SavedPlane_z{}".format(plane_n)
    filename = name + "." + config.savefig
    print("extracting plane as {}".format(filename))
    fig.savefig(filename)

if __name__ == "__main__":
    print("Clrbrain stack manipulations")
