# Stack manipulations
# Author: David Young, 2017, 2019
"""Import and export image stacks in various formats.
"""

import os
import glob
import multiprocessing as mp

import numpy as np
from skimage import transform
from skimage import io
from matplotlib import animation
from scipy import ndimage

from magmap import colormaps
from magmap import config
from magmap import lib_clrbrain
from magmap import np_io
from magmap import plot_support


class StackPlaneIO(object):
    """Worker class to export planes from a stack with support for 
    multiprocessing.
    
    Attributes:
        imgs: A list of images, with exact specification determined by 
            the calling function.
    """
    imgs = None
    
    @classmethod
    def set_data(cls, imgs):
        """Set data to be accessed by worker functions."""
        cls.imgs = imgs
    
    @classmethod
    def import_img(cls, i, rescale, multichannel):
        """Import and rescale an image.
        
        Assumes that :attr:``imgs`` is a list of paths to 2D images.
        
        Args:
            i: Index within :attr:``imgs`` to plot.
            rescale: Rescaling multiplier.
            multichannel: True if the images are multichannel.
        
        Returns:
            A tuple of ``i`` and a list of the processed images. The 
            processed image list has the same length as :attr:``imgs``, 
            or the number of image paths.
        """
        path = cls.imgs[i]
        print("importing {}".format(path))
        img = io.imread(path)
        img = transform.rescale(
            img, rescale, mode="reflect", multichannel=multichannel, 
            preserve_range=True, anti_aliasing=True)
        return i, img
    
    @classmethod
    def process_plane(cls, i, target_size, rotate=None):
        """Process corresponding planes from related images.
        
        Assumes that :attr:``imgs`` is a list of nested 2D image lists, 
        where the first nested list is assumed to be a sequence of 
        histology image planes, while subsequent images are 
        labels-based images.
        
        Args:
            i: Index within nested lists of :attr:``imgs`` to plot.
            target_size: Resize to this shape.
            rotate: Degrees by which to rotate; defaults to None.
        
        Returns:
            A tuple of ``i`` and a list of the processed images. The 
            processed image list has the same length as :attr:``imgs``, 
            or the number of nested lists.
        """
        print("processing plane {}".format(i))
        imgs_proc = []
        for j, img_stack in enumerate(cls.imgs):
            if j == 0:
                # atlas image
                img = transform.resize(
                    img_stack[i], target_size, mode="reflect",
                    preserve_range=True, anti_aliasing=True)
            else:
                # labels-based image, using nearest-neighbor interpolation
                img = transform.resize(
                    img_stack[i], target_size, mode="reflect",
                    preserve_range=True, anti_aliasing=False, order=0)
            imgs_proc.append(img)
        if rotate:
            # rotate, filling background with edge color
            for j, img in enumerate(imgs_proc):
                #img = img[10:-100, 10:-80] # manually crop out any border
                cval = np.mean(img[0, 0])
                img = ndimage.rotate(img, rotate, cval=cval)
                # additional cropping for oblique bottom edge after rotation
                #img = img[:-50]
                imgs_proc[j] = img
        return i, imgs_proc


def _build_stack(ax, images, process_fnc, rescale=1, aspect=None, 
                 origin=None, cmaps_labels=None, scale_bar=True):
    """Builds an animated GIF from a stack of images.
    
    Uses multiprocessing to load or resize each image.
    
    Args:
        images: Sequence of images. For import, each "image" is a path to 
            and image file. For export, each "image" is a sequence of 
            planes, with the first sequence assumed to an atlas, 
            followed by labels-based images, each consisting of 
            corresponding planes.
        process_fnc: Function to process each image through multiprocessing, 
            where the function should take an index and image and return the 
            index and processed plane.
        rescale (float): Rescale factor; defaults to 1.
        cmaps_labels: Sequence of colormaps for labels-based images; 
            defaults to None. Length should be equal to that of 
            ``images`` - 1.
        scale_bar: True to include scale bar; defaults to True.
    
    Returns:
        :List[List[:obj:`matplotlib.image.AxesImage`]]: Nested list of 
        axes image objects. The first list level contains planes, and
        the second level are channels within each plane.
    """
    # number of image types (eg atlas, labels) and corresponding planes
    num_image_types = len(images)
    if num_image_types < 1: return None
    num_images = len(images[0])
    if num_images < 1: return None
    
    # Matplotlib figure for building the animation
    plot_support.hide_axes(ax)
    
    # import the images as Matplotlib artists via multiprocessing
    plotted_imgs = [None] * num_images
    img_shape = images[0][0].shape
    target_size = np.multiply(img_shape, rescale).astype(int)
    multichannel = images[0][0].ndim >= 3
    if multichannel:
        print("building stack for channel: {}".format(config.channel))
        target_size = target_size[:-1]
    StackPlaneIO.set_data(images)
    pool = mp.Pool()
    pool_results = []
    for i in range(num_images):
        # add rotation argument if necessary
        pool_results.append(
            pool.apply_async(process_fnc, args=(i, target_size)))
    
    # setup imshow parameters
    colorbar = config.process_settings["colorbar"]
    cmaps_all = [config.cmaps, *cmaps_labels]
    alphas = lib_clrbrain.pad_seq(config.alphas, num_image_types, 0.9)

    img_size = None
    for result in pool_results:
        i, imgs = result.get()
        if img_size is None: img_size = imgs[0].shape
        
        # multiple artists can be shown at each frame by collecting 
        # each group of artists in a list; overlay_images returns 
        # a nested list containing a list for each image, which in turn 
        # contains a list of artists for each channel
        ax_imgs = plot_support.overlay_images(
            ax, aspect, origin, imgs, None, cmaps_all, alphas)
        if colorbar and len(ax_imgs) > 0 and len(ax_imgs[0]) > 0:
            # add colorbar with scientific notation if outside limits
            cbar = ax.figure.colorbar(ax_imgs[0][0], ax=ax, shrink=0.7)
            plot_support.set_scinot(cbar.ax, lbls=None, units=None)
        plotted_imgs[i] = np.array(ax_imgs).flatten()
    pool.close()
    pool.join()
    
    if scale_bar:
        plot_support.add_scale_bar(ax, 1 / rescale, config.plane)
    
    return plotted_imgs


def animate_imgs(base_path, plotted_imgs, delay, ext=None):
    """Export to an animated image.
    
    Defaults to an animated GIF unless ``ext`` specifies otherwise.
    
    Args:
        base_path (str): String from which an output path will be constructed.
        plotted_imgs: 
        delay: Delay between image display in ms. If None, the delay will 
            defaul to 100ms.

    Returns:

    """
    if ext is None: ext = "gif"
    out_path = lib_clrbrain.combine_paths(base_path, "animated", ext=ext)
    lib_clrbrain.backup_file(out_path)
    if delay is None:
        delay = 100
    if plotted_imgs and len(plotted_imgs[0]) > 0:
        fig = plotted_imgs[0][0].figure
    else:
        lib_clrbrain.warn("No images available to animate")
        return
    anim = animation.ArtistAnimation(
        fig, plotted_imgs, interval=delay, repeat_delay=0, blit=False)
    try:
        anim.save(out_path, writer="imagemagick")
        print("saved animation file to {}".format(out_path))
    except ValueError as e:
        print(e)
        lib_clrbrain.warn("No animation writer available for Matplotlib")


def _setup_labels_cmaps(imgs, cmaps_labels=None):
    """Setup labels colormaps for registered images.
    
    Args:
        imgs (List[:obj:`np.ndarray`]): Sequence of images where the first
            is assumed to be non-labels, the second is labels, and the
            third are label borders.
        cmaps_labels (List[List[:obj:`colormaps.DiscreteColormap`]]): 
            List of discrete colormaps corresponding to ``imgs[:1]``.

    """
    if cmaps_labels is None:
        cmaps_labels = []
    num_imgs = len(imgs)
    if num_imgs > 1:
        # 2nd image is main labels image, but use original set of 
        # labels if available
        cmaps_labels.append(
            colormaps.get_labels_discrete_colormap(
                imgs[1], 0, dup_for_neg=True, use_orig_labels=True))
    if num_imgs > 2:
        # subsequent image's colormap is based on first labels 
        # if possible
        cmaps_labels.append(
            colormaps.get_borders_colormap(
                imgs[2], imgs[1], cmaps_labels[0]))
    return cmaps_labels


def stack_to_ax_imgs(ax, image5d, path=None, offset=None, roi_size=None,
                     slice_vals=None, rescale=None, labels_imgs=None,
                     multiplane=False, fit=False):
    """Export a stack of images in a directory or a single volumetric image
    and associated labels images to :obj:`matplotlib.image.AxesImage`
    objects for export.
    
    Args:
        ax (:obj:`plt.Axes`): Matplotlib axes on which to plot images.
        image5d: Images as a 4/5D Numpy array (t,z,y,x[c]). Can be None if 
            ``path`` is set.
        path: Path to an image directory from which all files will be imported 
            in Python sorted order, taking precedence over ``imaged5d``; 
            defaults to None.
        offset: Tuple of offset given in user order (x, y, z); defaults to 
            None. Requires ``roi_size`` to not be None.
        roi_size: Size of the region of interest in user order (x, y, z); 
            defaults to None. Requires ``offset`` to not be None.
        slice_vals: List from which to construct a slice object to 
            extract only a portion of the image. Defaults to None, which 
            will give the whole image. If ``offset`` and ``roi_size`` are 
            also given, ``slice_vals`` will only be used for its interval term.
        rescale: Rescaling factor for each image, performed on a plane-by-plane 
            basis; defaults to None, in which case 1.0 will be used.
        labels_imgs: Sequence of labels-based images as a Numpy z,y,x arrays, 
            typically including labels and borders images; defaults to None.
        multiplane: True to extract the images as an animated GIF or movie 
            file; False to extract a single plane only. Defaults to False.
        fit (bool): True to fit the figure frame to the resulting image.
    
    Returns:
        List[:obj:`matplotlib.image.AxesImage`]: List of image objects.
    
    """
    print("Starting image stack export")
    
    # build "z" slice, which will be applied to the transposed image; 
    # reduce image to 1 plane if in single mode
    interval = 1
    if offset is not None and roi_size is not None:
        # transpose coordinates to given plane
        _, arrs_1d = plot_support.transpose_images(
            config.plane, arrs_1d=[offset[::-1], roi_size[::-1]])
        offset = arrs_1d[0][::-1]
        roi_size = arrs_1d[1][::-1]
        
        # ROI offset and size take precedence over slice vals except 
        # for use of the interval term
        interval = None
        if slice_vals is not None and len(slice_vals) > 2:
            interval = slice_vals[2]
        size = roi_size[2] if multiplane else 1
        img_sl = slice(offset[2], offset[2] + size, interval)
        if interval is not None and interval < 0:
            # reverse start/stop order to iterate backward
            img_sl = slice(img_sl.stop, img_sl.start, interval)
        print("using ROI offset {}, size {}, {}"
              .format(offset, size, img_sl))
    elif slice_vals is not None:
        # build directly from slice vals unless not an animation
        if multiplane:
            img_sl = slice(*slice_vals)
        else:
            # single plane only for non-animation
            img_sl = slice(slice_vals[0], slice_vals[0] + 1)
    else:
        # default to take the whole image stack
        img_sl = slice(None, None)
    if rescale is None:
        rescale = 1.0
    aspect = None
    origin = None
    cmaps_labels = []
    extracted_planes = []
    if path and os.path.isdir(path):
        # builds animations from all files in a directory
        planes = sorted(glob.glob(os.path.join(path, "*")))[::interval]
        print(planes)
        fnc = StackPlaneIO.import_img
        extracted_planes.append(planes)
    else:
        # load images from path and extract ROI based on slice parameters, 
        # assuming 1st image is atlas, 2nd and beyond are labels-based
        imgs = [image5d]
        if labels_imgs is not None:
            for img in labels_imgs:
                if img is not None: imgs.append(img[None])
            _setup_labels_cmaps(imgs, cmaps_labels)
        main_shape = None  # z,y,x shape of 1st image
        for img in imgs:
            sl = img_sl
            img_shape = img.shape[1:4]
            if main_shape:
                if main_shape != img_shape:
                    # scale slice bounds to the first image's shape
                    scaling = np.divide(img_shape, main_shape)
                    axis = plot_support.get_plane_axis(config.plane, True)
                    sl = lib_clrbrain.scale_slice(
                        sl, scaling[axis], img_shape[axis])
            else:
                main_shape = img_shape
            planes, aspect, origin = plot_support.extract_planes(
                img, sl, plane=config.plane)
            if offset is not None and roi_size is not None:
                # get ROI using transposed coordinates on transposed planes; 
                # returns list
                planes = planes[
                    :, offset[1]:offset[1]+roi_size[1], 
                    offset[0]:offset[0]+roi_size[0]]
            extracted_planes.append(planes)
        fnc = StackPlaneIO.process_plane
    
    # export planes
    plotted_imgs = _build_stack(
        ax, extracted_planes, fnc, rescale, aspect=aspect, 
        origin=origin, cmaps_labels=cmaps_labels, scale_bar=config.scale_bar)
    
    if fit and plotted_imgs:
        # fit frame to first plane's first available image
        ax_img = None
        for ax_img in plotted_imgs[0]:
            # images may be None if alpha set to 0
            if ax_img is not None: break
        if ax_img is not None:
            plot_support.fit_frame_to_image(
                ax_img.figure, ax_img.get_array().shape, aspect)
    
    return plotted_imgs


def stack_to_img(paths, series, offset, roi_size, animated=False, suffix=None):
    """Build an image file from a stack of images in a directory or an 
    array, exporting as an animated GIF or movie for multiple planes or 
    extracting a single plane to a standard image file format.
    
    Writes the file to the parent directory of path.
    
    Args:
        paths (List[str]): Image paths, which can each be either an image 
            directory or a base path to a single image, including 
            volumetric images.
        series (int): Image series number.
        offset (List[int]): Tuple of offset given in user order (x, y, z); 
            defaults to None. Requires ``roi_size`` to not be None.
        roi_size (List[int]): Size of the region of interest in user order 
            (x, y, z); defaults to None. Requires ``offset`` to not be None.
        animated (bool): True to export as an animated image; defaults to False.
        suffix (str): String to append to output path before extension; 
            defaults to None to ignore.

    """
    size = config.plot_labels[config.PlotLabels.LAYOUT]
    ncols, nrows = size if size else (1, 1)
    fig, gs = plot_support.setup_fig(
        nrows, ncols, config.plot_labels[config.PlotLabels.SIZE])
    plotted_imgs = None
    num_paths = len(paths)
    for i in range(nrows):
        for j in range(ncols):
            n = i * ncols + j
            if n >= num_paths: break
            ax = fig.add_subplot(gs[i, j])
            path_sub = paths[n]
            # TODO: test directory of images
            # TODO: avoid reloading first image
            np_io.setup_images(path_sub, series)
            plotted_imgs = stack_to_ax_imgs(
                ax, config.image5d, path_sub, offset=offset, 
                roi_size=roi_size, slice_vals=config.slice_vals, 
                rescale=config.rescale, 
                labels_imgs=(config.labels_img, config.borders_img), 
                multiplane=animated, 
                fit=(size is None or ncols * nrows == 1))
    if animated:
        animate_imgs(
            paths[0], plotted_imgs, config.delay, config.savefig)
    else:
        planei = offset[-1] if offset else config.slice_vals[0]
        path_base = paths[0]
        if num_paths > 1:
            # output filename as a collage of images
            if not os.path.isdir(path_base):
                path_base = os.path.dirname(path_base)
            path_base = os.path.join(path_base, "collage")
        mod = "_plane_{}{}".format(
            plot_support.get_plane_axis(config.plane), planei)
        if suffix: path_base = lib_clrbrain.insert_before_ext(path_base, suffix)
        plot_support.save_fig(path_base, config.savefig, mod)


def reg_planes_to_img(imgs, path=None, ax=None):
    """Export registered image single planes to a single figure.
    
    Simplified export tool taking a single plane from each registered image
    type, overlaying in a single figure, and exporting to file.
    
    Args:
        imgs (List[:obj:`np.ndarray`]): Sequence of image planes to display.
            The first image is assumed to be greyscale, the second is labels,
            and any subsequent images are borders.
        path (str): Output base path, which will be combined with
            :attr:`config.savefig`; defaults to None to not save.
        ax (:obj:`matplotlib.image.Axes`): Axes on which to plot; defaults
            to False, in which case a new figure and axes will be generated.

    """
    if ax is None:
        # set up new figure with single subplot
        fig, gs = plot_support.setup_fig(
            1, 1, config.plot_labels[config.PlotLabels.SIZE])
        ax = fig.add_subplot(gs[0, 0])
    imgs = [img[None] for img in imgs]
    cmaps_labels = _setup_labels_cmaps(imgs)
    plotted_imgs = _build_stack(
        ax, imgs, StackPlaneIO.process_plane,
        cmaps_labels=cmaps_labels, scale_bar=False)
    ax_img = plotted_imgs[0][0]
    aspect, origin = plot_support.get_aspect_ratio(config.plane)
    plot_support.fit_frame_to_image(
        ax_img.figure, ax_img.get_array().shape, aspect)
    if path:
        plot_support.save_fig(path, config.savefig)


if __name__ == "__main__":
    print("MagellanMapper stack manipulations")
