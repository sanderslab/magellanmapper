#!/bin/bash
# Plot Support for Clrbrain
# Author: David Young, 2018, 2019
"""Shared plotting functions with the Clrbrain package.
"""

import os
import warnings

import numpy as np
import matplotlib.backend_bases as backend_bases
from matplotlib import pyplot as plt

from clrbrain import colormaps
from clrbrain import config
from clrbrain import detector
from clrbrain import lib_clrbrain
from clrbrain import plot_3d

try:
    from matplotlib_scalebar import scalebar
except ImportError as e:
    scalebar = None
    warnings.warn(config.WARN_IMPORT_SITK, ImportWarning)


def imshow_multichannel(ax, img2d, channel, cmaps, aspect, alpha, 
                        vmin=None, vmax=None, origin=None, interpolation=None, 
                        norms=None):
    """Show multichannel 2D image with channels overlaid over one another.

    Applies :attr:``config.transpose`` to flip/rotate images. Also checks
    first :attr:``config.flip`` array element to rotate the image
    by 180 degrees if ``rotate`` is not available in :attr:``config.transpose``.

    Args:
        ax: Axes plot.
        img2d: 2D image either as 2D (y, x) or 3D (y, x, channel) array.
        channel: Channel to display; if None, all channels will be shown.
        cmaps: List of colormaps corresponding to each channel. Colormaps 
            can be the names of specific maps in :mod:``config``.
        aspect: Aspect ratio.
        alpha (List[float]): Sequence of transparency levels. If any 
            value is 0, the corresponding image will not be output. 
        vim: Sequence of vmin levels for each channel; defaults to None.
        vmax: Sequence of vmax levels for each channel; defaults to None.
        origin: Image origin; defaults to None.
        interpolation: Type of interpolation; defaults to None.
        norms: List of normalizations, which should correspond to ``cmaps``.
    
    Returns:
        List of ``AxesImage`` objects.
    """
    # assume that 3D array has a channel dimension
    multichannel, channels = plot_3d.setup_channels(img2d, channel, 2)
    img = []
    i = 0
    vmin_plane = None
    vmax_plane = None
    num_chls = len(channels)

    # transform image based on config parameters
    rotate = config.transform[config.Transforms.ROTATE]
    if rotate is not None or config.flip is not None and config.flip[0]:
        if rotate is None:
            # rotate image by 180deg if first flip setting is True;
            # TODO: consider replacing flip with transpose attribute
            rotate = 2
        last_axis = img2d.ndim - 1
        if multichannel:
            last_axis -= 1
        img2d = np.rot90(img2d, rotate, (last_axis - 1, last_axis))
    if config.transform[config.Transforms.FLIP_HORIZ]:
        img2d = img2d[..., ::-1, :] if multichannel else img2d[..., ::-1]
    if config.transform[config.Transforms.FLIP_VERT]:
        img2d = img2d[::-1]

    if num_chls > 1:
        # increasing numbers of channels are each more transluscent
        alpha /= np.sqrt(num_chls + 1)
    for chl in channels:
        img2d_show = img2d[..., chl] if multichannel else img2d
        cmap = cmaps[chl]
        norm = None if norms is None else norms[chl]
        cmap = colormaps.get_cmap(cmap)
        if cmap is not None:
            # show masked values such as NaNs as black to distinguish from 0
            cmap.set_bad(color="black")
        if vmin is not None:
            vmin_plane = vmin[chl]
        if vmax is not None:
            vmax_plane = vmax[chl]
        #print("vmin: {}, vmax: {}".format(vmin_plane, vmax_plane))
        img_chl = None
        if alpha > 0:
            # skip display if alpha is 0 to avoid outputting a hidden image 
            # that may show up in other renderers (eg PDf viewers)
            img_chl = ax.imshow(
                img2d_show, cmap=cmap, norm=norm, aspect=aspect, alpha=alpha, 
                vmin=vmin_plane, vmax=vmax_plane, origin=origin, 
                interpolation=interpolation)
        img.append(img_chl)
        i += 1
    return img

def overlay_images(ax, aspect, origin, imgs2d, channels, cmaps, alphas, 
                   vmins=None, vmaxs=None):
    """Show multiple, overlaid images.
    
    Wrapper function calling :func:``imshow_multichannel`` for multiple 
    images. The first image is treated as a sample image with potential 
    for multiple channels. Subsequent images are typically label images, 
    which may or may not have multple channels.
    
    Args:
        ax: Axes.
        aspect: Aspect ratio.
        origin: Image origin.
        imgs2d: List of 2D images to display.
        channels: A list of channels designators for each image, or None 
            to use :attr:``config.channel`` for the first image and 0 
            for all subsequent images.
        cmaps: Either a single colormap for all images or a list of 
            colormaps corresponding to each image. Colormaps of type 
            :class:``colormaps.DiscreteColormap`` will have their 
            normalization object applied as well.
        alphas: Either a single alpha for all images or a list of 
            alphas corresponding to each image.
        vmins: A list of vmins for each image; defaults to None to use 
            :attr:``config.vmins`` for the first image and None for all others.
        vmaxs: A list of vmaxs for each image; defaults to None to use 
            :attr:``config.vmax_overview`` for the first image and None 
            for all others.
    
    Returns:
        Nested list containing a list of ``AxesImage`` objects 
        corresponding to display of each ``imgs2d`` image.
    """
    ax_imgs = []
    num_imgs2d = len(imgs2d)
    if num_imgs2d < 1: return None
    
    def fill(fill_with, chls):
        # make a sequence with vals corresponding to each 2D image, where 
        # the first val is another seq whose values correspond to each of 
        # the channels in that image, starting with fill_with
        filled = [None] * num_imgs2d
        if fill_with is not None:
            filled[0] = lib_clrbrain.pad_seq(list(fill_with), len(chls))
        return filled
    
    if channels is None:
        # channels are designators rather than lists of specific channels
        channels = [0] * num_imgs2d
        channels[0] = config.channel
    if vmins is None or vmaxs is None:
        # fill vmin/vmax with None for each 2D image and config vals for 
        # each channel for the first image
        _, channels_main = plot_3d.setup_channels(imgs2d[0], None, 2)
        if vmins is None:
            fill_with = (config.near_min if config.vmins is None 
                         else config.vmins)
            vmins = fill(config.vmins, channels_main)
        if vmaxs is None:
            vmaxs = fill(config.vmax_overview, channels_main)
    
    for i in range(num_imgs2d):
        # generate a multichannel display image for each 2D image
        cmap = cmaps[i]
        norm = None
        if isinstance(cmap, colormaps.DiscreteColormap):
            # get normalization factor for discrete colormaps
            norm = [cmap.norm]
            cmap = [cmap]
        ax_img = imshow_multichannel(
            ax, imgs2d[i], channels[i], cmap, aspect, alphas[i], origin=origin, 
            interpolation="none", norms=norm, vmin=vmins[i], vmax=vmaxs[i])
        ax_imgs.append(ax_img)
    return ax_imgs

def extract_planes(image5d, plane_n, plane=None, max_intens_proj=False):
    """Extract a 2D plane or stack of planes.
    
    Args:
        image5d: The full image stack.
        plane_n: Slice of planes to extract, which can be a single index 
            or multiple indices such as would be used for an animation.
        plane: Type of plane to extract, which should be one of 
            :attribute:`config.PLANES`.
        max_intens_projection: True to show a max intensity projection, which 
            assumes that plane_n is an array of multiple, typically 
            contiguous planes along which the max intensity pixel will 
            be taken. Defaults to False.
    
    Returns:
        Tuple of an array of the image, which is 2D if ``plane_n`` is a 
        scalar or ``max_intens_projection`` is True, or 3D otherwise; 
        the aspect ratio; and the origin value.
    """
    origin = None
    aspect = None # aspect ratio
    img3d = None
    if image5d.ndim >= 4:
        img3d = image5d[0]
    else:
        img3d = image5d[:]
    arrs_3d, _ = transpose_images(plane, [img3d])
    aspect, origin = get_aspect_ratio(plane)
    img3d = arrs_3d[0]
    img2d = img3d[plane_n]
    if max_intens_proj:
        # max intensity projection assumes axis 0 is the "z" axis
        img2d = np.amax(img2d, axis=0)
    #print("aspect: {}, origin: {}".format(aspect, origin))
    return img2d, aspect, origin

def add_scale_bar(ax, downsample=None, plane=None):
    """Adds a scale bar to the plot.
    
    Uses the x resolution value and assumes that it is in microns per pixel. 
    The bar's color is taken from the setting in 
    :attr:``config.process_settings``.
    
    Args:
        ax: The plot that will show the bar.
        downsample: Downsampling factor by which the resolution will be 
            multiplied; defaults to None.
        plane: Plane of the image, used to transpose the resolutions to 
            find the corresponding x resolution for the given orientation. 
            Defaults to None.
    """
    # ensure that ScaleBar package exists
    if not scalebar: return
    
    resolutions = detector.resolutions[0]
    if plane:
        # transpose resolutions to the given plane
        _, arrs_1d = transpose_images(plane, arrs_1d=[resolutions])
        resolutions = arrs_1d[0]
    res = resolutions[2] # assume scale bar is along x-axis
    if downsample:
        res *= downsample
    scale_bar = scalebar.ScaleBar(
        res, u'\u00b5m', scalebar.SI_LENGTH, box_alpha=0, 
        color=config.process_settings["scale_bar_color"], location=3)
    ax.add_artist(scale_bar)

def max_plane(img3d, plane):
    """Get the max plane for the given 3D image.
    
    Args:
        img3d: Image array in (z, y, x) order.
        plane: Plane as a value from :attr:``config.PLANE``.
    
    Returns:
        Number of elements along ``plane``'s axis.
    """
    shape = img3d.shape
    if plane == config.PLANE[1]:
        return shape[1]
    elif plane == config.PLANE[2]:
        return shape[2]
    else:
        return shape[0]

def transpose_images(plane, arrs_3d=None, arrs_1d=None, rev=False):
    """Transpose images and associated coorinates to the given plane.
    
    Args:
        plane: Target plane, which should be one of :const:``config.PLANE``. 
            If ``rev`` is True, the array will be assumed to have been 
            transposed from ``plane``.
        arrs_3d: Sequence of 3D arrays to transpose; defaults to None.
        arrs_1d: Sequence of 1D arrays to transpose, typically coordinates 
            associated with the 3D arrays; defaults to None.
        rev: True to transpose in reverse, from ``plane`` to "xy".
    
    Returns:
        Tuple of a list of transposed 3D arrays, or None if no 3D arrays 
        are given; and a list of transposed 1D arrays, or None if no 1D 
        arrays are given.
    """
    
    def swap(indices):
        arrs_3d_swapped = None
        arrs_1d_swapped = None
        if arrs_3d is not None:
            arrs_3d_swapped = [
                None if arr is None else np.swapaxes(arr, *indices) 
                for arr in arrs_3d]
        if arrs_1d is not None:
            arrs_1d_swapped = [
                None if arr is None else 
                    lib_clrbrain.swap_elements(np.copy(arr), *indices) 
                for arr in arrs_1d]
        return arrs_3d_swapped, arrs_1d_swapped
    
    if plane == config.PLANE[1]:
        # xz plane: make y the "z" axis
        if rev:
            arrs_3d, arrs_1d = swap((0, 1))
        else:
            arrs_3d, arrs_1d = swap((0, 1))
    elif plane == config.PLANE[2]:
        # yz plane: make x the "z" axis for stack of 2D plots, eg animations
        if rev:
            arrs_3d, arrs_1d = swap((1, 2))
            arrs_3d, arrs_1d = swap((0, 2))
        else:
            arrs_3d, arrs_1d = swap((0, 2))
            arrs_3d, arrs_1d = swap((1, 2))
    # no changes for xy, the default plane
    return arrs_3d, arrs_1d

def get_aspect_ratio(plane):
    """Get the aspect ratio and origin for the given plane
    
    Args:
        plane: Planar orientation, which should be one of 
            :const:``config.PLANE``.
    
    Returns:
        Tuple of the aspect ratio as a float, or None if 
        :attr:``detector.resolutions`` has not been set; and origin as a 
        string, or None for default origin.
    """
    origin = None
    aspect = None
    if plane == config.PLANE[1]:
        # xz plane
        origin = "lower"
        if detector.resolutions is not None:
            aspect = detector.resolutions[0, 0] / detector.resolutions[0, 2]
    elif plane == config.PLANE[2]:
        # yz plane
        origin = "lower"
        if detector.resolutions is not None:
            aspect = detector.resolutions[0, 0] / detector.resolutions[0, 1]
    else:
        # defaults to "xy"
        if detector.resolutions is not None:
            aspect = detector.resolutions[0, 1] / detector.resolutions[0, 2]
    return aspect, origin

def scroll_plane(event, z_overview, max_size, jump=None, max_scroll=None):
    """Scroll through overview images along their orthogonal axis.
    
    Args:
        event: Mouse or key event. For mouse events, scroll step sizes 
            will be used for movements. For key events, up/down arrows 
            will be used.
        max_size: Maximum number of planes.
        jump: Function to jump to a given plane; defaults to None.
        max_scroll: Max number of planes to scroll by mouse. Ignored during 
            jumps.
    """
    step = 0
    if isinstance(event, backend_bases.MouseEvent):
        # scroll movements are scaled from 0 for each event
        steps = event.step
        if max_scroll is not None and abs(steps) > max_scroll:
            # cap scroll speed, preserving direction (sign)
            steps *= max_scroll / abs(steps)
        step += int(steps) # decimal point num on some platforms
    elif isinstance(event, backend_bases.KeyEvent):
        # finer-grained movements through keyboard controls since the 
        # finest scroll movements may be > 1
        if event.key == "up":
            step += 1
        elif event.key == "down":
            step -= 1
        elif jump is not None and event.key == "right":
            z = jump(event)
            if z: z_overview = z
    
    z_overview_new = z_overview + step
    #print("scroll step of {} to z {}".format(step, z_overview))
    if z_overview_new < 0:
        z_overview_new = 0
    elif z_overview_new >= max_size:
        z_overview_new = max_size - 1
    return z_overview_new

def hide_axes(ax):
    """Hides x- and y-axes.
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def fit_frame_to_image(fig, shape, aspect):
    """Compress figure to fit image only.
    
    Args:
        fig: Figure to compress.
        shape: Shape of image to which the figure will be fit.
        aspect: Aspect ratio of image.
    """
    #fig.tight_layout(pad=-0.4) # neg padding to remove thin left border
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

def set_overview_title(ax, plane, z_overview, zoom="", level=0, 
                        max_intens_proj=False):
    """Set the overview image title.
    
    Args:
        ax: Matplotlib axes on which to display the title.
        plane: Plane string.
        z_overview: Value along the axis corresponding to that plane.
        zoom: String showing zoom information; defaults to "".
        level: Overview view image level, where 0 is unzoomed, 1 is the 
            next zoom, etc; defaults to 0.
        max_intens_proj: True to add maximum intensity projection 
            information to the first overview subplot; defaults to False.
    """
    plane_axis = get_plane_axis(plane)
    if level == 0:
        # show the axis and axis value for unzoomed overview
        title = "{}={} at {}".format(plane_axis, z_overview, zoom)
        if max_intens_proj:
            title = "Max Intensity Projection of ROI\nstarting from {}".format(
                title)
    else:
        # show zoom for subsequent overviews
        title = zoom
    ax.set_title(title)

def set_scinot(ax, lims=(-3, 4), lbls=None, units=None):
    """Set scientific notation for tick labels and shift exponents from 
    axes to their labels.
    
    Scientific notation in Matplotlib positions the exponent at the top 
    of the y-axis and right of the x-axis, which may be missed or overlap 
    with the title or other labels. This method sets scientific notation 
    along with axis labels and units and moves any exponent to the 
    unit labels. Units will be formatted with math text.
    
    Args:
        ax: Axis object.
        lims: Scientific notation limits as a sequence of lower and 
            upper bounds outside of which scientific notation will 
            be used for each applicable axis. Defaults to ``(-2, 4)``.
        lbls: Sequence of axis labels given in the order ``(y-axis, x-axis)``. 
            Defaults to None, which causes the corresponding value from 
            :attr:`config.plot_labels` to be used if available. A None element 
            prevents the label main text from displaying and will show the 
            unit without parentheses if available.
        units: Sequence of units given in the order ``(y-axis, x-axis)``. 
            Defaults to None, which causes the corresponding value from 
            :attr:`config.plot_labels` to be used if available. A None 
            element prevents unit display other than any scientific notation 
            exponent.
    """
    # set scientific notation
    ax.ticklabel_format(style="sci", scilimits=lims, useMathText=True)
    if not lbls:
        lbls = (config.plot_labels[config.PlotLabels.Y_LABEL], 
                 config.plot_labels[config.PlotLabels.X_LABEL])
    if not units:
        units = (config.plot_labels[config.PlotLabels.Y_UNIT], 
                 config.plot_labels[config.PlotLabels.X_UNIT])
    num_lbls = len(lbls)
    num_units = len(units)
    for i, axis in enumerate((ax.yaxis, ax.xaxis)):
        # set labels and units for each axis unless the label is not given
        lbl = lbls[i] if num_lbls > i else None
        
        # either tighten layout or draw first to populate exp text
        ax.figure.canvas.draw()
        offset_text = axis.get_offset_text().get_text()
        unit_all = []
        if offset_text != "":
            # prepend unit with any exponent
            unit_all.append(offset_text)
            axis.offsetText.set_visible(False)
        unit = units[i] if num_units > i else None
        if unit is not None:
            # format unit with math text
            unit_all.append("${{{}}}$".format(unit))
        if lbl and unit_all:
            # put unit in parentheses and combine with label main text
            lbl = "{} ({})".format(lbl, " ".join(unit_all))
        elif unit_all:
            # display unit alone, without parentheses
            lbl = " ".join(unit_all)
        if lbl:
            axis.set_label_text(lbl)

def get_plane_axis(plane):
    """Gets the name of the plane corresponding to the given axis.
    
    Args:
        plane: An element of :attr:``config.PLANE``.
    
    Returns:
        The axis name orthogonal to :attr:``config.PLANE``.
    """
    plane_axis = "z"
    if plane == config.PLANE[1]:
        plane_axis = "y"
    elif plane == config.PLANE[2]:
        plane_axis = "x"
    return plane_axis

def get_roi_path(path, offset, roi_size):
    """Get a string describing an ROI for an image at a given path.
    
    Args:
        path: Path to include in string, without extension.
        offset: Offset of ROI.
        roi_size: Shape of ROI.
    
    Returns:
        String with ``path`` without extension followed immediately by 
        ``offset`` and ``roi_size`` as tuples, with all spaces removed.
    """
    return "{}_offset{}x{}".format(
        os.path.splitext(path)[0], tuple(offset), 
        tuple(roi_size)).replace(" ", "")


def save_fig(path, ext, modifier=""):
    """Save figure, swapping in the given extension for the extension
    in the given path.

    Args:
        path: Base path to use.
        ext: Extension to swap into the extension in ``path``. If None,
            the figure will not be saved.
        modifier: Modifier string to append before the extension;
            defaults to an empty string.
    """
    if ext is not None and not ext in config.FORMATS_3D:
        plot_path = "{}{}.{}".format(os.path.splitext(path)[0], modifier, ext)
        lib_clrbrain.backup_file(plot_path)
        plt.savefig(plot_path)
        print("exported figure to", plot_path)

if __name__ == "__main__":
    print("Starting plot support")
