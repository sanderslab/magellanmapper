# Plot Support for MagellanMapper
# Author: David Young, 2018, 2020
"""Shared plotting functions with the MagellanMapper package.
"""

import os
import warnings

import numpy as np
from matplotlib import backend_bases
from matplotlib import gridspec
from matplotlib import pyplot as plt
from skimage import transform

from magmap.plot import colormaps
from magmap.settings import config
from magmap.io import libmag
from magmap.plot import plot_3d

try:
    from matplotlib_scalebar import scalebar
except ImportError as e:
    scalebar = None
    warnings.warn(config.WARN_IMPORT_SCALEBAR, ImportWarning)


class ImageSyncMixin:
    """Mixin class for synchronizing image display settings."""

    @staticmethod
    def get_img_display_settings(plot_eds, imgi, chl=None):
        """Get display settings for the given image.

        Args:
            plot_eds (List[:obj:`magmap.gui.plot_editor.PlotEditor`]):
                Sequence of Plot Editors. Only the first editor will be
                used on the assumption that all the editors are synchronized.
            imgi (int): Index of image.
            chl (int): Index of channel; defaults to None.

        Returns:
            :obj:`magmap.gui.plot_editor.PlotAxImg`: The currently
                displayed image, or None if ``plot_eds`` is empty.

        """
        if plot_eds:
            return plot_eds[0].get_displayed_img(imgi, chl)
        return None

    @staticmethod
    def update_imgs_display(plot_eds, imgi, chl=None, minimum=np.nan,
                            maximum=np.nan, brightness=None, contrast=None,
                            alpha=None):
        """Update dislayed image settings in all Plot Editors.

        Args:
            plot_eds (List[:obj:`magmap.gui.plot_editor.PlotEditor`]):
                Sequence of Plot Editors. Only the first editor will be
                used on the assumption that all the editors are synchronized.
            imgi (int): Index of image.
            chl (int): Index of channel; defaults to None.
            minimum (float): Vmin; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            maximum (float): Vmax; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            brightness (float): Brightness addend; defaults to None.
            contrast (float): Contrast multiplier; defaults to None.
            alpha (float): Opacity value; defalts to None.
        
        Returns:
            :obj:`magmap.plot_editor.PlotAxImg`: The updated axes image plot.

        """
        plot_ax_img = None
        for ed in plot_eds:
            plot_ax_img = ed.update_img_display(
                imgi, chl, minimum, maximum, brightness, contrast, alpha)
        return plot_ax_img


def imshow_multichannel(ax, img2d, channel, cmaps, aspect, alpha=None,
                        vmin=None, vmax=None, origin=None, interpolation=None,
                        norms=None, nan_color=None, ignore_invis=False):
    """Show multichannel 2D image with channels overlaid over one another.

    Applies :attr:`config.transform` with :obj:`config.Transforms.ROTATE`
    to rotate images. If not available, also checks the first element in
    :attr:``config.flip`` to rotate the image by 180 degrees.
    
    Applies :attr:`config.transform` with :obj:`config.Transforms.FLIP_HORIZ`
    and :obj:`config.Transforms.FLIP_VERT` to invert images.

    Args:
        ax: Axes plot.
        img2d: 2D image either as 2D (y, x) or 3D (y, x, channel) array.
        channel: Channel to display; if None, all channels will be shown.
        cmaps: List of colormaps corresponding to each channel. Colormaps 
            can be the names of specific maps in :mod:``config``.
        aspect: Aspect ratio.
        alpha (float, List[float]): Transparency level for all channels or 
            sequence of levels for each channel. If any value is 0, the
            corresponding image will not be output. Defaults to None to use 1.
        vmin (float, List[float]): Scalar or sequence of vmin levels for
            all channels; defaults to None.
        vmax (float, List[float]): Scalar or sequence of vmax levels for
            all channels; defaults to None.
        origin: Image origin; defaults to None.
        interpolation: Type of interpolation; defaults to None.
        norms: List of normalizations, which should correspond to ``cmaps``.
        nan_color (str): String of color to use for NaN values; defaults to
            None to leave these pixels empty.
        ignore_invis (bool): True to give None instead of an ``AxesImage``
            object that would be invisible; defaults to False.
    
    Returns:
        List of ``AxesImage`` objects.
    """
    # assume that 3D array has a channel dimension
    multichannel, channels = plot_3d.setup_channels(img2d, channel, 2)
    img = []
    num_chls = len(channels)
    if alpha is None:
        alpha = 1
    if num_chls > 1 and not libmag.is_seq(alpha):
        # if alphas not explicitly set per channel, make all channels more
        # translucent at a fixed value that is higher with more channels
        alpha /= np.sqrt(num_chls + 1)

    # transform image based on config parameters
    rotate = config.transform[config.Transforms.ROTATE]
    if rotate is not None:
        last_axis = img2d.ndim - 1
        if multichannel:
            last_axis -= 1
        # use first rotation value
        img2d = np.rot90(
            img2d, libmag.get_if_within(rotate, 0), (last_axis - 1, last_axis))

    for chl in channels:
        img2d_show = img2d[..., chl] if multichannel else img2d
        cmap = None if cmaps is None else cmaps[chl]
        norm = None if norms is None else norms[chl]
        cmap = colormaps.get_cmap(cmap)
        if cmap is not None and nan_color:
            # given color for masked values such as NaNs to distinguish from 0
            cmap.set_bad(color=nan_color)
        # get setting corresponding to the channel index, or use the value
        # directly if it is a scalar
        vmin_plane = libmag.get_if_within(vmin, chl)
        vmax_plane = libmag.get_if_within(vmax, chl)
        alpha_plane = libmag.get_if_within(alpha, chl)
        img_chl = None
        if not ignore_invis or alpha_plane > 0:
            # skip display if alpha is 0 to avoid outputting a hidden image 
            # that may show up in other renderers (eg PDF viewers)
            img_chl = ax.imshow(
                img2d_show, cmap=cmap, norm=norm, aspect=aspect, 
                alpha=alpha_plane, vmin=vmin_plane, vmax=vmax_plane, 
                origin=origin, interpolation=interpolation)
        img.append(img_chl)
    
    # flip horizontally or vertically by inverting axes
    if config.transform[config.Transforms.FLIP_HORIZ]:
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
    if config.transform[config.Transforms.FLIP_VERT]:
        inverted = ax.yaxis_inverted()
        if (origin in (None, "lower") and inverted) or (
                origin == "upper" and not inverted):
            # invert only if inversion state is same as expected from origin
            # to avoid repeated inversions with repeated calls
            ax.invert_yaxis()
    
    return img


def overlay_images(ax, aspect, origin, imgs2d, channels, cmaps, alphas=None,
                   vmins=None, vmaxs=None, ignore_invis=False,
                   check_single=False):
    """Show multiple, overlaid images.
    
    Wrapper function calling :meth:`imshow_multichannel` for multiple 
    images. The first image is treated as a sample image with potential 
    for multiple channels. Subsequent images are typically label images, 
    which may or may not have multple channels.
    
    Args:
        ax: Axes.
        aspect: Aspect ratio.
        origin: Image origin.
        imgs2d (List[:obj:`np.ndarray`]): Sequence of 2D images to display,
            where the first image may be 2D+channel.
        channels (List[List[int]): A nested list of channels to display for
            each image, or None to use :attr:``config.channel`` for the
            first image and 0 for all subsequent images.
        cmaps: Either a single colormap for all images or a list of 
            colormaps corresponding to each image. Colormaps of type 
            :class:`colormaps.DiscreteColormap` will have their 
            normalization object applied as well. If a color is given for
            :obj:`config.AtlasLabels.BINARY` in :attr:`config.atlas_labels`,
            images with :class:`colormaps.DiscreteColormap` will be
            converted to NaN for foreground to use this color.
        alphas: Either a single alpha for all images or a list of 
            alphas corresponding to each image. Defaults to None to use
            :attr:`config.alphas`, filling with 0.9 for any additional
            values required and :attr:`config.plot_labels` for the first value.
        vmins: A list of vmins for each image; defaults to None to use 
            :attr:``config.vmins`` for the first image and None for all others.
        vmaxs: A list of vmaxs for each image; defaults to None to use 
            :attr:``config.vmax_overview`` for the first image and None 
            for all others.
        ignore_invis (bool): True to avoid creating ``AxesImage`` objects
            for images that would be invisible; defaults to False.
        check_single (bool): True to check for images with a single unique
            value displayed with a :class:`colormaps.DiscreteColormap`, which
            will not update for unclear reasons. If found, the final value
            will be incremented by one as a workaround to allow updates.
            Defaults to False.
    
    Returns:
        Nested list containing a list of ``AxesImage`` objects 
        corresponding to display of each ``imgs2d`` image.
    """
    ax_imgs = []
    num_imgs2d = len(imgs2d)
    if num_imgs2d < 1: return None

    # fill default values for each set of 2D images
    img_norm_setting = config.roi_profile["norm"]
    if channels is None:
        # list of first channel for each set of 2D images except config
        # channels for main (first) image
        channels = [[0]] * num_imgs2d
        channels[0] = config.channel
    _, channels_main = plot_3d.setup_channels(imgs2d[0], None, 2)
    if vmins is None:
        vmins = [None] * num_imgs2d
    if vmaxs is None:
        vmaxs = [None] * num_imgs2d
    if alphas is None:
        # start with config alphas and pad the remaining values
        alphas = libmag.pad_seq(config.alphas, num_imgs2d, 0.9)

    for i in range(num_imgs2d):
        # generate a multichannel display image for each 2D image
        img = imgs2d[i]
        if img is None: continue
        cmap = cmaps[i]
        norm = None
        nan_color = None
        discrete = isinstance(cmap, colormaps.DiscreteColormap)
        if discrete:
            # get normalization factor for discrete colormaps and convert
            # the image for this scaling
            norm = [cmap.norm]
            img = cmap.convert_img_labels(img)
            cmap = [cmap]
            nan_color = config.atlas_labels[config.AtlasLabels.BINARY]
            if nan_color:
                # convert all foreground to NaN to use the given color;
                # assumes DiscreteColormap sets background as transparent
                img[img != 0] = np.nan
        alpha = alphas[i]
        vmin = vmins[i]
        vmax = vmaxs[i]
        if i == 0:
            # first image is the main intensity image, potentially multichannel
            len_chls_main = len(channels_main)
            alphas_chl = config.plot_labels[config.PlotLabels.ALPHAS_CHL]
            if alpha is None and alphas_chl is not None:
                alpha = libmag.pad_seq(list(alphas_chl), len_chls_main, 0.5)
            if vmin is None and config.vmins is not None:
                vmin = libmag.pad_seq(list(config.vmins), len_chls_main)
            if vmax is None:
                vmax_fill = config.vmax_overview
                if config.vmaxs is None and img_norm_setting:
                    vmax_fill = [max(img_norm_setting)]
                vmax = libmag.pad_seq(list(vmax_fill), len_chls_main)
            if img_norm_setting:
                # normalize main intensity image
                img = libmag.normalize(img, *img_norm_setting)
        elif not all(np.equal(img.shape[:2], imgs2d[0].shape[:2])):
            # resize the image to the main image's shape if shapes differ in
            # xy; assume that the given image is a labels image whose integer
            # identity values should be preserved
            shape = list(img.shape)
            shape[:2] = imgs2d[0].shape[:2]
            img = transform.resize(
                img, shape, order=0, anti_aliasing=False,
                preserve_range=True, mode="reflect").astype(np.int)
        if check_single and discrete and len(np.unique(img)) < 2:
            # WORAROUND: increment the last val of single unique val images
            # shown with a DiscreteColormap (or any ListedColormap) since
            # they otherwise fail to update on subsequent imshow calls
            # for unknown reasons
            img[-1, -1] += 1
        ax_img = imshow_multichannel(
            ax, img, channels[i], cmap, aspect, alpha, vmin,
            vmax, origin, interpolation="none", norms=norm,
            nan_color=nan_color, ignore_invis=ignore_invis)
        ax_imgs.append(ax_img)
    return ax_imgs


def extract_planes(image5d, plane_n, plane=None, max_intens_proj=False):
    """Extract a 2D plane or stack of planes.
    
    Args:
        image5d (:obj:`np.ndarray`): The full image stack in either 
            ``t,z,y,x[,c]`` or ``z,y,x`` formate; if 4 or more dimensions, 
            the first dimension is assumed to be time and ignored.
        plane_n (int, slice): Slice of planes to extract, which can be
            a single index or multiple indices such as would be used for an
            animation.
        plane (str): Type of plane to extract, which should be one of 
            :attribute:`config.PLANES`.
        max_intens_proj (bool): True to show a max intensity projection, which 
            assumes that plane_n is an array of multiple, typically 
            contiguous planes along which the max intensity pixel will 
            be taken. Defaults to False.
    
    Returns:
        Tuple of an array of the image, which is 2D if ``plane_n`` is a 
        scalar or ``max_intens_projection`` is True, or 3D otherwise; 
        the aspect ratio; and the origin value.
    """
    if image5d.ndim >= 4:
        img3d = image5d[0]
    else:
        # TODO: remove?
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
    
    resolutions = config.resolutions[0]
    if plane:
        # transpose resolutions to the given plane
        _, arrs_1d = transpose_images(plane, arrs_1d=[resolutions])
        resolutions = arrs_1d[0]
    res = resolutions[2] # assume scale bar is along x-axis
    if downsample:
        res *= downsample
    scale_bar = scalebar.ScaleBar(
        res, u'\u00b5m', scalebar.SI_LENGTH, box_alpha=0, 
        color=config.roi_profile["scale_bar_color"], location=3)
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
                libmag.swap_elements(np.copy(arr), *indices) 
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
    """Get the aspect ratio and origin for the given plane.

    Inversts the ratio if :attr:`config.transform[config.Transforms.ROTATE]`
    is set to rotate the image by an odd number of 90 degree turns.
    
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
        if config.resolutions is not None:
            aspect = config.resolutions[0, 0] / config.resolutions[0, 2]
    elif plane == config.PLANE[2]:
        # yz plane
        origin = "lower"
        if config.resolutions is not None:
            aspect = config.resolutions[0, 0] / config.resolutions[0, 1]
    else:
        # defaults to "xy"
        if config.resolutions is not None:
            aspect = config.resolutions[0, 1] / config.resolutions[0, 2]
    rotate = config.transform[config.Transforms.ROTATE]
    if rotate and rotate % 2 != 0:
        # invert aspect ratio for 90 or 270 deg rotations
        aspect = 1 / aspect
    return aspect, origin


def scroll_plane(event, z_overview, max_size, jump=None, max_scroll=None):
    """Scroll through overview images along their orthogonal axis.
    
    Args:
        event: Mouse or key event. For mouse events, scroll step sizes 
            will be used for movements. For key events, up/down arrows 
            will be used.
        z_overview: Index of plane to show.
        max_size: Maximum number of planes.
        jump: Function to jump to a given plane; defaults to None.
            Activated if present and the right arrow button is pressed.
        max_scroll: Max number of planes to scroll by mouse. Ignored during 
            jumps.

    Returns:
        int: Index of plane after scrolling.
    """
    step = 0
    if jump is not None and event.key == "right":
        # jump to the given plane for Right arrow key presses +/- mouse press;
        # allowing mouse press serves as workaround to get axes from mouse
        # event, which is absent from key event if fig lost focus
        z = jump(event)
        if z: z_overview = z
    elif isinstance(event, backend_bases.MouseEvent):
        # scroll movements are scaled from 0 for each event
        steps = event.step
        if max_scroll is not None and abs(steps) > max_scroll:
            # cap scroll speed, preserving direction (sign)
            steps *= max_scroll / abs(steps)
        step += int(steps)  # decimal point num on some platforms
    elif isinstance(event, backend_bases.KeyEvent):
        # finer-grained movements through keyboard controls since the 
        # finest scroll movements may be > 1
        if event.key == "up":
            step += 1
        elif event.key == "down":
            step -= 1
    
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

    Use :attr:`config.plot_labels[config.PlotLabels.PADDING]` to configure
    figure padding, which will turn off the constrained layout.
    
    Args:
        fig: Figure to compress.
        shape: Shape of image to which the figure will be fit.
        aspect: Aspect ratio of image.
    """
    pad = config.plot_labels[config.PlotLabels.PADDING]
    if aspect is None:
        aspect = 1
    img_size_inches = np.divide(shape, fig.dpi)  # convert to inches
    print("image shape: {}, img_size_inches: {}, aspect: {}"
          .format(shape, img_size_inches, aspect))
    if aspect > 1:
        fig.set_size_inches(img_size_inches[1], img_size_inches[0] * aspect)
    else:
        # multiply both sides by 1 / aspect => number > 1 to enlarge
        fig.set_size_inches(img_size_inches[1] / aspect, img_size_inches[0])
    if pad:
        # use neg padding to remove thin left border that sometimes appears;
        # NOTE: this setting will turn off constrained layout
        fig.tight_layout(pad=libmag.get_if_within(pad, 0, 0))
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
        if unit is not None and unit != "":
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


def get_plane_axis(plane, get_index=False):
    """Gets the name of the plane corresponding to the given axis.
    
    Args:
        plane (str): An element of :attr:``config.PLANE``.
        get_index (bool): True to get the axis as an index.
    
    Returns:
        The axis name orthogonal to :attr:``config.PLANE`` as string, or 
        the axis index in the order ``z,y,x`` if ``get_index`` is True.
    """
    plane_axis = "z"
    i = 0  # axis index, assuming z,y,x order
    if plane == config.PLANE[1]:
        plane_axis = "y"
        i = 1
    elif plane == config.PLANE[2]:
        plane_axis = "x"
        i = 2
    if get_index:
        return i
    return plane_axis


def setup_images_for_plane(plane, arrs_3d):
    """Set up image arrays and scaling for the given planar orientation.

    Args:
        plane (str): Target planar orientation as one of :const:`config.PLANE`.
        arrs_3d (List[:obj:`np.ndarray`]): Sequence of 3D arrays to transpose.

    Returns:
        List[:obj:`np.ndarray`], float, str, List[float]: Sequence of transposed
        3D arrays; aspect ratio, or None if :attr:``detector.resolutions``
        has not been set; origin, or None for default origin; and transposed
        labels scaling.

    """
    scaling = config.labels_scaling
    if scaling is not None:
        scaling = [scaling]
    arrs_3d_tr, arrs_1d = transpose_images(plane, arrs_3d, scaling)
    aspect, origin = get_aspect_ratio(plane)
    if arrs_1d is not None and len(arrs_1d) > 0:
        scaling = arrs_1d[0]
    return arrs_3d_tr, aspect, origin, scaling


def get_roi_path(path, offset, roi_size=None):
    """Get a string describing an ROI for an image at a given path.
    
    Args:
        path (str): Path to include in string, without extension.
        offset (List[int]): Offset of ROI.
        roi_size (List[int]): Shape of ROI; defaults to None to ignore.
    
    Returns:
        str: String with ``path`` without extension followed immediately by
        ``offset`` and ``roi_size`` as tuples, with all spaces removed.
    """
    size = ""
    if roi_size is not None:
        size = "x{}".format(tuple(roi_size))
    return "{}_offset{}{}".format(
        os.path.splitext(path)[0], tuple(offset), size).replace(" ", "")


def save_fig(path, ext=None, modifier="", fig=None):
    """Save figure, swapping in the given extension for the extension
    in the given path.

    Args:
        path (str): Base path to use.
        ext (str): File format extension for saving, without period. Defaults
            to None to use ``png``.
        modifier (str): Modifier string to append before the extension;
            defaults to an empty string.
        fig (:obj:`matplotlib.figure.Figure`): Figure; defaults to None
            to use the current figure.
    """
    if ext in config.FORMATS_3D:
        print("Extension \"{}\" is a 3D type, will skip saving 2D figure"
              .format(ext))
        return
    if ext is None:
        ext = "png"
    plot_path = "{}{}.{}".format(os.path.splitext(path)[0], modifier, ext)
    libmag.backup_file(plot_path)
    if fig is None:
        # save the current figure
        plt.savefig(plot_path)
    else:
        fig.savefig(plot_path)
    print("exported figure to", plot_path)


def setup_fig(nrows, ncols, size=None):
    """Setup a figure and associated :class:`gridspec.GridSpec`.
    
    Args:
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        size (List[float]): Sequence of figure size in ``(width, height)``
            in inches; defaults to None.

    Returns:
        Tuple of figure and :obj:`gridspec.GridSpec`.

    """
    fig = plt.figure(frameon=False, constrained_layout=True, figsize=size)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    return fig, gs


def show():
    """Simple wrapper to show the current Matplotlib figure using
    :class:`matplotlib.pyplot`, which manages the event loop.
    """
    plt.show()


def get_downsample_max_sizes():
    """Get the maximum sizes by axis to keep an image within size limits
    during downsampling as set in the current atlas profile based on whether
    the images is loaded as a Numpy memmapped array or not.

    Returns:
        List[int]: Sequence of maximum sizes by axis set in the current profile
        if it is also set to downsample images loaded by Numpy, otherwise None.

    """
    max_sizes = None
    downsample_io = config.atlas_profile["editor_downsample_io"]
    if downsample_io and config.image5d_io in downsample_io:
        max_sizes = config.atlas_profile["editor_max_sizes"]
        if max_sizes:
            max_sizes = max_sizes[::-1]
    return max_sizes


if __name__ == "__main__":
    print("Starting plot support")
