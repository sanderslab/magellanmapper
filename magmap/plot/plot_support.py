# Plot Support for MagellanMapper
# Author: David Young, 2018, 2023
"""Shared plotting functions with the MagellanMapper package.
"""
import dataclasses
import pathlib
from collections import OrderedDict
import math
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, \
    TYPE_CHECKING, Tuple, Union

import numpy as np
from matplotlib import backend_bases, gridspec, pyplot as plt
try:
    from matplotlib import layout_engine
except ImportError as e:
    # not available in Matplotlib on Python 3.6
    layout_engine = None
import matplotlib.transforms as transforms
from skimage import filters, img_as_float32, transform

from magmap.atlas import ontology
from magmap.cv import cv_nd
from magmap.plot import colormaps
from magmap.settings import config
from magmap.io import libmag
from magmap.plot import plot_3d

try:
    from matplotlib_scalebar import scalebar
except ImportError as e:
    scalebar = None
    warnings.warn(config.WARN_IMPORT_SCALEBAR, ImportWarning)

if TYPE_CHECKING:
    from matplotlib import axes, colors, figure, image
    from magmap.gui import plot_editor
    from magmap.io import np_io
    import pandas as pd

_logger = config.logger.getChild(__name__)


class ImageSyncMixin:
    """Mixin class for synchronizing editors with Matplotlib figures."""
    
    def __init__(self, img5d):
        #: Image5d image.
        self.img5d: "np_io.Image5d" = img5d
        
        #: Matplotlib figure.
        self.fig: Optional["figure.Figure"] = None
        #: Dictionary of plot editors.
        self.plot_eds: Dict[Any, "plot_editor.PlotEditor"] = OrderedDict()
        #: Edited flag.
        self.edited: bool = False
        
        #: Display images with additive blending; default to False.
        self.additive_blend: bool = False

        #: Plane(s) for max intensity projections.
        self._max_intens_proj: Optional[Union[int, Sequence[int]]] = None
        
        #: Listeners attached to the editor.
        self._listeners: List["backend_bases.Event"] = []

    @property
    def additive_blend(self) -> bool:
        """Get additive blend setting."""
        return self._additive_blend
    
    @additive_blend.setter
    def additive_blend(self, val: bool):
        """Set additive blend setting and propagate to Plot Editors."""
        self._additive_blend = val
        for ed in self.plot_eds.values():
            ed.overlayer.additive_blend = val

    def get_img_display_settings(self, imgi, chl=None):
        """Get display settings for the given image.
        
        Only settings from the first editor will be retrieved on the
        assumption that all the editors are synchronized.

        Args:
            imgi (int): Index of image.
            chl (int): Index of channel; defaults to None.

        Returns:
            :obj:`magmap.gui.plot_editor.PlotAxImg`: The currently
                displayed image, or None if ``plot_eds`` is empty.

        """
        if self.plot_eds:
            return tuple(self.plot_eds.values())[0].get_displayed_img(imgi, chl)
        return None

    def update_imgs_display(
            self, imgi: int, chl: Optional[int] = None,
            minimum: Optional[float] = np.nan,
            maximum: Optional[float] = np.nan,
            brightness: Optional[float] = None,
            contrast: Optional[float] = None, alpha: Optional[float] = None,
            alpha_blend: Optional[float] = None, refresh: bool = False, **kwargs
    ) -> "plot_editor.PlotAxImg":
        """Update dislayed image settings in all Plot Editors.

        Args:
            imgi: Index of image group.
            chl: Index of channel; defaults to None.
            minimum: Vmin; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            maximum: Vmax; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            brightness: Brightness addend; defaults to None.
            contrast: Contrast multiplier; defaults to None.
            alpha: Opacity value; defalts to None.
            alpha_blend: Opacity blending value; defaults to None.
            refresh: True to refresh all zoomed Plot Editors; defaults to False.
            kwargs: Additional arguments, which are ignored.
        
        Returns:
            The updated axes image plot.

        """
        plot_ax_img = None
        for ed in self.plot_eds.values():
            # update the displayed image
            plot_ax_img = ed.update_img_display(
                imgi, chl, minimum, maximum, brightness, contrast, alpha,
                alpha_blend)
            
            if refresh:
                # fully refresh editor
                ed.show_overview()
        
        return plot_ax_img
    
    def save_fig(self, path: str, **kwargs):
        """Save the figure to file, with path based on filename, ROI,
        and overview plane shown.
        
        Args:
            path: Save path.
            kwargs: Additional arguments passed to :meth:`save_fig`.
        
        """
        if not self.fig:
            print("Figure not yet initialized, skipping save")
            return
        # use module save fig function
        save_fig(path, fig=self.fig, **kwargs)
    
    def set_show_labels(self, val):
        """Set whether to show labels for all Plot Editors.
        
        Args:
            val (bool): True to show labels, false otherwise.

        """
        for plot_ed in self.plot_eds.values():
            plot_ed.set_show_label(val)
    
    def set_show_crosslines(self, val):
        """Set the attribute to show crosslines for all Plot Editors.

        Args:
            val (bool): True to show crosslines, false otherwise.

        """
        for plot_ed in self.plot_eds.values():
            plot_ed._show_crosslines = val

    def set_labels_level(self, val: int):
        """Set the labels level all Plot Editors.

        Args:
            val: Labels level.

        """
        for plot_ed in self.plot_eds.values():
            plot_ed.labels_level = val
    
    def show_labels_annots(
            self, show_annots: bool = True,
            annots: Dict[str, Sequence["axes.Axes.Text"]] = None
    ) -> Dict[str, Sequence["axes.Axes.Text"]]:
        """Show labels annotations.
        
        Args:
            show_annots: True (default) to show annotations.
            annots: Dictionary of plane to text artist sequence, which
                serves as a cache to redisplay labels for the same plane.
                Defaults to None, in which case an empty dict will be used.

        Returns:
            ``annots`` dictionary.

        """
        if annots is None:
            annots = {}
        for plot_ed in self.plot_eds.values():
            # create annotations, retrieving from cache if available; assumes
            # that images for the same plane are the same across editors
            annots_plane = annots[
                plot_ed.plane] if plot_ed.plane in annots else None
            plot_ed.show_labels(show_annots, labels_annots=annots_plane)
            annots[plot_ed.plane] = plot_ed.overlayer.labels_annots
        return annots
    
    def update_max_intens_proj(self, shape, display=False):
        """Update max intensity projection planes.
        
        Args:
            shape (Union[int, Sequence[int]]): Number of planes for all
                Plot Editors or sequence of plane counts in ``z,y,x``.
            display (bool): True to trigger an update in the Plot Editors;
                defaults to False.

        """
        self._max_intens_proj = shape
        is_seq = libmag.is_seq(shape)
        for i, ed in enumerate(self.plot_eds.values()):
            n = shape[i] if is_seq else shape
            if n != ed.max_intens_proj:
                ed.max_intens_proj = n
                if display: ed.update_coord()
    
    @staticmethod
    def enable_btn(btn, enable=True, color=None, max_color=0.99):
        """Display a button or other widget as enabled or disabled.

        Note that the button's active state will not change since doing so
        prevents the coloration from changing.

        Args:
            btn (:class:`matplotlib.widgets.AxesWidget`): Widget to change,
                which must have ``color`` and ``hovercolor`` attributes.
            enable (bool): True to enable (default), False to disable.
            color (float): Intensity value from 0-1 for the main color. The
                hovercolor will be just above this value, while the disabled
                main and hovercolors will be just below this value. Defaults
                to None, which will use :attr:`config.widget_color`.
            max_color (float): Max intensity value for hover color; defaults
                to 0.99 to provide at least some contrast with white backgrounds.
        """
        if color is None:
            color = config.widget_color
        if enable:
            # "enable" button by changing to default grayscale color intensities
            btn.color = str(color)
            hover = color + 0.1
            if hover > max_color:
                # intensities > 1 appear to recycle, so clip to max allowable val
                hover = max_color
            btn.hovercolor = str(hover)
        else:
            # "disable" button by making darker and no hover response
            color_disabled = color - 0.2
            if color_disabled < 0: color_disabled = 0
            color_disabled = str(color_disabled)
            btn.color = color_disabled
            btn.hovercolor = color_disabled

    @staticmethod
    def toggle_btn(btn, on=True, shift=0.2, text=None):
        """Toggle a button between on/off modes.

        Args:
            btn: Button widget to change.
            on: True to display the button as on, False as off.
            shift: Float of amount to shift the button color intensity;
                defaults to 0.2.
            text: Tuple of ``(on_text, off_text)`` for the button label;
                defaults to None to keep the original text.
        """
        if on:
            # turn button "on" by darkening intensities and updating label
            btn.color = str(float(btn.color) - shift)
            btn.hovercolor = str(float(btn.hovercolor) - shift)
            if text: btn.label.set_text(text[1])
        else:
            # turn button "off" by lightening intensities and updating label
            btn.color = str(float(btn.color) + shift)
            btn.hovercolor = str(float(btn.hovercolor) + shift)
            if text: btn.label.set_text(text[0])
    
    def axes_exit(self, event: "backend_bases.LocationEvent"):
        """Trigger axes exit for all plot editors.

        Args:
            event: Axes exit event.
        
        """
        for key in self.plot_eds:
            self.plot_eds[key].on_axes_exit(event)

    def on_close(self, *args):
        """Figure close handler.
        
        Disconnects all Plot Editors and listeners.
        
        Args:
            args: Additional arguments, currently ignored.

        """
        for plot_ed in self.plot_eds.values():
            # disconnect listeners in Plot Editor
            plot_ed.disconnect()
        
        if self.fig:
            # disconnect stored listeners
            for listener in self._listeners:
                self.fig.canvas.mpl_disconnect(listener)


class ImageOverlayer:
    """Manager for overlaying multiple images on top of one another."""
    
    @dataclasses.dataclass
    class RotTransform:
        """Rotation transformation settings."""
        #: Matplotlib transformation.
        transform: Optional[transforms.Transform] = None
        #: x-axis limits.
        xlims: Optional[Tuple[int, int]] = None
        #: y-axis limits
        ylims: Optional[Tuple[int, int]] = None
    
    def __init__(
            self, ax, aspect, origin=None, ignore_invis=False, rgb=False,
            additive_blend=False):
        #: Plot axes.
        self.ax: "axes.Axes" = ax
        #: Aspect ratio.
        self.aspect: Union[str, float] = aspect
        #: Image planar orientation, usually either "lower" or None; defaults
        #: to None.
        self.origin: Optional[str] = origin
        #: True to avoid creating ``AxesImage`` objects for images that would
        #: be invisible; defaults to False.
        self.ignore_invis: bool = ignore_invis
        #: True to show images as RGB(A); defaults to False.
        self.rgb: bool = rgb
        
        #: Display images with additive blending; defaults to False.
        self.additive_blend: bool = additive_blend

        #: Dictionary of label IDs to annotation text artists; defaults to an
        #: empty dictionary.
        self.labels_annots: Dict[int, "axes.Axes.Text"] = {}
        
        #: Matplotlib transform object.
        self._transform: Optional["ImageOverlayer.RotTransform"] = None
    
    def setup_transform(
            self, img2d: np.ndarray, rotate: Optional[int] = None,
            center: Optional[Tuple[int, int]] = None
    ) -> Optional["ImageOverlayer.RotTransform"]:
        """Set up transformation from config settings.
        
        Args:
            img2d: 2D+/-channel array.
            rotate: Clockwise rotation in degrees.
            center: Rotation center in `x, y`. Default of None gives `0, 0`.

        Returns:
            :attr:`self._transform` for chained calls.

        """
        if rotate is None:
            # rotate in increments of 90 deg clockwise
            rotate_n = config.transform[config.Transforms.ROTATE]
            rotate = rotate_n * 90 if rotate_n else 0
            
            # rotate by specific deg clockwise
            rotate_deg = config.transform[config.Transforms.ROTATE_DEG]
            if rotate_deg:
                rotate += rotate_deg
        
        if not rotate:
            # no rotation set
            self._transform = self.RotTransform()
            return self._transform
        
        if center is None:
            # default rotation center to origin
            center = (0, 0)
        ctr_x, ctr_y = center
        
        # rotate around given center in data coords
        transf = transforms.Affine2D().rotate_deg_around(
            ctr_x, ctr_y, rotate)
        
        # convert to display coordinates
        transf += self.ax.transData
        
        # measure rotation corner points for the rotation angle in radians
        rad = np.radians(rotate)
        ht, wd = img2d.shape[:2]
        
        x2 = ctr_x + wd * np.cos(rad)
        y2 = ctr_y + wd * np.sin(rad)

        x4 = ctr_x - ht * np.sin(rad)
        y4 = ctr_y + ht * np.cos(rad)

        x3 = x4 + wd * np.cos(rad)
        y3 = y4 + wd * np.sin(rad)
        
        # set axes limits to fit the rotated image
        xlims = (ctr_x, x2, x3, x4)
        xlims = min(xlims), max(xlims)
        ylims = (ctr_y, y2, y3, y4)
        ylims = max(ylims), min(ylims)
        
        self._transform = self.RotTransform(transf, xlims, ylims)

        return self._transform
    
    def imshow_multichannel(
            self, img2d: np.ndarray,
            channel: Optional[Sequence[int]],
            cmaps: Sequence[Union[str, "colors.Colormap"]],
            alpha: Optional[Union[float, Sequence[float]]] = None,
            vmin: Optional[Union[float, Sequence[float]]] = None,
            vmax: Optional[Union[float, Sequence[float]]] = None,
            interpolation: Optional[str] = None,
            norms: Sequence["colors.Normalize"] = None,
            nan_color: Optional[str] = None,
            alpha_blend: Optional[float] = None, rgb: bool = False
    ) -> List["image.AxesImage"]:
        """Show multichannel 2D image with channels overlaid over one another.
    
        Applies :attr:`config.transform` with :obj:`config.Transforms.ROTATE`
        to rotate images. If not available, also checks the first element in
        :attr:``config.flip`` to rotate the image by 180 degrees.
        
        Applies :attr:`config.transform` with :obj:`config.Transforms.FLIP_HORIZ`
        and :obj:`config.Transforms.FLIP_VERT` to invert images.
    
        Args:
            img2d: 2D image either as 2D (y, x) or 3D (y, x, channel) array.
            channel: Sequence of channels to display; if None, all channels
                will be shown.
            cmaps: List of colormaps corresponding to each channel. Colormaps
                can be the names of specific maps in :mod:``config``.
            alpha: Transparency level for all channels or sequence of levels
                for each channel. If any value is 0, the corresponding image
                will not be output. Defaults to None to use 1.
            vmin: Scalar or sequence of vmin levels for
                all channels; defaults to None.
            vmax: Scalar or sequence of vmax levels for
                all channels; defaults to None.
            interpolation: Type of interpolation; defaults to None.
            norms: List of normalizations, which should correspond to ``cmaps``.
            nan_color: String of color to use for NaN values; defaults to
                None to leave these pixels empty.
            alpha_blend: Opacity blending value; defaults to None.
            rgb: True to display as RGB(A); defaults to False.
        
        Returns:
            List of Matplotlib image objects.
        """
        # assume that 3D array has a channel dimension
        multichannel, channels = plot_3d.setup_channels(img2d, channel, 2)
        if rgb:
            # only generate one image, using 1st channel's settings
            channels = [0]
        img = []
        num_chls = len(channels)
        
        if alpha is None:
            alpha = 1
        if num_chls > 1:
            alpha_bl = libmag.get_if_within(alpha_blend, 0)
            if alpha_bl is not None:
                # alpha blend first two images
                alpha1, alpha2 = alpha_blend_intersection(
                    img2d[..., 0], img2d[..., 1], alpha_bl)
                alpha = np.stack((alpha1, alpha2))
            elif not libmag.is_seq(alpha):
                # if alphas not explicitly set per channel, make all channels more
                # translucent at a fixed value that is higher with more channels
                alpha /= np.sqrt(num_chls + 1)
        
        for chl in channels:
            if rgb:
                # Matplotlib requires 0-1 float or 0-255 int range
                img2d_show = img_as_float32(img2d)
            else:
                # get single channel
                img2d_show = img2d[..., chl] if multichannel else img2d
            
            cmap = None if cmaps is None else cmaps[chl]
            norm = None if norms is None else norms[chl]
            cmap = colormaps.get_cmap(cmap)
            if cmap is not None and nan_color:
                # set color for masked values such as NaNs to distinguish from 0
                cmap.set_bad(color=nan_color)
            
            # get setting corresponding to the channel index, or use the value
            # directly if it is a scalar
            vmin_plane = libmag.get_if_within(vmin, chl)
            vmax_plane = libmag.get_if_within(vmax, chl)
            alpha_plane = libmag.get_if_within(alpha, chl)
            
            img_chl = None
            if not self.ignore_invis or alpha_plane > 0:
                # skip display if alpha is 0 to avoid outputting a hidden image
                # that may show up in other renderers (eg PDF viewers)
                
                if self.additive_blend:
                    # colorize channel before merging; normalize to fit in
                    # expected colormap range
                    in_range = (
                        "image" if vmin_plane is None or vmax_plane is None
                        else (vmin_plane, vmax_plane))
                    img2d_norm = libmag.normalize(img2d_show, 0, 1, in_range)
                    img_chl = cmap(img2d_norm)
                
                else:
                    # display the channel
                    img_chl = self.ax.imshow(
                        img2d_show, cmap=cmap, norm=norm, aspect=self.aspect,
                        alpha=alpha_plane, vmin=vmin_plane, vmax=vmax_plane,
                        origin=self.origin, interpolation=interpolation)
            
            img.append(img_chl)
        
        if self.additive_blend:
            # merge colorized channels, set to full opacity, and set reference
            # to image in each output channel
            img_blended = np.max(np.stack(img, axis=2), axis=2)
            img_chl = self.ax.imshow(img_blended, alpha=1)
            img = [img_chl] * num_chls
        
        if self._transform is None:
            # set up transformation such as rotation
            self.setup_transform(img2d)

        # apply transformation to main axes components and axes limits
        transf = self._transform
        if transf.transform:
            for n in self.ax.images + self.ax.lines + self.ax.collections:
                n.set_transform(transf.transform)
        if transf.xlims is not None:
            self.ax.set_xlim(*transf.xlims)
        if transf.ylims is not None:
            self.ax.set_ylim(*transf.ylims)
        
        # flip horizontally or vertically by inverting axes
        if config.transform[config.Transforms.FLIP_HORIZ]:
            if not self.ax.xaxis_inverted():
                self.ax.invert_xaxis()
        if config.transform[config.Transforms.FLIP_VERT]:
            inverted = self.ax.yaxis_inverted()
            if (self.origin in (None, "lower") and inverted) or (
                    self.origin == "upper" and not inverted):
                # invert only if inversion state is same as expected from origin
                # to avoid repeated inversions with repeated calls
                self.ax.invert_yaxis()
        
        bgd = config.plot_labels[config.PlotLabels.BACKGROUND]
        if bgd:
            # change the background color
            self.ax.set_facecolor(bgd)
        
        return img
    
    def overlay_images(
            self, imgs2d: Sequence[np.ndarray],
            channels: Optional[List[List[int]]],
            cmaps: Sequence[Union[
                str, "colors.Colormap", colormaps.DiscreteColormap]],
            alphas: Optional[Union[
                float, Sequence[Union[float, Sequence[float]]]]] = None,
            vmins: Optional[Union[
                float, Sequence[Union[float, Sequence[float]]]]] = None,
            vmaxs: Optional[Union[
                float, Sequence[Union[float, Sequence[float]]]]] = None,
            check_single: bool = False,
            alpha_blends: Optional[Union[
                float, Sequence[Union[float, Sequence[float]]]]] = None
    ) -> Optional[List[List["image.AxesImage"]]]:
        """Show multiple, overlaid images.
        
        Wrapper function calling :meth:`imshow_multichannel` for multiple
        images. The first image is treated as a sample image with potential
        for multiple channels. Subsequent images are typically label images,
        which may or may not have multple channels.
        
        Args:
            imgs2d: Sequence of 2D images to display,
                where the first image may be 2D+channel.
            channels: A nested list of channels to display for
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
            check_single: True to check for images with a single unique
                value displayed with a :class:`colormaps.DiscreteColormap`, which
                will not update for unclear reasons. If found, the final value
                will be incremented by one as a workaround to allow updates.
                Defaults to False.
            alpha_blends: Opacity blending values for each image in ``imgs2d``;
                defaults to None.
        
        Returns:
            Nested list containing a list of Matplotlib image objects 
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
        if alpha_blends is None:
            alpha_blends = [None] * num_imgs2d
    
        for i in range(num_imgs2d):
            # generate a multichannel display image for each 2D image
            img = imgs2d[i]
            if img is None: continue
            cmap = cmaps[i]
            norm = None
            nan_color = config.plot_labels[config.PlotLabels.NAN_COLOR]
            discrete = isinstance(cmap, colormaps.DiscreteColormap)
            if discrete:
                if config.atlas_labels[config.AtlasLabels.BINARY]:
                    # binarize copy of labels image plane
                    img = np.copy(img)
                    img[img != 0] = 1
                # get normalization factor for discrete colormaps and convert
                # the image for this indexing
                img = cmap.convert_img_labels(img)
                norm = [cmap.norm]
                cmap = [cmap]
            alpha = alphas[i]
            alpha_blend = alpha_blends[i]
            vmin = vmins[i]
            vmax = vmaxs[i]
            rgb = False
            
            if i == 0:
                # first image is the main intensity image, potentially multichannel
                len_chls_main = len(channels_main)
                alphas_chl = config.plot_labels[config.PlotLabels.ALPHAS_CHL]
                if alphas_chl is not None:
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
                # currently only support RGB in main image
                rgb = self.rgb
            
            elif not all(np.equal(img.shape[:2], imgs2d[0].shape[:2])):
                # resize the image to the main image's shape if shapes differ in
                # xy; assume that the given image is a labels image whose integer
                # identity values should be preserved
                shape = list(img.shape)
                shape[:2] = imgs2d[0].shape[:2]
                img = transform.resize(
                    img, shape, order=0, anti_aliasing=False,
                    preserve_range=True, mode="reflect").astype(int)
            
            if check_single and discrete and len(np.unique(img)) < 2:
                # WORKAROUND: increment the last val of single unique val images
                # shown with a DiscreteColormap (or any ListedColormap) since
                # they otherwise fail to update on subsequent imshow calls
                # for unknown reasons
                img[-1, -1] += 1
            
            # use nearest neighbor interpolation for consistency across backends;
            # "none" would fallback to this method, but PDF would use no interp
            ax_img = self.imshow_multichannel(
                img, channels[i], cmap, alpha, vmin, vmax,
                interpolation="nearest", norms=norm, nan_color=nan_color,
                alpha_blend=alpha_blend, rgb=rgb)
            ax_imgs.append(ax_img)
        
        return ax_imgs
    
    def annotate_labels(
            self, labels_2d: np.ndarray, ref_lookup: Optional[Dict[int, Any]],
            level: Optional[int] = None,
            labels_annots: Optional[Dict[int, "axes.Axes.Text"]] = None,
            over_label: bool = True,
            cmap: Optional["colormaps.DiscreteColormap"] = None,
            color_bbox: bool = True,
            label_names: Optional[Dict[int, str]] = None,
            kwargs: Dict[str, Any] = None):
        """Annotate labels with acronyms.
        
        Args:
            labels_2d: 2D labels image.
            ref_lookup: Labels lookup dictionary of label IDs to label metadata.
                Can be None if ``over_label`` is False and ``label_names`` is
                given.
            level: Ontology level; defaults to None.
            labels_annots: Text artists from which new artists will be
                re-created with the same name and positions. Takes precedence
                over ``labels_2d`` and defaults to None.
            over_label: True (default) to ensure that the annotation is over
                a label pixel. Otherwise, places the annotation at the label's
                centroid, whether or not it is a label pixel, which may be
                useful for label edges.
            cmap: Discrete colormap to color the label based on its ID.
                Defaults to None, in which case the color will be black.
            color_bbox: True (default) to color the label bounding box instead
                of the text. Only used if ``cmap`` is given.
            label_names: Dictionary of label IDs to names; defaults to None.
                If given, only these labels and names will be shown.
            kwargs: Dictionary of additional arguments for the text artist.
                Defaults to None.

        """
        if self.labels_annots:
            # reset any existing labels
            self.remove_labels()
        
        labels = {}
        if labels_annots:
            # use existing annotation artists to build dict
            for label_id, annot in labels_annots.items():
                x, y = annot.get_position()
                labels[label_id] = (x, y, annot.get_text())
       
        else:
            # get given label IDs, defaulting to labels image
            label_ids = (np.unique(labels_2d) if label_names is None
                         else label_names.keys())
            
            for label_id in label_ids:
                if over_label:
                    # position label acronym at middle of coordinate list
                    # to ensure that text is over a label pixel
                    coord, reg, region_ids = ontology.get_region_middle(
                        ref_lookup, label_id, labels_2d, incl_children=False)
                    if coord is None: continue
                    y, x = coord
                else:
                    # get measurement properties' centroid for given label
                    props = cv_nd.get_label_props(labels_2d, label_id)
                    if not props: continue
                    y, x = props[0].centroid[:2]
                
                if label_names is None:
                    # get name at the given ontology level
                    atlas_label = ontology.get_label_at_level(
                        label_id, ref_lookup, level)
                    name = ontology.get_label_name(
                        atlas_label, aba_key=config.ABAKeys.ACRONYM)
                    if not name:
                        # make acronym if not in reference
                        name = ontology.get_label_name(atlas_label)
                        name = libmag.make_acronym(name)
                else:
                    # get provided name
                    name = label_names[label_id]
                labels[label_id] = (x, y, name)
        
        # set args for artist bounding box
        bbox = dict(boxstyle="Round,pad=0.1", linewidth=0, alpha=0.3)
        if kwargs is not None and "bbox" in kwargs:
            # update from kwargs and remove from kwargs copy
            bbox.update(kwargs["bbox"])
            kwargs = {k: v for k, v in kwargs.items() if k != "bbox"}
        
        # small annotations with subtle background in case label is dark
        args = dict(
            fontsize="x-small", clip_on=True, horizontalalignment="center",
            verticalalignment="center", bbox=bbox)
        if self._transform and self._transform.transform:
            args["transform"] = self._transform.transform
        text_color = "k"
        facecolor = "xkcd:silver"
        for label_id, label in labels.items():
            if cmap:
                # get color for label in colormap
                color = cmap(cmap.convert_img_labels(label_id))
                if color_bbox:
                    # color bounding box with white text
                    text_color = "w"
                    facecolor = color
                else:
                    # color text
                    text_color = color
            args["color"] = text_color
            bbox["facecolor"] = facecolor
            
            # add label
            if kwargs is not None:
                args.update(kwargs)
            text = self.ax.text(*label, **args)
            self.labels_annots[label_id] = text
    
    def remove_labels(self):
        """Remove label annotations."""
        for text in self.labels_annots.values():
            text.remove()
        self.labels_annots = {}


def alpha_blend_intersection(
        img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5,
        mask1: Optional[np.ndarray] = None,
        mask2: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha blend the intersecting foreground of two images.
    
    Adjust the opacity to blend the parts of images that overlap while
    retaining full opacity for their non-overlapping parts to increase
    contrast and highlight potential misalignments.
    
    Args:
        img1: First image.
        img2: Second image.
        alpha: Alpha level from 0-1 for the first image to use for its
            intersecting area; the second image will use ``1 - alpha``.
        mask1: Foreground mask for ``img1``; defaults to None, in which case
            the foreground will be segmented using Otsu's method.
        mask2: Same for ``img2``; defaults to None.

    Returns:
        The foreground masks with alpha blending for the intersection area.

    """
    # default to getting foreground by Otsu's method
    if mask1 is None:
        mask1 = img1 > filters.threshold_otsu(img1)
    if mask2 is None:
        mask2 = img2 > filters.threshold_otsu(img2)
    
    # alpha blend the intersecting area while leaving the non-overlapping
    # foreground at full opacity and background at full transparency
    intersection = np.logical_and(mask1, mask2)
    mask1 = mask1.astype(float)
    mask2 = mask2.astype(float)
    mask1[intersection] = alpha
    mask2[intersection] = 1 - alpha
    return mask1, mask2


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
    res = resolutions[2]  # assume scale bar is along x-axis
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


def scroll_plane(
        event: backend_bases.MouseEvent, z_overview: int, max_size: int,
        jump: Optional[Callable[[backend_bases.MouseEvent], Any]] = None,
        max_scroll: Optional[int] = None
) -> int:
    """Scroll through overview images along their orthogonal axis.
    
    Args:
        event: Mouse or key event. For mouse events, scroll step sizes
            will be used for movements. For key events, arrows will be used.
        z_overview: Index of plane to show.
        max_size: Maximum number of planes.
        jump: Function to jump to a given plane; defaults to None.
            Activated if present and "j"+click is pressed.
        max_scroll: Max number of planes to scroll by mouse. Ignored during
            jumps.

    Returns:
        int: Index of plane after scrolling.
    
    """
    step = 0
    if isinstance(event, backend_bases.MouseEvent):
        if jump is not None and event.button == 1 and event.key == "j":
            # jump to the given plane for "j"+left-click press; using a
            # mouse event also serves as workaround to get axes as they are
            # absent from key event if fig lost focus as happens sporadically
            z = jump(event)
            if z: z_overview = z
        else:
            # scroll movements are scaled from 0 for each event
            steps = event.step
            if max_scroll is not None and abs(steps) > max_scroll:
                # cap scroll speed, preserving direction (sign)
                steps *= max_scroll / abs(steps)
            step += int(steps)  # decimal point num on some platforms
    elif isinstance(event, backend_bases.KeyEvent):
        # finer-grained movements through keyboard controls since the
        # finest scroll movements may be > 1
        if event.key in ("up", "right"):
            step += 1
        elif event.key in ("down", "left"):
            step -= 1
    
    z_overview_new = z_overview + step
    #print("scroll step of {} to z {}".format(step, z_overview))
    if z_overview_new < 0:
        z_overview_new = 0
    elif z_overview_new >= max_size:
        z_overview_new = max_size - 1
    return z_overview_new


def hide_axes(ax: "axes.Axes", frame_off: bool = False):
    """Hides x- and y-axes and the axes frame.
    
    Args:
        ax: Plot axes.
        frame_off: True to turn off the frame; defaults to False.
    
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if frame_off:
        ax.set_frame_on(False)


def scale_axes(ax, scale_x=None, scale_y=None):
    """Scale axes.
    
    Args:
        ax (:obj:`matplotlib.axes.Axes`): Matplotlib axes.
        scale_x (str): Matplotlib scale mode, eg "linear", "log", "symlog"
            (to include negative values), and "logit", for the x-axis.
            Defaults to None to ignore.
        scale_y (str): Matplotlib scale mode for the y-axis. Defaults to
            None to ignore.

    """
    if scale_x:
        ax.set_xscale(scale_x)
    if scale_y:
        ax.set_yscale(scale_y)


def fit_frame_to_image(
        fig: "figure.Figure", shape: Optional[Sequence[int]] = None,
        aspect: Optional[float] = None):
    """Compress figure to fit image only.

    Use :attr:`config.plot_labels[config.PlotLabels.PADDING]` to configure
    figure padding.
    
    Args:
        fig: Figure to compress.
        shape: Shape of image to which the figure will be fit. Default of None
            uses the first axes bounding box.
        aspect: Aspect ratio of image. Default of None gives 1.
    
    """
    axs = fig.axes
    if shape is None and axs:
        # default shape is bounding box of axes
        shape = np.abs([
            np.diff(axs[0].get_ylim())[0],
            np.diff(axs[0].get_xlim())[0]])
    
    pad = config.plot_labels[config.PlotLabels.PADDING]
    if aspect is None:
        aspect = 1
    img_size_inches = np.divide(shape, fig.dpi)  # convert to inches
    
    if aspect > 1:
        fig.set_size_inches(img_size_inches[1], img_size_inches[0] * aspect)
    else:
        # multiply both sides by 1 / aspect => number > 1 to enlarge
        fig.set_size_inches(img_size_inches[1] / aspect, img_size_inches[0])
    
    if pad:
        # set padding in layout engine
        engine = fig.get_layout_engine()
        padding = libmag.get_if_within(pad, 0, 0)
        if layout_engine is not None and isinstance(
                engine, layout_engine.ConstrainedLayoutEngine):
            engine.set(h_pad=padding, w_pad=padding)
        else:
            fig.tight_layout(pad=padding)


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


def set_scinot(
        ax: "axes.Axes", lims: Sequence[int] = (-3, 4),
        lbls: Optional[Sequence[str]] = None,
        units: Optional[Sequence[str]] = None):
    """Set axes tick scientific notation and shift exponents to their labels.
    
    Scientific notation in Matplotlib positions the exponent at the top
    of the y-axis and right of the x-axis, which may be missed or overlap
    with the title or other labels. This method sets scientific notation
    along with axis labels and units and moves any exponent to the
    unit labels. Units will be formatted with math text.
    
    In some cases, scientific notation is incompatible with the axes'
    formatter and will be ignored. It can often be set up before the plot,
    however, and this function can be called both before and after the plot
    to set up the notation and later override any labeling set up by the plot.
    
    Args:
        ax: Axis object.
        lims: Scientific notation limits as a sequence of lower
            and upper bounds outside of which scientific notation will
            be used for each applicable axis. Defaults to ``(-2, 4)``.
        lbls: Sequence of axis labels given in the order
            ``(y-axis, x-axis)``. Defaults to None, which causes the
            corresponding value from :attr:`config.plot_labels` to be used
            if available. A None element prevents the label main text from
            displaying and will show the unit without parentheses if available.
        units: Sequence of units given in the order
            ``(y-axis, x-axis)``. Defaults to None, which causes the
            corresponding value from :attr:`config.plot_labels` to be used
            if available. A None element prevents unit display other than
            any scientific notation exponent.
    
    """
    # set scientific notation for axes ticks
    try:
        ax.ticklabel_format(style="sci", scilimits=lims, useMathText=True)
    except AttributeError:
        _logger.debug("Could not set up scientific notation, skipping")
    
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
            # format unit with math text, using 3 sets of curly braces
            # (inner = formatting; outer = MathText, x2 to escape from
            # formatting)
            unit_all.append("${{{}}}$".format(unit))
        if lbl and unit_all:
            # put unit in parentheses and combine with label main text
            lbl = "{} ({})".format(lbl, " ".join(unit_all))
        elif unit_all:
            # display unit alone, without parentheses
            lbl = " ".join(unit_all)
        if lbl:
            axis.set_label_text(lbl)


def scale_xticks(
        ax: "axes.Axes", rotation: float,
        x_labels: Optional[Sequence[Any]] = None):
    """Draw x-tick labels with smaller font for increasing number of labels.
    
    Args:
        ax: Matplotlib axes.
        rotation: Label rotation angle.
        x_labels: X-axis labels; defaults to None, in which case the current
            labels will be used.

    """
    if x_labels is None:
        # default to use existing labels
        x_labels = ax.get_xticklabels()
    
    font_size = plt.rcParams["axes.titlesize"]
    if libmag.is_number(font_size):
        # scale font size of x-axis labels by a sigmoid function to rapidly
        # decrease size for larger numbers of labels so they don't overlap
        font_size *= (math.atan(len(x_labels) / 10 - 5) * -2 / math.pi + 1) / 2
    font_dict = {"fontsize": font_size}
    
    # draw x-ticks based on number of bars per group and align to right
    # since center shifts the horiz middle of the label to the center;
    # rotation_mode in dict helps but still slightly off
    ax.set_xticklabels(
        x_labels, rotation=rotation, horizontalalignment="right",
        fontdict=font_dict)
    
    # translate to right since "right" alignment shift the right of labels
    # too far to the left of tick marks; shift less with more groups
    offset = transforms.ScaledTranslation(
        30 / np.cbrt(len(x_labels)) / ax.figure.dpi, 0,
        ax.figure.dpi_scale_trans)
    for lbl in ax.xaxis.get_majorticklabels():
        lbl.set_transform(lbl.get_transform() + offset)


def setup_vspans(
        df: "pd.DataFrame", col_vspan: str, vspan_fmt: str
) -> Tuple[np.ndarray, Sequence[str]]:
    """Set up vertical spans to group axis groups.
    
    Args:
        df: Data frame.
        col_vspan: Column in ``df``, assumed to be ordered by group.
            Changes in value denote the start of the next vertical span.
        vspan_fmt: String formatter for span labels.

    Returns:
        Tuple of a vertical span array of starting indices and a
        sequence of span labels.

    """
    # further group bar groups by vertical spans with location based
    # on each change in value in col_vspan
    # TODO: change .values to .to_numpy when Pandas req >= 0.24
    vspan_vals = df[col_vspan].values
    vspans = np.insert(
        np.where(vspan_vals[:-1] != vspan_vals[1:])[0] + 1, 0, 0)
    vspan_lbls = [vspan_fmt.format(val) if vspan_fmt else str(val)
                  for val in vspan_vals[vspans]]
    return vspans, vspan_lbls


def add_vspans(
        ax: "axes.Axes", vspans: np.ndarray,
        vspan_lbls: Optional[Sequence[str]] = None, padding: float = 1,
        vspan_alt_y: bool = False):
    """Add vertical spans to group x-values.
    
    Shifts legend away from span labels.
    
    Args:
        ax: Matplotlib axes.
        vspans: Sequence of vertical span x-vals in data units.
        vspan_lbls: Sequence of span labels; defaults to None.
        padding: Padding around each span; defaults to 1.
        vspan_alt_y: True to alternate the height of labels; defaults to False.

    """
    # set up span x-val indices
    num_groups = len(ax.get_xticklabels())
    x_offset = padding / 2
    xs = vspans - x_offset
    num_xs = len(xs)
    
    if vspans is not None:
        # show vertical spans alternating in white and black; assume
        # background is already white, so simply skip white shading
        for i, x in enumerate(xs):
            if i % 2 == 0: continue
            end = xs[i + 1] if i < num_xs - 1 else num_groups
            ax.axvspan(x, end, facecolor="k", alpha=0.2, zorder=0)

    if vspan_lbls is not None:
        # show labels for vertical spans
        x_max = num_groups
        for i, x in enumerate(xs):
            # set x to middle of span
            end = xs[i + 1] if i < num_xs - 1 else num_groups
            x = (x + end + x_offset) / 2 / x_max
            
            # position 4% down from top in data coordinates
            y_frac = 0.04
            if vspan_alt_y and i % 2 != 0:
                # shift alternating labels further down to avoid overlap
                y_frac += 0.03
            y = 1 - y_frac
            
            # add text
            ax.text(
                x, y, vspan_lbls[i], color="k", horizontalalignment="center",
                transform=ax.transAxes)
    
    legend = ax.get_legend()
    if legend:
        # shift legend away from span labels
        legend.loc = "best"
        legend.set_bbox_to_anchor((0, 0, 1, 0.9))


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


def save_fig(
        path: Union[str, pathlib.Path], ext: Optional[str] = None,
        modifier: str = "", fig: Optional["figure.Figure"] = None,
        backup: bool = True, **kwargs
) -> Optional[str]:
    """Save figure with support for backup and alternative file formats.
    
    Dots per inch is set by :attr:`config.plot_labels[config.PlotLabels.DPI]`.
    Backs up any existing file before saving. If the found extension is
    not for a supported format for the figure's backend, the figure is not
    saved. Any non-existing parents folders will be created.

    Args:
        path: Base path to use, with or without extension.
        ext: File format extension for saving, with or without period. Defaults
            to None, in which case any extension in ``path`` is used. If no
            extension is found, :const:`magmap.settings.config.DEFAULT_SAVEFIG`
            is used. If the extension is in :const:`config.FORMATS_3D` or
            not supported by Matplotlib, the figure will not be saved.
        modifier: Modifier string to append before the extension;
            defaults to an empty string.
        fig: Figure; defaults to None to use the current figure.
        kwargs: Additional arguments to :meth:`matplotlib.figure.savefig`.
        backup: True (default) to back up any existing file before saving.
    
    Returns:
        The output path, or None if the file was not saved.
    
    """
    # convert potential pathlib path to str
    path = str(path)
    
    # set up additional args to savefig
    if kwargs is None:
        kwargs = {}
    if "dpi" not in kwargs:
        # save the current or given figure with config DPI
        kwargs["dpi"] = config.plot_labels[config.PlotLabels.DPI]
    
    if fig is None:
        # default to using the current figure
        fig = plt.gcf()
    
    # set up output path
    if ext is None:
        # extract extension from path if not given directly
        path_no_ext, ext = os.path.splitext(path)
        if ext:
            # use path without extension
            path = path_no_ext
        else:
            # default to PNG
            ext = config.DEFAULT_SAVEFIG
    if ext.startswith("."):
        # remove preceding period
        ext = ext[1:]
    if path.endswith("."):
        # remove ending period since it will be added later
        path = path[:-1]
    
    if ext in config.FORMATS_3D:
        # skip saving if 3D extension
        _logger.warn(
            f"Extension '{ext}' is a 3D type, will skip saving 2D figure")
        return
    
    if ext not in fig.canvas.get_supported_filetypes().keys():
        # avoid saving if the figure backend does not support the output format
        _logger.warn(
            f"Figure for '{path}' not saved as '{ext}' is not a recognized "
            f"save extension")
        return None
    
    plot_path = f"{path}{modifier}.{ext}"
    if backup:
        # backup any existing file
        libmag.backup_file(plot_path)
    
    # make parent directories if necessary
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    fig.savefig(plot_path, **kwargs)
    _logger.info(f"Exported figure to {plot_path}")
    return plot_path


def setup_fig(
        nrows: int = 1, ncols: int = 1, size: Sequence[float] = None
) -> Tuple["figure.Figure", "gridspec.GridSpec"]:
    """Setup a figure and associated :class:`gridspec.GridSpec`.
    
    Args:
        nrows: Number of rows; defaults to 1.
        ncols: Number of columns; defaults to 1.
        size: Sequence of figure size in ``(width, height)`` in inches;
            defaults to None.

    Returns:
        The figure and grid spec used for its layout.

    """
    fig = plt.figure(frameon=False, layout="constrained", figsize=size)
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    return fig, gs


def show():
    """Simple wrapper to show the current Matplotlib figure using
    :class:`matplotlib.pyplot`, which manages the event loop.
    """
    plt.show()


def get_downsample_max_sizes() -> Dict[str, int]:
    """Get the maximum sizes by axis to keep an image within size limits
    during downsampling as set in the current atlas profile based on whether
    the image is loaded as a NumPy memmapped array or not.

    Returns:
        Dictionary of plane in `xy` format to maximum sizes by axis set in
        the current profile if it is also set to downsample images loaded by
        NumPy, otherwise None.

    """
    max_sizes = None
    downsample_io = config.atlas_profile["editor_downsample_io"]
    if downsample_io and config.img5d and config.img5d.img_io in downsample_io:
        max_sizes = config.atlas_profile["editor_max_sizes"]
        if max_sizes:
            # reverse order to z,y,x since profile is in x,y,z order
            max_sizes = max_sizes[::-1]
    
    if max_sizes:
        # map to planes in xy format
        max_sizes = {p: m for p, m in zip(config.PLANE, max_sizes)}
    return max_sizes


if __name__ == "__main__":
    print("Starting plot support")
