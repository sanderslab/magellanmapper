# 2D overlaid plot editor
# Author: David Young, 2018, 2020
"""Editor for 2D plot with overlaid planes.

Integrates with :class:``atlas_editor.AtlasEditor`` for synchronized 3D 
view of orthogonal planes.
"""

import textwrap
from typing import Callable, List, Optional, Sequence, TYPE_CHECKING

from matplotlib import patches
import numpy as np
from skimage import draw

from magmap.gui import pixel_display
from magmap.io import libmag
from magmap.settings import config
from magmap.atlas import ontology
from magmap.plot import plot_support

if TYPE_CHECKING:
    from matplotlib import axes, colors, image

_logger = config.logger.getChild(__name__)


class PlotAxImg:
    """Axes image storage class.
    
    Tracks settings that may differ between the currently displayed image and
    input values, such as ``None`` to specify auto-intensity and values for
    brightness and contrast that are not stored in the image itself.

    Attributes:
        ax_img: Displayed Matplotlib image.
        vmin: Specified vmin; defaults to None for auto-scaling. See
            ``ax_img.norm.vmin`` for the output vmin.
        vmax: Specified vmax; defaults to None for auto-scaling. See
            ``ax_img.norm.vmax`` for the output vmax.
        brightness: Brightness addend; defaults to 0.0.
        contrast: Contrast factor; defaults to 1.0.
        alpha: Opacity level; defaults to None.
        alpha_blend: Opacity level of the first level in the area of blending
            between two images; defaults to None.

    """
    def __init__(
            self, ax_img: "image.AxesImage", vmin: Optional[float] = None,
            vmax: Optional[float] = None, img: Optional[np.ndarray] = None):
        
        # set from arguments
        self.ax_img = ax_img
        self.vmin = vmin
        self.vmax = vmax
        
        # additional image attributes
        self.brightness: float = 0.0
        self.contrast: float = 1.0
        self.alpha: Optional[float] = None
        self.alpha_blend: Optional[float] = None
        #: True if the image is displayed as RGB(A); defaults to False.
        self.rgb: bool = False
        
        #: Original underlying image data, copied to allow adjusting the array
        #: in ``ax_img`` while retaining the original data unless given by
        #: directly by ``img``.
        self.img: np.ndarray = np.copy(
            self.ax_img.get_array()) if img is None else img


class PlotEditor:
    """Show a scrollable, editable plot of sequential planes in a 3D image.

    Attributes:
        intensity (int): Chosen intensity value of ``img3d_labels``.
        intensity_spec (int): Intensity value specified directly
            rather than chosen from the labels image.
        intensity_shown (int): Displayed intensity in the
            :obj:`matplotlib.AxesImage` corresponding to ``img3d_labels``.
        coord (List[int]): Coordinates in ``z, y, x``.
        scale_bar (bool): True to add a scale bar; defaults to False.
        enable_painting (bool): True to enable label painting; defaults to True.
        max_intens_proj (int): Number of planes to include in a maximum
            intensity projection; defaults to 0 for no projection. The
            planes are taken starting from the given z-value in ``coord``,
            limited by the number of planes available. Applied to the first
            intensity image.

    """
    ALPHA_DEFAULT = 0.5
    _KEY_MODIFIERS = ("shift", "alt", "control")
    
    def __init__(self, overlayer, img3d,
                 img3d_labels, cmap_labels, plane=None,
                 fn_update_coords=None, fn_refresh_images=None,
                 scaling=None, plane_slider=None, img3d_borders=None,
                 cmap_borders=None, fn_show_label_3d=None, interp_planes=None,
                 fn_update_intensity=None, max_size=None, fn_status_bar=None,
                 img3d_extras=None):
        """Initialize the plot editor.
        
        Args:
            fn_refresh_images (function): Callback when refreshing the image.
                Typically takes two arguments, this ``PlotEditor`` object
                and a boolean where True will update synchronized
                ``AtlasEditor``s.
            scaling (List[float]): Scaling/spacing in z,y,x.
            plane_slider (:obj:`matplotlib.widgets.Slider`): Slider for 
                scrolling through planes.
            img3d_labels (:obj:`np.ndarray`): Borders 3D image; defaults 
            to None.
            cmap_borders (:obj:`matplotlib.colors.ListedColormap`): Borders 
                colormap, generally a :obj:`colormaps.DiscreteColormap`; 
                defaults to None.
            fn_show_label_3d (function): Callback to show a label at the 
                current 3D coordinates; defaults to None.
            interp_planes (:obj:`atlas_editor.InterpolatePlanes`): Plane 
                interpolation object; defaults to None.
            fn_update_intensity (function): Callback when updating the 
                intensity value; defaults to None.
            max_size (int): Maximum size of either side of the 2D plane shown;
                defaults to None.
            fn_status_bar (func): Function to call during status bar updates
                in :class:`pixel_display.PixelDisplay`; defaults to None.
            img3d_extras (List[:obj:`np.ndarray`]): Sequence of additional
                intensity images to display; defaults to None.

        """
        #: Manager for plotting overlaid images.
        self.overlayer: "plot_support.ImageOverlayer" = overlayer
        self.axes: "axes.Axes" = self.overlayer.ax
        #: Main 3D image.
        self.img3d: np.ndarray = img3d
        #: Labels 3D image.
        self.img3d_labels: np.ndarray = img3d_labels
        #: Labels  colormap, generally of 
        #: :class:`magmap.plot.colormaps.DiscreteColormap`.
        self.cmap_labels: "colors.ListedColormap" = cmap_labels
        #: One of :attr:`magmap.settings.config.PLANE` specifying the
        #: orthogonal plane to view.
        self.plane: str = plane if plane else config.PLANE[0]
        #: Callback when updating coordinates, typically mouse click events
        #: in x,y; takes two arguments, the updated coordinates and ``plane``
        #: to indicate the coordinates' orientation.
        self.fn_update_coords: Callable[
            [Sequence[int], str], None] = fn_update_coords
        self.fn_refresh_images = fn_refresh_images
        self.scaling = config.labels_scaling if scaling is None else scaling
        self.plane_slider = plane_slider
        if self.plane_slider:
            self.plane_slider.on_changed(self.update_plane_slider)
        self.img3d_borders = img3d_borders
        self.cmap_borders = cmap_borders
        self.fn_show_label_3d = fn_show_label_3d
        self.interp_planes = interp_planes
        self.fn_update_intensity = fn_update_intensity
        self.fn_status_bar = fn_status_bar
        self.img3d_extras = img3d_extras
        
        self.alpha: float = self.ALPHA_DEFAULT
        self.intensity = None  # picked intensity of underlying img3d_label
        self.intensity_spec = None  # specified intensity
        self.intensity_shown = None  # shown intensity in AxesImage
        self.cidpress = None
        self.cidrelease = None
        self.cidmotion = None
        self.cidenter = None
        self.cidleave = None
        self.cidkeypress = None
        self.radius = 5
        self.circle = None
        self.background = None
        self.last_loc = None
        self.last_loc_data = None
        self.press_loc_data = None
        self.connected = False
        self.hline = None
        self.vline = None
        self.coord = None
        self.xlim = None
        self.ylim = None
        self.edited = False  # True if labels image was edited
        self.edit_mode = False  # True to edit with mouse motion
        self.region_label = None
        self.scale_bar = False
        self.max_intens_proj = 0
        self.enable_painting = True
        #: Ontology level at which to show region names.
        self.labels_level: Optional[int] = None

        self._plot_ax_imgs = None
        self._ax_img_labels = None  # displayed labels image
        self._channels = None  # displayed channels list
        # track label editing during mouse click/movement for plane interp
        self._editing = False
        self._show_labels = True  # show atlas labels on mouseover
        self._show_crosslines = False  # show crosslines to orthogonal views
        self._colorbar = None  # colorbar for first main image

        # ROI offset and size in z,y,x
        self._roi_offset = None
        self._roi_size = None
        self._roi_patch_preview = None

        # pre-compute image shapes, scales, and downsampling factors for
        # each type of 3D image
        img3ds = [self.img3d, self.img3d_labels, self.img3d_borders]
        if self.img3d_extras is not None:
            img3ds.extend(self.img3d_extras)
        num_img3ds = len(img3ds)
        self._img3d_shapes = [None] * num_img3ds
        self._img3d_scales = [None] * num_img3ds
        self._downsample = [1] * num_img3ds
        for i, img in enumerate(img3ds):
            if img is None: continue
            self._img3d_shapes[i] = img.shape
            if i > 0:
                lbls_scale = np.divide(img.shape[:3], self.img3d.shape[:3])
                if not all(lbls_scale == 1):
                    # replace with scale if not all 1
                    self._img3d_scales[i] = lbls_scale
            if max_size:
                downsample = max(img.shape[1:3]) // max_size
                if downsample > 1:
                    # only downsample if factor is over 1
                    self._downsample[i] = downsample
        print("plane {} downsampling factors by image: {}"
              .format(self.plane, self._downsample))

    def connect(self):
        """Connect events to functions.
        """
        canvas = self.axes.figure.canvas
        self.cidpress = canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = canvas.mpl_connect(
            "button_release_event", self.on_release)
        self.cidmotion = canvas.mpl_connect(
            "motion_notify_event", self.on_motion)
        self.cidkeypress = canvas.mpl_connect(
            "key_press_event", self.on_key_press)
        self.connected = True

    def disconnect(self):
        """Disconnect event listeners.
        """
        self.circle = None
        listeners = [
            self.cidpress, self.cidrelease, self.cidmotion, self.cidenter,
            self.cidleave, self.cidkeypress]
        for listener in listeners:
            if listener and self._ax_img_labels is not None:
                self._ax_img_labels.figure.canvas.mpl_disconnect(listener)
        self.connected = False

    def update_coord(self, coord=None):
        """Update the displayed image for the given coordinates.

        Scroll to the given z-plane if changed and draw crosshairs to
        indicated the corresponding ``x,y`` values.

        Args:
            coord (List[int]): Coordinates in `z,y,x`, assumed to be transposed
                so the z-plane is show in this editor; defaults to None
                to use :attr:`coord`.

        """
        update_overview = True
        if coord is not None:
            # no need to refresh z-plane if same coordinate
            update_overview = self.coord is None or coord[0] != self.coord[0]
            self.coord = coord
        if update_overview:
            self.show_overview()
        if self._show_crosslines:
            self.draw_crosslines()

    def translate_coord(self, coord, up=False, coord_slice=None):
        """Translate coordinate based on downsampling factor of the main image.

        Coordinates sent to and received from the Atlas Editor are assumed to
        be in the original image space. All overlaid images are assumed to be
        resized to the shape of the main image.

        Args:
            coord (List[int]): Coordinates in z,y,x.
            up (bool): True to upsample; defaults to False to adjust
                coordinates for downsampled images.
            coord_slice (slice): Slice of each set of coordinates to
                transpose. Defaults to None, which gives a slice starting
                at 1 so that the z-value will not be adjusted on the
                assumption that downsampling is only performed in ``x,y``.

        Returns:
            List[int]: The translated coordinates.

        """
        if coord_slice is None:
            coord_slice = slice(1, None)
        coord_tr = np.copy(coord)
        if up:
            coord_tr[coord_slice] = np.multiply(
                coord_tr[coord_slice], self._downsample[0])
        else:
            coord_tr[coord_slice] = np.divide(
                coord_tr[coord_slice], self._downsample[0])
        coord_tr = list(coord_tr.astype(np.int))
        # print("translated from {} to {}".format(coord, coord_tr))
        return coord_tr

    def draw_crosslines(self, show=None):
        """Draw crosshairs depicting the x and y values in orthogonal viewers.
        
        Args:
            show (bool): True to show crosslines, False to make them invisible;
                defaults to None to use :attr:`show_crosslines`.
        
        """
        if show is None:
            show = self._show_crosslines
        
        # translate coordinate down for any downsampling
        coord = self.translate_coord(self.coord)
        if self.hline is None:
            # draw new crosshairs
            self.hline = self.axes.axhline(coord[1], linestyle=":")
            self.vline = self.axes.axvline(coord[2], linestyle=":")
        else:
            # toggle visibility
            self.hline.set_visible(show)
            self.vline.set_visible(show)
            if show:
                # update positions of current crosshairs
                self.hline.set_ydata(coord[1])
                self.vline.set_xdata(coord[2])
    
    def set_show_label(self, val):
        """Set whether to show labels on hover.
        
        Args:
            val (bool): True to show labels, False otherwise.

        """
        if not val:
            # reset text to trigger a figure refresh
            self.region_label.set_text("")
        self._show_labels = val
    
    def show_labels(self, show: bool = True):
        """Show or remove labels for all regions.
        
        Args:
            show: True (default) to show all labels; False to remove them.

        """
        if show:
            if (len(self._plot_ax_imgs) > 1 and
                    self._plot_ax_imgs[1] is not None and config.labels_ref
                    and config.labels_ref.ref_lookup):
                # add label annotations
                self.overlayer.annotate_labels(
                    self._plot_ax_imgs[1][0].img, config.labels_ref.ref_lookup,
                    self.labels_level)
        else:
            # remove all labels
            self.overlayer.remove_labels()
        
        # refresh view
        self.axes.figure.canvas.draw_idle()
    
    def _get_img2d(self, i, img, max_intens=0):
        """Get the 2D image from the given 3D image, scaling and downsampling
        as necessary.

        Args:
            i (int): Index of 3D image in sequence of 3D images, assuming
                order of ``(main_image, labels_img, borders_img)``.
            img (:obj:`np.ndarray`): 3D image from which to extract a 2D plane.
            max_intens (int): Number of planes to incorporate for maximum
                intensity projection; defaults to 0 to not perform this
                projection.

        Returns:
            :obj:`np.ndarray`: 2D plane, downsampled if necessary.

        """
        z = self.coord[0]
        z_scale = 1
        if self._img3d_scales[i] is not None:
            # rescale z-coordinate based on image scaling to the main image
            z_scale = self._img3d_scales[i][0]
            z = int(z * z_scale)
        # downsample to reduce access time; use same factor for both x and y
        # to retain aspect ratio
        downsample = self._downsample[i]
        img = img[:, ::downsample, ::downsample]
        if max_intens:
            # max intensity projection (MIP) across the given number of
            # planes available
            z_stop = z + int(max_intens * z_scale)
            num_z = len(img)
            if z_stop > num_z:
                z_stop = num_z
            z_range = np.arange(z, z_stop)
            img2d = plot_support.extract_planes(
                img[None], z_range, max_intens_proj=True)[0]
        else:
            img2d = img[z]
        return img2d

    def show_overview(self):
        """Show the main 2D plane, taken as a z-plane."""
        self.axes.clear()
        self.hline = None
        self.vline = None
        
        # prep 2D image from main image, assumed to be an intensity image,
        # with settings for each channel within this main image
        imgs2d = [self._get_img2d(0, self.img3d, self.max_intens_proj)]
        self._channels = [config.channel]
        cmaps = [config.cmaps]
        alphas = [config.alphas[0]]
        alpha_blends = [None]
        shapes = [self._img3d_shapes[0][1:3]]
        vmaxs = [None]
        vmins = [None]
        brightnesses = [None]
        contrasts = [None]
        if self._plot_ax_imgs:
            # use vmin/vmax from norm values in previously displayed images
            # if available; None specifies auto-scaling
            vmaxs[0] = [p.vmax if p.vmax is None else p.ax_img.norm.vmax
                        for p in self._plot_ax_imgs[0]]
            vmins[0] = [p.vmin if p.vmin is None else p.ax_img.norm.vmin
                        for p in self._plot_ax_imgs[0]]
            
            # use opacity, brightness, anc contrast from prior images
            alphas[0] = [p.alpha for p in self._plot_ax_imgs[0]]
            alpha_blends[0] = [p.alpha_blend for p in self._plot_ax_imgs[0]]
            brightnesses[0] = [p.brightness for p in self._plot_ax_imgs[0]]
            contrasts[0] = [p.contrast for p in self._plot_ax_imgs[0]]
        
        img2d_lbl = None
        if self.img3d_labels is not None:
            # prep labels with discrete colormap and prior alpha if available
            img2d_lbl = self._get_img2d(1, self.img3d_labels)
            imgs2d.append(img2d_lbl)
            self._channels.append([0])
            cmaps.append(self.cmap_labels)
            alphas.append(
                self._ax_img_labels.get_alpha() if self._ax_img_labels
                else self.alpha)
            alpha_blends.append(None)
            shapes.append(self._img3d_shapes[1][1:3])
            vmaxs.append(None)
            vmins.append(None)
        
        if self.img3d_borders is not None:
            # prep borders image, which may have an extra channels 
            # dimension for multiple sets of borders
            img2d = self._get_img2d(2, self.img3d_borders)
            channels = img2d.ndim if img2d.ndim >= 3 else 1
            for i, channel in enumerate(range(channels - 1, -1, -1)):
                # show first (original) borders image last so that its 
                # colormap values take precedence to highlight original bounds
                img_add = img2d[..., channel] if channels > 1 else img2d
                imgs2d.append(img_add)
                self._channels.append([0])
                cmaps.append(self.cmap_borders[channel])
                
                # get alpha for last corresponding borders plane if available
                ax_img = libmag.get_if_within(self._plot_ax_imgs, 2 + i, None)
                alpha = (ax_img[i].alpha if ax_img else
                         libmag.get_if_within(config.alphas, 2 + i, 1))
                alphas.append(alpha)
                alpha_blends.append(None)
                
                shapes.append(self._img3d_shapes[2][1:3])
                vmaxs.append(None)
                vmins.append(None)

        if self.img3d_extras is not None:
            for i, img in enumerate(self.img3d_extras):
                # prep additional intensity image
                imgi = 3 + i
                imgs2d.append(self._get_img2d(imgi, img))
                self._channels.append([0])
                cmaps.append(("Greys",))
                alphas.append(0.4)
                alpha_blends.append(None)
                shapes.append(self._img3d_shapes[imgi][1:3])
                vmaxs.append(None)
                vmins.append(None)

        # overlay all images and set labels for footer value on mouseover;
        # if first time showing image, need to check for images with single
        # value since they fail to update on subsequent updates for unclear
        # reasons
        ax_imgs = self.overlayer.overlay_images(
            imgs2d, self._channels, cmaps, alphas, vmins, vmaxs,
            check_single=(self._ax_img_labels is None),
            alpha_blends=alpha_blends)
        
        # add or update colorbar
        colobar_prof = config.roi_profile["colorbar"]
        if self._colorbar:
            self._colorbar.update_normal(ax_imgs[0][0])
        elif colobar_prof:
            # store colorbar since it's tied to the artist, which will be
            # replaced with the next display and cannot be further accessed
            self._colorbar = self.axes.figure.colorbar(
                ax_imgs[0][0], ax=self.axes, **colobar_prof)
        
        # display coordinates and label values for each image
        self.axes.format_coord = pixel_display.PixelDisplay(
            imgs2d, ax_imgs, shapes, cmap_labels=self.cmap_labels)

        # trigger actual display through slider or call update directly
        if self.plane_slider:
            self.plane_slider.set_val(self.coord[0])
        else:
            self._update_overview(self.coord[0])
        self.show_roi()
        if self.scale_bar:
            plot_support.add_scale_bar(self.axes, self._downsample[0])

        # store displayed images in the PlotAxImg container class and update
        # displayed brightness/contrast
        if len(ax_imgs) > 1: self._ax_img_labels = ax_imgs[1][0]
        self._plot_ax_imgs = []
        for i, imgs in enumerate(ax_imgs):
            plot_ax_imgs = []
            for j, img in enumerate(imgs):
                # use original 2D labels, without cmap index conversion
                img_orig = img2d_lbl if i == 1 else None
                plot_ax_img = PlotAxImg(img, img=img_orig)
                
                if i == 0:
                    # specified vmin/vmax, in contrast to the AxesImages's
                    # norm, which holds the values used for the displayed image
                    plot_ax_img.vmin = libmag.get_if_within(vmins[i], j)
                    plot_ax_img.vmax = libmag.get_if_within(vmaxs[i], j)
                    
                    # set brightness/contrast
                    self.change_brightness_contrast(
                        plot_ax_img, libmag.get_if_within(brightnesses[i], j),
                        libmag.get_if_within(contrasts[i], j))
                
                # store rest of settings
                plot_ax_img.alpha = libmag.get_if_within(alphas[i], j)
                plot_ax_img.alpha_blend = libmag.get_if_within(
                    alpha_blends[i], j)
                plot_ax_img.rgb = self.overlayer.rgb
                plot_ax_imgs.append(plot_ax_img)
            
            self._plot_ax_imgs.append(plot_ax_imgs)
        
        if self.xlim is not None and self.ylim is not None:
            # restore pan/zoom view
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)
        if not self.connected:
            # connect once get AxesImage
            self.connect()
        # text label with color for visibility on axes plus fig background
        self.region_label = self.axes.text(
            0, 0, "", color="k", bbox=dict(facecolor="xkcd:silver", alpha=0.5))
        self.circle = None
    
    def _update_overview(self, z_overview_new: int):
        """Update overview plot to the given plane.
        
        Ignores updates if the update function is missing or the plane is
        unchanged. In these cases, call :meth:`show_overview` directly.
        
        Args:
            z_overview_new: Z-plane index to show.

        """
        if self.fn_update_coords and z_overview_new != self.coord[0]:
            # move only if step registered and changing position
            coord = list(self.coord)
            coord[0] = z_overview_new
            self.fn_update_coords(coord, self.plane)
    
    def scroll_overview(self, event, only_in_axes=True, fn_jump=None):
        if only_in_axes and event.inaxes != self.axes: return
        z_overview_new = plot_support.scroll_plane(
            event, self.coord[0], self.img3d.shape[0], fn_jump,
            config.max_scroll)
        self._update_overview(z_overview_new)
    
    def update_plane_slider(self, val):
        self._update_overview(int(val))

    def view_subimg(self, offset, size):
        """View a sub-image.

        Args:
            offset (List[int]): Sub-image offset in ``y, x``.
            size (List[int]): Sub-image size in ``y, x``.

        """
        coord_slice = slice(0, None)
        off_trans = self.translate_coord(offset, coord_slice=coord_slice)
        size_trans = self.translate_coord(size, coord_slice=coord_slice)
        # print("view subimg offset", offset, "translated offset", off_trans,
        #       "size", size, "translated size", size_trans)
        self.axes.set_xlim(off_trans[1], off_trans[1] + size_trans[1] - 1)
        ylim = (off_trans[0], off_trans[0] + size_trans[0] - 1)
        ylim_orig = self.axes.get_ylim()
        if ylim_orig[1] - ylim_orig[0] < 0:
            # set "bottom" first (higher y-values) if originally flipped
            ylim = ylim[::-1]
        self.axes.set_ylim(*ylim)
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()

    def show_roi(self, offset=None, size=None, preview=False):
        """Show an ROI as an empty rectangular patch.
        
        If ``offset`` and ``size`` cannot be retrieved, no ROI will be shown.

        Args:
            offset (List[int]): ROI offset in ``y, x``. Defaults to None
                to use the saved ROI offset if available.
            size (List[int]): ROI size in ``y, x``. Defaults to None
                to use the saved ROI size if available.
            preview (bool): True if the ROI should be displayed as a preview,
                which is lighter and transient, removed when another preview
                ROI is displayed. Defaults to False. If False, ROI parameters
                for a displayed ROI will be shown.

        """
        # translate coordinates if given; otherwise, use stored coordinates
        coord_slice = slice(0, None)
        if offset is None:
            offset = self._roi_offset
        else:
            offset = self.translate_coord(offset, coord_slice=coord_slice)
        if size is None:
            size = self._roi_size
        else:
            size = self.translate_coord(size, coord_slice=coord_slice)
        if offset is None or size is None: return
        
        # generate and display ROI patch as empty rectangle, with lighter
        # border for preview ROI
        linewidth = 1 if preview else 2
        linestyle = "--" if preview else "-"
        alpha = 0.5 if preview else 1
        patch = patches.Rectangle(
            offset[::-1], *size[::-1], fill=False, edgecolor="yellow",
            linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        if preview:
            # remove prior preview and store new preview
            if self._roi_patch_preview:
                self._roi_patch_preview.remove()
            self._roi_patch_preview = patch
        else:
            # store coordinates
            self._roi_offset = offset
            self._roi_size = size
        self.axes.add_patch(patch)

    def refresh_img3d_labels(self):
        """Replace the displayed labels image with underlying plane's data.
        """
        if self._ax_img_labels is not None:
            # underlying plane may need to be translated to a linear
            # scale for the colormap before display
            self._ax_img_labels.set_data(self.cmap_labels.convert_img_labels(
                self.img3d_labels[self.coord[0]]))
    
    @staticmethod
    def get_plot_ax_img(plot_ax_imgs, imgi, channels=None, chl=None):
        """Get a plotted image based on image group and channel.

        Args:
            plot_ax_imgs (list[list[:class:`PlotAxImg`]]): Plotted image
                objects, organized as
                ``[[img0_chl0, img0_chl1, ...], [img1_chl0, ...], ....]``.
            imgi (int): Index of image group in ``plot_ax_imgs``.
            channels (list[list[int]]): List of channel lists corresponding
                to ``plot_ax_imgs``; defalts to None to use ``chl`` directly.
            chl (int): Index of channel within the selected image group;
                defaults to None to use the first channel.

        Returns:
            :class:`PlotAxImg`: The selected plotted image, or None if image
            corresponding to ``imgi`` and ``chl`` is not found.

        """
        plot_ax_img = None
        if plot_ax_imgs and imgi < len(plot_ax_imgs):
            if chl is None:
                chl = 0
            if channels:
                # translate channel to index of displayed channels
                if chl not in channels[imgi]:
                    return None
                chl = channels[imgi].index(chl)
            if plot_ax_imgs[imgi] and chl < len(plot_ax_imgs[imgi]): 
                # get image channel within given set of images
                plot_ax_img = plot_ax_imgs[imgi][chl]
        return plot_ax_img

    def get_displayed_img(self, imgi, chl=None):
        """Get display settings for the given image.

        Args:
            imgi (int): Index of image group.
            chl (int): Index of channel within the group; defaults to None.

        Returns:
            :obj:`PlotAxImg`: The currently displayed image.

        """
        return self.get_plot_ax_img(
            self._plot_ax_imgs, imgi, self._channels, chl)

    @staticmethod
    def change_brightness_contrast(
            plot_ax_img: PlotAxImg, brightness: Optional[float],
            contrast: Optional[float]):
        """Change image brightness and contrast.
        
        All changes are made relative to the original image, so both
        brightness and contrast should be given together. For example, if
        brightness is changed, and later contrast is changed, the contrast
        change will wipe out the brightness change unless the brightness
        value is given again.
        
        Args:
            plot_ax_img: Axes image storage instance.
            brightness: Brightness value, centered on 0. Can be None to
                ignore brightness changes.
            contrast: Contrast value, centered on 1. Can be None to
                ignore contrast changes.

        """
        # get displayed image array, which adjusts dynamically to array changes
        data = plot_ax_img.ax_img.get_array()
        
        # get info range from data type, or assume 0-1 for RGB images
        info = libmag.get_dtype_info(data)
        info_range = (0, 1) if plot_ax_img.rgb else (info.min, info.max)
        img = plot_ax_img.img
        
        if brightness is not None:
            # shift original image array by brightness
            img = np.clip(img + brightness, *info_range)
            data[:] = img
            plot_ax_img.brightness = brightness
        
        if contrast is not None:
            # stretch adjusted image array by contrast
            img = np.clip(img * contrast, info.min, info.max)
            data[:] = img
            plot_ax_img.contrast = contrast

    @staticmethod
    def update_plot_ax_img_display(
            plot_ax_img: PlotAxImg, minimum: float = np.nan,
            maximum: float = np.nan, brightness: Optional[float] = None,
            contrast: Optional[float] = None, alpha: Optional[float] = None,
            **kwargs) -> Optional[PlotAxImg]:
        """Update plotted image display settings.

        Args:
            plot_ax_img: Plotted image.
            minimum: Vmin; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            maximum: Vmax; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            brightness: Brightness addend; defaults to None.
            contrast: Contrast multiplier; defaults to None.
            alpha: Opacity value; defalts to None.
            **kwargs: Extra arguments, currently ignored.
        
        Returns:
            The updated axes image plot.

        """
        if not plot_ax_img:
            return None
        if minimum is not np.nan or maximum is not np.nan:
            # store the specified intensity limits
            if minimum is not np.nan:
                plot_ax_img.vmin = minimum
            if maximum is not np.nan:
                plot_ax_img.vmax = maximum
            
            # set vmin and vmax; use norm rather than get_clim since norm
            # holds the actual limits
            clim = [minimum, maximum]
            norm = plot_ax_img.ax_img.norm
            for j, (lim, ax_lim) in enumerate(zip(
                    clim, (norm.vmin, norm.vmax))):
                if lim is np.nan:
                    # default to using current value
                    clim[j] = ax_lim
            if None not in clim and clim[0] > clim[1]:
                # ensure min is <= max, setting to the currently adjusted val
                if minimum is np.nan:
                    clim[0] = clim[1]
                else:
                    clim[1] = clim[0]
            # directly update norm rather than clim so that None vals are
            # not skipped for auto-scaling
            norm.vmin, norm.vmax = clim
            plot_ax_img.ax_img.autoscale_None()
            if norm.vmin > norm.vmax:
                # auto-scaling may cause vmin to exceed vmax
                if minimum is np.nan:
                    norm.vmin = norm.vmax
                else:
                    norm.vmax = norm.vmin
        if brightness is not None or contrast is not None:
            # adjust brightness and contrast together
            PlotEditor.change_brightness_contrast(
                plot_ax_img, brightness, contrast)
        if alpha is not None:
            # adjust opacity
            plot_ax_img.ax_img.set_alpha(alpha)
        return plot_ax_img

    def update_alpha_blend(
            self, imgi: int, alpha_blend: float) -> List[PlotAxImg]:
        """Update alpha blending between two images.

        Args:
            imgi: Index of image group. The first two images in this group
                will be blended.
            alpha_blend: Alpha opacity level for the blending.

        Returns:
            List of the updated axes image storage instances.

        """
        # get the first two channels in the image group
        plot_ax_imgs = [self.get_displayed_img(imgi, c) for c in (0, 1)]
        if None in plot_ax_imgs:
            return plot_ax_imgs
        
        # blend the images
        alpha1, alpha2 = plot_support.alpha_blend_intersection(
            plot_ax_imgs[0].img, plot_ax_imgs[1].img, alpha_blend)
        plot_ax_imgs[0].ax_img.set_alpha(alpha1)
        plot_ax_imgs[1].ax_img.set_alpha(alpha2)
        
        # store the alpha blend level while retaining the original stored alpha
        for p in plot_ax_imgs: p.alpha_blend = alpha_blend
        return plot_ax_imgs

    def update_img_display(
            self, imgi: int, chl: Optional[int] = None, minimum: float = np.nan,
            maximum: float = np.nan, brightness: Optional[float] = None,
            contrast: Optional[float] = None, alpha: Optional[float] = None,
            alpha_blend: Optional[float] = None) -> PlotAxImg:
        """Update dislayed image settings.

        Args:
            imgi: Index of image group.
            chl: Index of channel within the group; defaults to None.
            minimum: Vmin; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            maximum: Vmax; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            brightness: Brightness addend; defaults to None.
            contrast: Contrast multiplier; defaults to None.
            alpha: Opacity value; defaults to None.
            alpha_blend: Opacity blending value; defaults to None. False
                turns off alpha blending, resetting the images to their
                stored alpha values.
        
        Returns:
            The updated axes image plot.

        """
        if alpha_blend is False:
            # turn off alpha blending, assumed to be applied to first 2
            # images, and reset them to stored alpha values
            for c in (0, 1):
                plot_ax_img = self.get_displayed_img(imgi, c)
                if plot_ax_img is not None:
                    plot_ax_img.alpha_blend = None
                    plot_ax_img = self.update_img_display(
                        imgi, c, minimum, maximum, brightness, contrast,
                        plot_ax_img.alpha)
        elif alpha_blend:
            # alpha blend the images
            plot_ax_img = self.update_alpha_blend(imgi, alpha_blend)[0]
        else:
            # update the selected image
            plot_ax_img = self.get_displayed_img(imgi, chl)
            plot_ax_img = self.update_plot_ax_img_display(
                plot_ax_img, minimum, maximum, brightness, contrast, alpha)
        self.axes.figure.canvas.draw_idle()
        return plot_ax_img
    
    def alpha_updater(self, alpha):
        """Update labels image opacity level.
        
        Params:
            alpha (float): Alpha level.
        
        """
        self.alpha = alpha
        if self._ax_img_labels is not None:
            self._ax_img_labels.set_alpha(self.alpha)
        #print("set image alpha to {}".format(self.alpha))
    
    def on_press(self, event):
        """Respond to mouse press events."""
        if event.inaxes != self.axes: return
        x = int(event.xdata)
        y = int(event.ydata)
        self.press_loc_data = (x, y)
        self.last_loc_data = tuple(self.press_loc_data)
        self.last_loc = (int(event.x), int(event.y))
        # re-translate downsampled coordinates back up
        coord = self.translate_coord([self.coord[0], y, x], up=True)
        
        if event.button == 1:
            if self.edit_mode and self.img3d_labels is not None:
                # label painting in edit mode
                if event.key is not None and "alt" in event.key:
                    print("using previously picked intensity instead,",
                          self.intensity)
                elif self.intensity_spec is None:
                    # click while in editing mode to initialize intensity value
                    # for painting, using values at current position for
                    # the underlying and displayed images
                    self.intensity = self.img3d_labels[tuple(coord)]
                    self.intensity_shown = self._ax_img_labels.get_array()[y, x]
                    print("got intensity {} at x,y,z = {},{},{}"
                          .format(self.intensity, *coord[::-1]))
                    if self.fn_update_intensity:
                        # trigger text box update
                        self.fn_update_intensity(self.intensity)
                else:
                    # apply user-specified intensity value; assume it will
                    # be reset on mouse release
                    print("using specified intensity of", self.intensity_spec)
                    self.intensity = self.intensity_spec
                    self.intensity_shown = self.cmap_labels.convert_img_labels(
                        self.intensity)
            
            elif event.key not in self._KEY_MODIFIERS:
                # click without modifiers to update crosshairs and 
                # corresponding planes
                self.coord = coord
                if self.fn_update_coords:
                    self.fn_update_coords(self.coord, self.plane)
            
            if event.key == "3" and self.fn_show_label_3d is not None:
                if self.img3d_labels is not None:
                    # extract label ID and display in 3D viewer
                    self.fn_show_label_3d(self.img3d_labels[tuple(coord)])
    
    def on_axes_exit(self, event):
        if event.inaxes != self.axes: return
        if self.circle:
            self.circle.remove()
            self.circle = None
    
    def on_motion(self, event):
        """Move the editing pen's circle and draw with the chosen intensity 
        value if set.
        """
        if event.inaxes != self.axes: return
        
        # get mouse position and return if no change from last pixel coord
        x = int(event.xdata)
        y = int(event.ydata)
        x_fig = int(event.x)
        y_fig = int(event.y)
        
        loc = (x_fig, y_fig)
        if self.last_loc is not None and self.last_loc == loc:
            return
        
        loc_data = (x, y)
        if event.button == 2 or (event.button == 1 and event.key == "shift"):
            # pan by middle-click or shift+left-click during mouseover
            
            # use data coordinates so same part of image stays under mouse
            dx = x - self.last_loc_data[0]
            dy = y - self.last_loc_data[1]
            xlim = self.axes.get_xlim()
            self.axes.set_xlim(xlim[0] - dx, xlim[1] - dx)
            ylim = self.axes.get_ylim()
            self.axes.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.axes.figure.canvas.draw_idle()
            self.xlim = self.axes.get_xlim()
            self.ylim = self.axes.get_ylim()
            # data itself moved, so update location along with movement
            loc_data = (x - dx, y - dy)
            
        elif event.button == 3 or (
                event.button == 1 and event.key == "control"):
            
            # zooming by right-click or ctrl+click (which coverts 
            # button event to 3 on Mac at least) while moving mouse up/down in y
            
            # use figure coordinates since data pixels will scale 
            # during zoom
            zoom_speed = (y_fig - self.last_loc[1]) * 0.01
            xlim = self.axes.get_xlim()
            xlim_update = (
                xlim[0] + (self.press_loc_data[0] - xlim[0]) * zoom_speed, 
                xlim[1] + (self.press_loc_data[0] - xlim[1]) * zoom_speed)
            ylim = self.axes.get_ylim()
            ylim_update = (
                ylim[0] + (self.press_loc_data[1] - ylim[0]) * zoom_speed, 
                ylim[1] + (self.press_loc_data[1] - ylim[1]) * zoom_speed)
            
            # avoid flip by checking that relationship between high and 
            # low values in updated limits is in the same order as in the 
            # current limits, which might otherwise flip if zoom speed is high
            if ((xlim_update[1] - xlim_update[0]) * (xlim[1] - xlim[0]) > 0 and 
                (ylim_update[1] - ylim_update[0]) * (ylim[1] - ylim[0]) > 0):
                
                self.axes.set_xlim(xlim_update)
                self.axes.set_ylim(ylim_update)
                self.axes.figure.canvas.draw_idle()
                self.xlim = self.axes.get_xlim()
                self.ylim = self.axes.get_ylim()
            
        else:
            # hover movements over image
            if 0 <= x < self.img3d.shape[2] and 0 <= y < self.img3d.shape[1]:

                if self.enable_painting and self.edit_mode:
                    if self.circle:
                        # update pen circle position
                        self.circle.center = x, y
                        # does not appear to be necessary since text update already
                        # triggers a redraw, but this would also trigger if no text
                        # update
                        self.circle.stale = True
                    else:
                        # generate new circle if not yet present
                        self.circle = patches.Circle(
                            (x, y), radius=self.radius, linestyle=":",
                            fill=False, edgecolor="w")
                        self.axes.add_patch(self.circle)

                # re-translate downsampled coordinates to original space
                coord = self.translate_coord([self.coord[0], y, x], up=True)
                if event.button == 1:
                    if (self.enable_painting and self.edit_mode
                            and self.intensity is not None):
                        # click in editing mode to overwrite images with pen
                        # of the current radius using chosen intensity for the
                        # underlying and displayed images
                        if self._ax_img_labels is not None:
                            # edit displayed image
                            img = self._ax_img_labels.get_array()
                            rr, cc = draw.disk(
                                (y, x), self.radius, shape=img.shape)
                            img[rr, cc] = self.intensity_shown

                            # edit underlying labels image
                            rr, cc = draw.disk(
                                (coord[1], coord[2]),
                                self.radius * self._downsample[0],
                                shape=self.img3d_labels[self.coord[0]].shape)
                            self.img3d_labels[
                                self.coord[0], rr, cc] = self.intensity
                            print("changed intensity at x,y,z = {},{},{} to {}"
                                  .format(*coord[::-1], self.intensity))
                            if self.fn_refresh_images is None:
                                self.refresh_img3d_labels()
                            else:
                                self.fn_refresh_images(self, True)
                            self.edited = True
                            self._editing = True
                    else:
                        # mouse left-click drag (separate from the initial
                        # press) moves crosshairs
                        self.coord = coord
                        if self.fn_update_coords:
                            self.fn_update_coords(self.coord, self.plane)
                
                if (self.img3d_labels is not None and
                        config.labels_ref is not None and
                        config.labels_ref.ref_lookup):
                    # show atlas label description
                    name = ""
                    if self._show_labels:
                        # get name from labels reference corresponding to
                        # labels image value under mouse pointer
                        atlas_label = ontology.get_label(
                            coord, self.img3d_labels,
                            config.labels_ref.ref_lookup, self.scaling,
                            self.labels_level)
                        if atlas_label is not None:
                            # extract name and ID from label dict
                            name = "{} ({})".format(
                                ontology.get_label_name(atlas_label),
                                ontology.get_label_item(
                                    atlas_label, config.ABAKeys.ABA_ID.value))
                            _logger.debug("Found label: %s", name)
                            
                            # minimize chance of text overflowing out of axes by
                            # word-wrapping and switching sides at midlines
                            name = "\n".join(textwrap.wrap(name, 30))
                            if x > self.img3d_labels.shape[2] / 2:
                                alignment_x = "right"
                                label_x = x - 20
                            else:
                                alignment_x = "left"
                                label_x = x + 20
                            if y > self.img3d_labels.shape[1] / 2:
                                alignment_y = "top"
                                label_y = y - 20
                            else:
                                alignment_y = "bottom"
                                label_y = y + 20
                            self.region_label.set_horizontalalignment(alignment_x)
                            self.region_label.set_verticalalignment(alignment_y)
                            self.region_label.set_position((label_x, label_y))
                    self.region_label.set_text(name)

            # need explicit draw call for figs embedded in TraitsUI
            self.axes.figure.canvas.draw_idle()

        if self.fn_status_bar:
            self.fn_status_bar(self.axes.format_coord.get_msg(event))
        self.last_loc = loc
        self.last_loc_data = loc_data
    
    def on_release(self, event):
        """Respond to mouse button release events.

        If labels were edited during the current mouse press, update
        plane interpolation values. Also reset any specified intensity value.
        
        Args:
            event: Key press event.
        """
        if self._editing:
            if self.interp_planes is not None:
                self.interp_planes.update_plane(
                    self.plane, self.coord[0], self.intensity)
            self._editing = False

        if self.intensity_spec is not None:
            # reset all plot editors' specified intensity to allow updating
            # with clicked intensities
            self.intensity_spec = None
            if self.fn_update_intensity:
                self.fn_update_intensity(None)
    
    def on_key_press(self, event):
        """Change pen radius with bracket ([/]) buttons.
        
        The "ctrl" modifier will have the increment.
        
        Args:
            event: Key press event.
        """
        rad_orig = self.radius
        increment = 0.5 if "ctrl" in event.key else 1
        if "[" in event.key and self.radius > 1:
            self.radius -= increment
        elif "]" in event.key:
            self.radius += increment
        #print("radius: {}".format(self.radius))
        if rad_orig != self.radius and self.circle:
            self.circle.radius = self.radius
