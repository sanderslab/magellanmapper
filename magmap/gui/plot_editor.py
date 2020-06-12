# 2D overlaid plot editor
# Author: David Young, 2018, 2020
"""Editor for 2D plot with overlaid planes.

Integrates with :class:``atlas_editor.AtlasEditor`` for synchronized 3D 
view of orthogonal planes.
"""

import textwrap

from matplotlib import patches
import numpy as np
from skimage import draw

from magmap.gui import pixel_display
from magmap.io import libmag
from magmap.settings import config
from magmap.atlas import ontology
from magmap.plot import plot_support


class PlotAxImg:
    """Axes image storage class to contain additional information for
    display such as brightness and contrast.

    Attributes:
        ax_img (List[List[:obj:`matplotlib.image.AxesImage`]]): Nested list
            of displayed axes image in the format:
            ``[[AxesImage0_ch0, AxesImage0_ch1, ...], ...]``.
        brightness (float): Brightness addend; defaults to 0.0.
        contrast (float): Contrast factor; defaults to 1.0.
        img (:obj:`np.ndarray`): The original underlying image data,
            copied to allow adjusting the array in ``ax_img`` while
            retaining the original data.

    """
    def __init__(self, ax_img):
        self.ax_img = ax_img
        self.brightness = 0.0
        self.contrast = 1.0
        self.img = np.copy(self.ax_img.get_array())


class PlotEditor:
    """Show a scrollable, editable plot of sequential planes in a 3D image.

    Attributes:
        intensity (int): Chosen intensity value of ``img3d_labels``.
        intensity_spec (int): Intensity value specified directly
            rather than chosen from the labels image.
        intensity_shown (int): Displayed intensity in the
            :obj:`matplotlib.AxesImage` corresponding to ``img3d_labels``.

    """
    ALPHA_DEFAULT = 0.5
    _KEY_MODIFIERS = ("shift", "alt", "control")
    
    def __init__(self, axes, img3d, img3d_labels, cmap_labels, plane, 
                 aspect, origin, fn_update_coords, fn_refresh_images, scaling, 
                 plane_slider, img3d_borders=None, cmap_borders=None, 
                 fn_show_label_3d=None, interp_planes=None,
                 fn_update_intensity=None, max_size=None, fn_status_bar=None):
        """Initialize the plot editor.
        
        Args:
            axes (:obj:`matplotlib.Axes`): Containing subplot axes.
            img3d (:obj:`np.ndarray`): Main 3D image.
            img3d_labels (:obj:`np.ndarray`): Labels 3D image.
            cmap_labels (:obj:`matplotlib.colors.ListedColormap`): Labels 
                colormap, generally a :obj:`colormaps.DiscreteColormap`.
            plane (str): One of :attr:`config.PLANE` specifying the orthogonal 
                plane to view.
            aspect (float): Aspect ratio.
            origin (str): Planar orientation, usually either "lower" or None.
            fn_update_coords (function): Callback when updating coordinates.
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

        """
        self.axes = axes
        self.img3d = img3d
        self.img3d_labels = img3d_labels
        self.cmap_labels = cmap_labels
        self.plane = plane
        self.alpha = self.ALPHA_DEFAULT
        self.aspect = aspect
        self.origin = origin
        self.fn_update_coords = fn_update_coords
        self.fn_refresh_images = fn_refresh_images
        self.scaling = config.labels_scaling if scaling is None else scaling
        self.plane_slider = plane_slider
        self.plane_slider.on_changed(self.update_plane_slider)
        self.img3d_borders = img3d_borders
        self.cmap_borders = cmap_borders
        self.fn_show_label_3d = fn_show_label_3d
        self.interp_planes = interp_planes
        self.fn_update_intensity = fn_update_intensity
        self._downsample = 1
        if max_size:
            # calculate downsampling factor based on max size
            img = self.img3d[0]
            self._downsample = max(img.shape[:2]) // max_size
            if self._downsample < 1:
                self._downsample = 1
            print("downsampling factor for plane {}: {}"
                  .format(self.plane, self._downsample))
        self.fn_status_bar = fn_status_bar
        
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

        self._plot_ax_imgs = None
        self._ax_img_labels = None  # displayed labels image
        # track label editing during mouse click/movement for plane interp
        self._editing = False
    
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

    def update_coord(self, coord):
        update_overview = self.coord is None or coord[0] != self.coord[0]
        self.coord = coord
        if update_overview:
            self.show_overview()
        self.draw_crosslines()

    def translate_coord(self, coord, up=False):
        """Translate coordinate based on downsampling factor.

        Coordinates sent to and received from the Atlas Editor are assumed to
        be in the original image space.

        Args:
            coord (List[int]): Coordinates in z,y,x.
            up (bool): True to upsample; defaults to False to adjust
                coordinates for downsampled images.

        Returns:
            List[int]: The translated coordinates.

        """
        coord_tr = np.copy(coord)
        if up:
            coord_tr[1:] = np.multiply(coord_tr[1:], self._downsample)
        else:
            coord_tr[1:] = np.divide(coord_tr[1:], self._downsample)
        coord_tr = list(coord_tr.astype(np.int))
        # print("translated from {} to {}".format(coord, coord_tr))
        return coord_tr

    def draw_crosslines(self):
        """Draw crosshairs depicting the x and y values in orthogonal viewers.
        """
        # translate coordinate down for any downsampling
        coord = self.translate_coord(self.coord)
        if self.hline is None:
            # draw new crosshairs
            self.hline = self.axes.axhline(coord[1], linestyle=":")
            self.vline = self.axes.axvline(coord[2], linestyle=":")
        else:
            # update positions of current crosshairs
            self.hline.set_ydata(coord[1])
            self.vline.set_xdata(coord[2])
    
    def show_overview(self):
        """Show the main 2D plane, taken as a z-plane."""
        # assume colorbar already shown if set and image previously displayed
        colorbar = (config.roi_profile["colorbar"]
                    and len(self.axes.images) < 1)
        self.axes.clear()
        self.hline = None
        self.vline = None
        
        # prep main image in grayscale and labels with discrete colormap
        imgs2d = [self.img3d[self.coord[0]]]
        cmaps = [config.cmaps]
        alphas = [config.alphas[0]]
        
        if self.img3d_labels is not None:
            imgs2d.append(self.img3d_labels[self.coord[0]])
            cmaps.append(self.cmap_labels)
            alphas.append(self.alpha)
        
        if self.img3d_borders is not None:
            # prep borders image, which may have an extra channels 
            # dimension for multiple sets of borders
            img2d = self.img3d_borders[self.coord[0]]
            channels = img2d.ndim if img2d.ndim >= 3 else 1
            for i, channel in enumerate(range(channels - 1, -1, -1)):
                # show first (original) borders image last so that its 
                # colormap values take precedence to highlight original bounds
                img_add = img2d[..., channel] if channels > 1 else img2d
                imgs2d.append(img_add)
                cmaps.append(self.cmap_borders[channel])
                alphas.append(libmag.get_if_within(config.alphas, 2 + i, 1))

        if self._downsample:
            # downsample images to reduce access time
            for i, img in enumerate(imgs2d):
                imgs2d[i] = img[::self._downsample, ::self._downsample]

        # overlay all images and set labels for footer value on mouseover;
        # if first time showing image, need to check for images with single
        # value since they fail to update on subsequent updates for unclear
        # reasons
        ax_imgs = plot_support.overlay_images(
            self.axes, self.aspect, self.origin, imgs2d, None, cmaps, alphas,
            check_single=(self._ax_img_labels is None))
        if colorbar:
            self.axes.figure.colorbar(ax_imgs[0][0], ax=self.axes)
        self.axes.format_coord = pixel_display.PixelDisplay(
            imgs2d, ax_imgs, self._downsample, cmap_labels=self.cmap_labels)
        self.plane_slider.set_val(self.coord[0])
        if len(ax_imgs) > 1: self._ax_img_labels = ax_imgs[1][0]
        self._plot_ax_imgs = [
            [PlotAxImg(img) for img in imgs] for imgs in ax_imgs]
        
        if self.xlim is not None and self.ylim is not None:
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)
        if not self.connected:
            # connect once get AxesImage
            self.connect()
        # text label with color for visibility on axes plus fig background
        self.region_label = self.axes.text(
            0, 0, "", color="k", bbox=dict(facecolor="xkcd:silver", alpha=0.5))
        self.circle = None
    
    def _update_overview(self, z_overview_new):
        if z_overview_new != self.coord[0]:
            # move only if step registered and changing position
            coord = list(self.coord)
            coord[0] = z_overview_new
            self.fn_update_coords(coord, self.plane)
    
    def scroll_overview(self, event):
        if event.inaxes != self.axes: return
        z_overview_new = plot_support.scroll_plane(
            event, self.coord[0], self.img3d.shape[0], 
            max_scroll=config.max_scroll)
        self._update_overview(z_overview_new)
    
    def update_plane_slider(self, val):
        self._update_overview(int(val))
    
    def refresh_img3d_labels(self):
        """Replace the displayed labels image with underlying plane's data.
        """
        if self._ax_img_labels is not None:
            # underlying plane may need to be translated to a linear
            # scale for the colormap before display
            self._ax_img_labels.set_data(self.cmap_labels.convert_img_labels(
                self.img3d_labels[self.coord[0]]))

    def get_img_display_settings(self, imgi, chl=None):
        """Get display settings for the given image.

        Args:
            imgi (int): Index of image.
            chl (int): Index of channel; defaults to None.

        Returns:
            float, flat, float, float, float: Vmin, vmax, brightness,
            contrast, and alpha.

        """
        if not self._plot_ax_imgs or len(self._plot_ax_imgs) <= imgi:
            return
        if chl is None:
            chl = 0
        plot_ax_img = self._plot_ax_imgs[imgi][chl]
        return (*plot_ax_img.ax_img.get_clim(), plot_ax_img.brightness,
                plot_ax_img.contrast, plot_ax_img.ax_img.get_alpha())

    def update_img_display(self, imgi, chl=None, minimum=np.nan,
                           maximum=np.nan, brightness=None, contrast=None,
                           alpha=None):
        """Update dislayed image settings.

        Args:
            imgi (int): Index of image.
            chl (int): Index of channel; defaults to None.
            minimum (float): Vmin; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            maximum (float): Vmax; can be None for auto setting; defaults
                to ``np.nan`` to ignore.
            brightness (float): Brightness addend; defaults to None.
            contrast (float): Contrast multiplier; defaults to None.
            alpha (float): Opacity value; defalts to None.

        """
        if not self._plot_ax_imgs or len(self._plot_ax_imgs) <= imgi:
            return
        if chl is None:
            chl = 0
        plot_ax_imgs = self._plot_ax_imgs[imgi]
        for i, plot_ax_img in enumerate(plot_ax_imgs):
            if i != chl: continue
            if minimum is not np.nan or maximum is not np.nan:
                # set vmin and vmax
                clim = [minimum, maximum]
                for j, (lim, ax_lim) in enumerate(zip(
                        clim, plot_ax_img.ax_img.get_clim())):
                    if lim is np.nan:
                        # default to using current value
                        clim[j] = ax_lim
                if None not in clim and clim[0] > clim[1]:
                    # ensure min is <= max
                    clim[0] = clim[1]
                plot_ax_img.ax_img.set_clim(clim)
            if brightness is not None or contrast is not None:
                data = plot_ax_img.ax_img.get_array()
                info = libmag.get_dtype_info(data)
                if brightness is not None:
                    # shift original image array by brightness
                    data[:] = np.clip(
                        plot_ax_img.img + brightness, info.min, info.max)
                if contrast is not None:
                    # stretch original image array by contrast
                    data[:] = np.clip(
                        plot_ax_img.img * contrast, info.min, info.max)
            if alpha is not None:
                # adjust opacity
                plot_ax_img.ax_img.set_alpha(alpha)
        self.axes.figure.canvas.draw_idle()

    def alpha_updater(self, alpha):
        self.alpha = alpha
        if self._ax_img_labels is not None:
            self._ax_img_labels.set_alpha(self.alpha)
        #print("set image alpha to {}".format(self.alpha))
    
    def on_press(self, event):
        """Pick intensities by mouse clicking on a given pixel.
        """
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
            #print("didn't move")
            return
        
        loc_data = (x, y)
        if event.button == 2 or (event.button == 1 and event.key == "shift"):
            # pan by middle-click or shift+click during mouseover
            
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
            # data itself moved, so update location aong with movement
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
                        (x, y), radius=self.radius, linestyle=":", fill=False, 
                        edgecolor="w")
                    self.axes.add_patch(self.circle)

                # re-translate downsampled coordinates to original space
                coord = self.translate_coord([self.coord[0], y, x], up=True)
                if event.button == 1:
                    if self.edit_mode and self.intensity is not None:
                        # click in editing mode to overwrite images with pen
                        # of the current radius using chosen intensity for the
                        # underlying and displayed images
                        if self._ax_img_labels is not None:
                            # edit displayed image
                            img = self._ax_img_labels.get_array()
                            rr, cc = draw.circle(y, x, self.radius, img.shape)
                            img[rr, cc] = self.intensity_shown

                            # edit underlying labels image
                            rr, cc = draw.circle(
                                coord[1], coord[2], self.radius * self._downsample,
                                self.img3d_labels[self.coord[0]].shape)
                            self.img3d_labels[
                                self.coord[0], rr, cc] = self.intensity
                            print("changed intensity at x,y,z = {},{},{} to {}"
                                  .format(*coord[::-1], self.intensity))
                            self.fn_refresh_images(self, True)
                            self.edited = True
                            self._editing = True
                    else:
                        # mouse left-click drag (separate from the initial
                        # press) moves crosshairs
                        self.coord = coord
                        self.fn_update_coords(self.coord, self.plane)
                
                if self.img3d_labels is not None:
                    # show atlas label name
                    atlas_label = ontology.get_label(
                        coord, self.img3d_labels, config.labels_ref_lookup, 
                        self.scaling)
                    name = ""
                    if atlas_label is not None:
                        # extract name and ID from label dict
                        name = "{} ({})".format(
                            ontology.get_label_name(atlas_label),
                            ontology.get_label_item(
                                atlas_label, config.ABAKeys.ABA_ID.value))

                    # minimize chance of text overflowing out of axes by
                    # word-wrapping and switching sides at midlines
                    name = "\n".join(textwrap.wrap(name, 30))
                    self.region_label.set_text(name)
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
