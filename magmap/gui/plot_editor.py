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
        self.ax_img = None  # displayed labels image
        self.edited = False  # True if labels image was edited
        self.edit_mode = False  # True to edit with mouse motion
        self.region_label = None
        
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
            if listener and self.ax_img is not None:
                self.ax_img.figure.canvas.mpl_disconnect(listener)
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
            check_single=(self.ax_img is None))
        if colorbar:
            self.axes.figure.colorbar(ax_imgs[0][0], ax=self.axes)
        self.axes.format_coord = pixel_display.PixelDisplay(
            imgs2d, ax_imgs, self._downsample, cmap_labels=self.cmap_labels)
        self.plane_slider.set_val(self.coord[0])
        if len(ax_imgs) > 1: self.ax_img = ax_imgs[1][0]
        
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
    
    def update_image(self):
        """Replace the displayed labels image with underlying plane's data.
        """
        if self.ax_img is not None:
            # underlying plane may need to be translated to a linear
            # scale for the colormap before display
            self.ax_img.set_data(self.cmap_labels.convert_img_labels(
                self.img3d_labels[self.coord[0]]))
    
    def alpha_updater(self, alpha):
        self.alpha = alpha
        if self.ax_img is not None:
            self.ax_img.set_alpha(self.alpha)
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
                    self.intensity_shown = self.ax_img.get_array()[y, x]
                    print("got intensity {} at x,y,z = {},{},{}"
                          .format(self.intensity, *coord[::-1]))
                    if self.fn_update_intensity:
                        # trigger text box update
                        self.fn_update_intensity(self.intensity)
                else:
                    # use user-specified intensity value, resetting it
                    # afterward to allow updating with clicked intensities
                    print("using specified intensity of", self.intensity_spec)
                    self.intensity = self.intensity_spec
                    self.intensity_shown = self.cmap_labels.convert_img_labels(
                        self.intensity)
                    self.intensity_spec = None
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
                        if self.ax_img is not None:
                            # edit displayed image
                            img = self.ax_img.get_array()
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
                    # word-wrapping and switching sides at vertical midline
                    name = "\n".join(textwrap.wrap(name, 30))
                    self.region_label.set_text(name)
                    if x > self.img3d_labels.shape[2] / 2:
                        alignment = "right"
                        label_x = x - 20
                    else:
                        alignment = "left"
                        label_x = x + 20
                    self.region_label.set_horizontalalignment(alignment)
                    self.region_label.set_position((label_x, y - 20))
            # need explicit draw call for figs embedded in TraitsUI
            self.axes.figure.canvas.draw_idle()

        if self.fn_status_bar:
            self.fn_status_bar(self.axes.format_coord.get_msg(event))
        self.last_loc = loc
        self.last_loc_data = loc_data
    
    def on_release(self, event):
        """If labels were edited during the current mouse press, update 
        plane interpolation values.
        
        Args:
            event: Key press event.
        """
        if self._editing:
            if self.interp_planes is not None:
                self.interp_planes.update_plane(
                    self.plane, self.coord[0], self.intensity)
            self._editing = False
    
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
