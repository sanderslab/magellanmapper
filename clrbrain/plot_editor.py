# 2D plot editor
# Author: David Young, 2018
"""2D plot editor.
"""

import matplotlib.patches as patches
from skimage import draw

from clrbrain import config
from clrbrain import register
from clrbrain import plot_support

class PlotEditor:
    ALPHA_DEFAULT = 0.5
    _KEY_MODIFIERS = ("shift", "alt", "control")
    
    def __init__(self, axes, img3d, img3d_labels, cmap_labels, plane, 
                 aspect, origin, fn_update_coords, fn_refresh_images, scaling, 
                 plane_slider, img3d_borders=None, cmap_borders=None, 
                 fn_show_label_3d=None, interp_planes=None):
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
        
        self.intensity = None
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
        """Connect events to functions.
        """
        canvas = self.axes.figure.canvas
        canvas.mpl_discconnect(self.cidpress)
        canvas.mpl_discconnect(self.cidrelease)
        canvas.mpl_discconnect(self.cidmotion)
        canvas.mpl_discconnect(self.cidkeypress)
        self.connected = False
    
    def update_coord(self, coord):
        update_overview = self.coord is None or coord[0] != self.coord[0]
        self.coord = coord
        if update_overview:
            self.show_overview()
        self.draw_crosslines()
    
    def draw_crosslines(self):
        if self.hline is None:
            self.hline = self.axes.axhline(self.coord[1], linestyle=":")
            self.vline = self.axes.axvline(self.coord[2], linestyle=":")
        else:
            self.hline.set_ydata(self.coord[1])
            self.vline.set_xdata(self.coord[2])
    
    def show_overview(self):
        self.axes.clear()
        self.hline = None
        self.vline = None
        # prep main image in grayscale and labels with discrete colormap
        imgs2d = [self.img3d[self.coord[0]], self.img3d_labels[self.coord[0]]]
        cmaps = [config.process_settings["channel_colors"], self.cmap_labels]
        alphas = [1, self.alpha]
        if self.img3d_borders is not None:
            # show borders image
            img2d = self.img3d_borders[self.coord[0]]
            for channel in range(img2d.shape[-1] - 1, -1, -1):
                # show first (original) borders image last so that its 
                # colormap values take precedence to highlight original bounds
                imgs2d.append(img2d[..., channel])
                cmaps.append(self.cmap_borders[channel])
                alphas.append(1)
        
        # overlay all images and set labels for footer value on mouseover
        ax_imgs = plot_support.overlay_images(
            self.axes, self.aspect, self.origin, imgs2d, 0, cmaps, alphas)
        self.axes.format_coord = PixelDisplay(imgs2d[1])
        self.plane_slider.set_val(self.coord[0])
        self.ax_img = ax_imgs[1][0]
        
        if self.xlim is not None and self.ylim is not None:
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)
        if not self.connected:
            # connect once get AxesImage
            self.connect()
        # text label with color for visibility on axes plus fig background
        self.region_label = self.axes.text(0, 0, "", color="xkcd:silver")
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
        """Replace current image with underlying plane's data.
        """
        self.ax_img.set_data(self.img3d_labels[self.coord[0]])
    
    def alpha_updater(self, alpha):
        self.alpha = alpha
        self.ax_img.set_alpha(self.alpha)
        print("set image alpha to {}".format(self.alpha))
    
    def on_press(self, event):
        """Pick intensities by mouse clicking on a given pixel.
        """
        if event.inaxes != self.axes: return
        x = int(event.xdata)
        y = int(event.ydata)
        self.press_loc_data = (x, y)
        self.last_loc_data = tuple(self.press_loc_data)
        self.last_loc = (int(event.x), int(event.y))
        
        if event.button == 1:
            if event.key not in self._KEY_MODIFIERS:
                # click without modifiers to update crosshairs and 
                # corresponding planes
                self.coord[1:] = y, x
                self.fn_update_coords(self.coord, self.plane)
            elif event.key == "alt":
               # get intensity under cursor in prep to paint it
               self.intensity = self.img3d_labels[self.coord[0], y, x]
               print("got intensity {} at x,y,z = {},{},{}"
                     .format(self.intensity, x, y, self.coord[0]))
            
            if event.key == "3" and self.fn_show_label_3d is not None:
                # extract label ID and display in 3D viewer
                self.fn_show_label_3d(self.img3d_labels[tuple(self.coord)])
    
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
            zoom_x = self.scaling[2] * zoom_speed
            xlim = self.axes.get_xlim()
            xlim_update = (
                xlim[0] + (self.press_loc_data[0] - xlim[0]) * zoom_x, 
                xlim[1] + (self.press_loc_data[0] - xlim[1]) * zoom_x)
            zoom_y = self.scaling[1] * zoom_speed
            ylim = self.axes.get_ylim()
            ylim_update = (
                ylim[0] + (self.press_loc_data[1] - ylim[0]) * zoom_y, 
                ylim[1] + (self.press_loc_data[1] - ylim[1]) * zoom_y)
            
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
            if (x >= 0 and y >= 0 and x < self.img3d.shape[2] 
                and y < self.img3d.shape[1]):
                
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
                
                coord = [self.coord[0], y, x]
                if event.button == 1:
                    if event.key == "alt" and self.intensity is not None:
                        # alt+click to use the chosen intensity value to 
                        # overwrite the image with a pen of the chosen radius
                        rr, cc = draw.circle(
                            y, x, self.radius, 
                            self.img3d_labels[self.coord[0]].shape)
                        self.img3d_labels[
                            self.coord[0], rr, cc] = self.intensity
                        print("changed intensity to {} at x,y,z = {},{},{}"
                              .format(self.intensity, x, y, self.coord[0]))
                        self.ax_img.set_data(self.img3d_labels[self.coord[0]])
                        self.fn_refresh_images(self)
                    else:
                        # click and mouseover otherwise moves crosshairs
                        self.coord = coord
                        self.fn_update_coords(self.coord, self.plane)
                
                # show atlas label name
                atlas_label = register.get_label(
                    coord, self.img3d_labels, config.labels_ref_lookup, 
                    self.scaling)
                name = ""
                if atlas_label is not None:
                    name = register.get_label_name(atlas_label)
                self.region_label.set_text(name)
                # minimize chance of text overflowing out of axes by switching 
                # alignment at midline horizontally
                if x > self.img3d_labels.shape[2] / 2:
                    alignment = "right"
                    label_x = x - 10
                else:
                    alignment = "left"
                    label_x = x + 10
                self.region_label.set_horizontalalignment(alignment)
                self.region_label.set_position((label_x, y - 10))
                
                
        self.last_loc = loc
        self.last_loc_data = loc_data
    
    def on_release(self, event):
        """On mouse click release, reset the intensity value to prevent 
        further edits and update plane interpolation values if intensity 
        value was set.
        """
        if self.intensity is not None:
            if self.interp_planes is not None:
                self.interp_planes.update_plane(
                    self.plane, self.coord[0], self.intensity)
            self.intensity = None
        print("released!")
    
    def on_key_press(self, event):
        """Change pen radius with bracket ([/]) buttons.
        """
        rad_orig = self.radius
        if event.key == "[" and self.radius > 1:
            self.radius -= 1
        elif event.key == "]":
            self.radius += 1
        #print("radius: {}".format(self.radius))
        if rad_orig != self.radius and self.circle:
            self.circle.radius = self.radius
    
    def disconnect(self):
        """Disconnect event listeners.
        """
        self.circle = None
        listeners = [
            self.cidpress, self.cidrelease, self.cidmotion, self.cidenter, 
            self.cidleave, self.cidkeypress]
        for listener in listeners:
            if listener:
                self.ax_img.figure.canvas.mpl_disconnect(listener)

class PixelDisplay(object):
    def __init__(self, img):
        self.img = img
    def __call__(self, x, y):
        if x < 0 or y < 0 or x >= self.img.shape[1] or y >= self.img.shape[0]:
            z = "n/a"
        else:
            z = self.img[int(y), int(x)]
        return "x={:.01f}, y={:.01f}, z={}".format(x, y, z)
