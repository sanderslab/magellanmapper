# 2D plot editor
# Author: David Young, 2018
"""2D plot editor.
"""

import matplotlib.patches as patches
from skimage import draw

from clrbrain import config
from clrbrain import register
from clrbrain import plot_2d

class PlotEditor:
    ALPHA_DEFAULT = 0.5
    
    def __init__(self, axes, img3d, img3d_labels, cmap_labels, norm, plane, 
                 alpha_slider, alpha_reset_btn, aspect, origin, 
                 fn_update_coords, fn_refresh_images, scaling):
        self.axes = axes
        self.img3d = img3d
        self.img3d_labels = img3d_labels
        self.cmap_labels = cmap_labels
        self.norm = norm
        self.plane = plane
        self.alpha = self.ALPHA_DEFAULT
        self.alpha_slider = alpha_slider
        self.alpha_reset_btn = alpha_reset_btn
        self.aspect = aspect
        self.origin = origin
        self.fn_update_coords = fn_update_coords
        self.fn_refresh_images = fn_refresh_images
        self.scaling = config.labels_scaling if scaling is None else scaling
        
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
        self.press_loc_data = None
        self.connected = False
        self.hline = None
        self.vline = None
        self.coord = None
        self.plot_axes = (
            self.axes, self.alpha_slider.ax, self.alpha_reset_btn.ax)
        self.xlim = None
        self.ylim = None
    
    def connect(self):
        """Connect events to functions.
        """
        canvas = self.ax_img.figure.canvas
        self.cidpress = canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = canvas.mpl_connect(
            "button_release_event", self.on_release)
        self.cidmotion = canvas.mpl_connect(
            "motion_notify_event", self.on_motion)
        self.cidkeypress = canvas.mpl_connect(
            "key_press_event", self.on_key_press)
        self.connected = True
    
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
        # show main image in grayscale
        img2d = self.img3d[self.coord[0]]
        plot_2d.imshow_multichannel(
            self.axes, img2d, 0, ["gray"], self.aspect, 1, origin=self.origin, 
            interpolation="none")
        
        # show labels image
        img2d = self.img3d_labels[self.coord[0]]
        label_ax_img = plot_2d.imshow_multichannel(
            self.axes, img2d, 0, [self.cmap_labels], self.aspect, self.alpha, 
            origin=self.origin, interpolation="none", norms=[self.norm])
        self.axes.format_coord = plot_2d.PixelDisplay(img2d)
        plot_2d._set_overview_title(self.axes, self.plane, self.coord[0])
        self.ax_img = label_ax_img[0]
        
        if self.xlim is not None and self.ylim is not None:
            self.axes.set_xlim(self.xlim)
            self.axes.set_ylim(self.ylim)
        if not self.connected:
            # connect once get AxesImage
            self.connect()
        # text label with color for visibility on axes plus fig background
        self.region_label = self.axes.text(0, 0, "", color="xkcd:silver")
        self.circle = None
    
    def scroll_overview(self, event):
        if event.inaxes != self.axes: return
        z_overview_new = plot_2d._scroll_plane(event, self.coord[0], self.img3d.shape[0])
        if z_overview_new != self.coord[0]:
            # move only if step registered and changing position
            coord = list(self.coord)
            coord[0] = z_overview_new
            self.fn_update_coords(coord, self.plane)
    
    def update_image(self):
        """Replace current image with underlying plane's data.
        """
        self.ax_img.set_data(self.img3d_labels[self.coord[0]])
    
    def alpha_updater(self, alpha):
        self.alpha = alpha
        self.ax_img.set_alpha(self.alpha)
        print("set image alpha to {}".format(self.alpha))
    
    def alpha_reset(self, event):
        print("resetting slider")
        self.alpha_slider.reset()
    
    def on_press(self, event):
        """Pick intensities by clicking on a given pixel.
        """
        if event.inaxes != self.axes: return
        x = int(event.xdata)
        y = int(event.ydata)
        self.intensity = self.img3d_labels[self.coord[0], y, x]
        print("got intensity {} at x,y,z = {},{},{}"
              .format(self.intensity, x, y, self.coord[0]))
        self.press_loc_data = (x, y)
        self.last_loc = (int(event.x), int(event.y))
    
    def on_axes_exit(self, event):
        if event.inaxes not in self.plot_axes: return
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
            print("didn't move")
            return
        
        if event.button == 3:
            # zooming by right-click while moving mouse up/down in y
            print("x: {}, y: {}, press_loc_data: {}, "
                  "last_loc: {}, loc: {}, event.x: {}, event.y: {}"
                  .format(x, y, self.press_loc_data, 
                  self.last_loc, loc, event.x, event.y))
            dy = y_fig - self.last_loc[1]
            xlim = self.axes.get_xlim()
            zoom_speed = dy * 0.001
            zoom_x = self.scaling[2] * zoom_speed
            self.axes.set_xlim(
                xlim[0] + (self.press_loc_data[0] - xlim[0]) * zoom_x, 
                xlim[1] + (self.press_loc_data[0] - xlim[1]) * zoom_x)
            ylim = self.axes.get_ylim()
            zoom_y = self.scaling[1] * zoom_speed
            self.axes.set_ylim(
                ylim[0] + (self.press_loc_data[1] - ylim[0]) * zoom_y, 
                ylim[1] + (self.press_loc_data[1] - ylim[1]) * zoom_y)
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
                    # triggers a redraw, but this would also trigger if no text update
                    self.circle.stale = True
                else:
                    # generate new circle if not yet present
                    self.circle = patches.Circle(
                        (x, y), radius=self.radius, linestyle=":", fill=False, 
                        edgecolor="w")
                    self.axes.add_patch(self.circle)
                
                if self.intensity is not None:
                    # use the chosen intensity value to overwrite the image with 
                    # a pen of the chosen radius
                    rr, cc = draw.circle(
                        y, x, self.radius, self.img3d_labels[self.coord[0]].shape)
                    self.img3d_labels[self.coord[0], rr, cc] = self.intensity
                    print("changed intensity to {} at x,y,z = {},{},{}"
                          .format(self.intensity, x, y, self.coord[0]))
                    self.ax_img.set_data(self.img3d_labels[self.coord[0]])
                    self.fn_refresh_images(self)
                
                # show atlas label name
                coord = [self.coord[0], y, x]
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
                
                if event.key == "shift":
                    self.coord = coord
                    self.fn_update_coords(self.coord, self.plane)
                
        self.last_loc = loc
    
    def on_release(self, event):
        """Reset the intensity value to prevent further edits.
        """
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
        elif event.key == "shift" and event.inaxes == self.axes:
            # "shift" to update crosshairs and corresponding planes
            self.coord[1:] = int(event.ydata), int(event.xdata)
            self.fn_update_coords(self.coord, self.plane)
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
