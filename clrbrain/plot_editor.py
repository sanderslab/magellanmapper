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
    ALPHA_DEFAULT = 0.7
    
    def __init__(self, axes, img3d, img3d_labels, cmap_labels, plane, 
                 alpha_slider, alpha_reset_btn, aspect, origin, 
                 fn_update_coords, fn_refresh_images, scaling):
        self.axes = axes
        self.img3d = img3d
        self.img3d_labels = img3d_labels
        self.cmap_labels = cmap_labels
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
        self.connected = False
        self.hline = None
        self.vline = None
        self.coord = None
        self.plot_axes = (
            self.axes, self.alpha_slider.ax, self.alpha_reset_btn.ax)
    
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
        # show main image
        img2d = self.img3d[self.coord[0]]
        plot_2d.imshow_multichannel(
            self.axes, img2d, 0, 
            config.process_settings["channel_colors"], self.aspect, 1, origin=self.origin, 
            interpolation="none")
        
        # show labels image
        img2d = self.img3d_labels[self.coord[0]]
        label_ax_img = plot_2d.imshow_multichannel(
            self.axes, img2d, 0, [self.cmap_labels], self.aspect, self.alpha, origin=self.origin, 
            interpolation="none")
        self.axes.format_coord = plot_2d.PixelDisplay(img2d)
        plot_2d._set_overview_title(self.axes, self.plane, self.coord[0])
        self.ax_img = label_ax_img[0]
        
        if not self.connected:
            # connect once get AxesImage
            self.connect()
        self.region_label = self.axes.text(0, 0, "", color="w")
        self.circle = None
    
    def scroll_overview(self, event):
        if event.inaxes != self.axes: return
        z_overview_new = plot_2d._scroll_plane(event, self.coord[0], self.img3d.shape[0])
        if z_overview_new != self.coord[0]:
            # move only if step registered and changing position
            coord = list(self.coord)
            coord[0] = z_overview_new
            self.fn_update_coords(coord, self.plane)
    
    def setup_animation(self):
        """Store the pixel buffer in background to prep for animations.
        """
        PlotEditor.lock = self
        canvas = self.ax_img.figure.canvas
        self.ax_img.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.axes.bbox)
        canvas.blit(self.axes.bbox)
        print("setup animation")
    
    def update_animation(self):
        """Update animations used cached artists and background and only in 
        the axes area for improved efficiency.
        """
        canvas = self.ax_img.figure.canvas
        canvas.restore_region(self.background)
        axes = self.axes
        axes.draw_artist(self.ax_img)
        # WORKAROUND: axes brightens for some reason increased during update 
        # unless alpha changed to any value
        axes.set_alpha(1.0)
        if self.circle:
            axes.draw_artist(self.circle)
        canvas.blit(self.axes.bbox)
    
    def update_image(self):
        """Replace current image with underlying plane's data.
        """
        self.ax_img.set_data(self.img3d_labels[self.coord[0]])
    
    def reset_animation(self):
        """Turn off animations and redraw entire figure.
        """
        PlotEditor.lock = None
        self.ax_img.set_animated(False)
        self.background = None
        self.intensity = None
        self.ax_img.figure.canvas.draw()
        print("reset animation")
    
    def alpha_updater(self, alpha):
        self.alpha = alpha
        self.ax_img.set_alpha(self.alpha)
        print("set image alpha to {}".format(self.alpha))
        # assume animation set up during axes enter
        self.update_animation()
    
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
        self.circle.radius += 1
        self.update_animation()
    
    def on_axes_enter(self, event):
        if event.inaxes not in self.plot_axes: return
        self.setup_animation()
    
    def on_axes_exit(self, event):
        if event.inaxes not in self.plot_axes: return
        if self.circle:
            self.circle.remove()
            self.circle = None
        self.reset_animation()
    
    def on_motion(self, event):
        """Move the editing pen's circle and draw with the chosen intensity 
        value if set.
        """
        if event.inaxes != self.axes: return
        
        # get mouse position and return if no change from last pixel coord
        x = int(event.xdata)
        y = int(event.ydata)
        if self.last_loc is not None and self.last_loc == (x, y):
            #print("didn't move")
            return
        self.last_loc = (x, y)
        
        if self.circle:
            # update pen circle position
            self.circle.center = x, y
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
        
        self.update_animation()
        
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
