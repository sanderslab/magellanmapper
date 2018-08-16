# 2D plot editor
# Author: David Young, 2018
"""2D plot editor.
"""

import matplotlib.patches as patches
from skimage import draw

from clrbrain import config
from clrbrain import register

class PlotEditor:
    ALPHA_DEFAULT = 0.7
    
    def __init__(self, img3d, alpha_slider, alpha_reset_btn):
        self.img3d = img3d
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
        self.alpha = self.ALPHA_DEFAULT
        self.alpha_slider = alpha_slider
        self.alpha_reset_btn = alpha_reset_btn
        self.connected = False
    
    def set_plane(self, ax_img, plane_n, plane=None):
        self.ax_img = ax_img
        self.plane_n = plane_n
        self.plane = plane
        if not self.connected:
            # connect once get AxesImage
            self.connect()
        self.region_label = self.ax_img.axes.text(0, 0, "", color="w")
        self.circle = None
    
    def connect(self):
        """Connect events to functions.
        """
        canvas = self.ax_img.figure.canvas
        self.cidpress = canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = canvas.mpl_connect(
            "button_release_event", self.on_release)
        self.cidmotion = canvas.mpl_connect(
            "motion_notify_event", self.on_motion)
        self.cidenter = canvas.mpl_connect(
            "axes_enter_event", self.on_axes_enter)
        self.cidleave = canvas.mpl_connect(
            "axes_leave_event", self.on_axes_exit)
        self.cidkeypress = canvas.mpl_connect(
            "key_press_event", self.on_key_press)
        self.connected = True
    
    def setup_animation(self):
        """Store the pixel buffer in background to prep for animations.
        """
        PlotEditor.lock = self
        canvas = self.ax_img.figure.canvas
        self.ax_img.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.ax_img.axes.bbox)
        canvas.blit(self.ax_img.axes.bbox)
        print("setup animation")
    
    def update_animation(self):
        """Update animations used cached artists and background and only in 
        the axes area for improved efficiency.
        """
        canvas = self.ax_img.figure.canvas
        canvas.restore_region(self.background)
        axes = self.ax_img.axes
        axes.draw_artist(self.ax_img)
        # WORKAROUND: axes brightens for some reason increased during update 
        # unless alpha changed to any value
        axes.set_alpha(1.0)
        if self.circle:
            axes.draw_artist(self.circle)
        canvas.blit(self.ax_img.axes.bbox)
    
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
        self.update_animation()
    
    def alpha_reset(self, event):
        #alpha_updater(alpha_updater)
        print("resetting slider")
        self.alpha_slider.reset()
    
    #def update_alpha(self, alpha):
    
    def on_press(self, event):
        """Initiate drag events with Shift- or Alt-click inside a circle.
        
        Shift-click to move a circle, and Alt-click to resize a circle's radius.
        """
        if event.inaxes != self.ax_img.axes: return
        x = int(event.xdata)
        y = int(event.ydata)
        self.intensity = self.img3d[self.plane_n, y, x]
        print("got intensity {} at x,y,z = {},{},{}"
              .format(self.intensity, x, y, self.plane_n))
        self.circle.radius += 1
        self.update_animation()
    
    def on_axes_enter(self, event):
        self.setup_animation()
    
    def on_axes_exit(self, event):
        if self.circle:
            self.circle.remove()
            self.circle = None
        self.reset_animation()
    
    def on_motion(self, event):
        """Move the editing pen's circle and draw with the chosen intensity 
        value if set.
        """
        if event.inaxes != self.ax_img.axes: return
        
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
            self.ax_img.axes.add_patch(self.circle)
        
        if self.intensity is not None:
            # use the chosen intensity value to overwrite the image with 
            # a pen of the chosen radius
            rr, cc = draw.circle(
                y, x, self.radius, self.img3d[self.plane_n].shape)
            self.img3d[self.plane_n, rr, cc] = self.intensity
            print("changed intensity to {} at x,y,z = {},{},{}"
                  .format(self.intensity, x, y, self.plane_n))
            self.ax_img.set_data(self.img3d[self.plane_n])
        
        self.update_animation()
        
        # show atlas label name
        atlas_label = register.get_label(
            (self.plane_n, y, x), config.labels_img, config.labels_ref_lookup, 
            config.labels_scaling)
        name = ""
        if atlas_label is not None:
            name = register.get_label_name(atlas_label)
        self.region_label.set_text(name)
        # minimize chance of text overflowing out of axes by switching 
        # alignment at midline horizontally
        if x > self.img3d.shape[2] / 2:
            alignment = "right"
            label_x = x - 10
        else:
            alignment = "left"
            label_x = x + 10
        self.region_label.set_horizontalalignment(alignment)
        self.region_label.set_position((label_x, y - 10))
    
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
