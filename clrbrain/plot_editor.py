# 2D plot editor
# Author: David Young, 2018
"""2D plot editor.
"""


class PlotEditor:
    def __init__(self, img3d):
        self.img3d = img3d
        self.intensity = None
        self.cidpress = None
        self.cidrelease = None
        self.cidmotion = None
    
    def set_plane(self, ax_img, plane_n, plane=None):
        self.ax_img = ax_img
        self.plane_n = plane_n
        self.plane = plane
        self.connect()
    
    def connect(self):
        """Connect events to functions.
        """
        self.cidpress = self.ax_img.figure.canvas.mpl_connect(
            "button_press_event", self.on_press)
        self.cidrelease = self.ax_img.figure.canvas.mpl_connect(
            "button_release_event", self.on_release)
        self.cidmotion = self.ax_img.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion)
        #print("connected circle at {}".format(self.circle.center))
    
    def on_press(self, event):
        """Initiate drag events with Shift- or Alt-click inside a circle.
        
        Shift-click to move a circle, and Alt-click to resize a circle's radius.
        """
        x = int(event.xdata)
        y = int(event.ydata)
        self.intensity = self.img3d[self.plane_n, y, x]
        print("got intensity {} at x,y,z = {},{},{}"
              .format(self.intensity, x, y, self.plane_n))
        PlotEditor.lock = self
        
        # draw everywhere except the circle itself, store the pixel buffer 
        # in background, and draw the circle
        canvas = self.ax_img.figure.canvas
        ax = self.ax_img.axes
        self.ax_img.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.ax_img.axes.bbox)
        #ax.draw_artist(self.ax_img)
        canvas.blit(ax.bbox)
    
    def on_motion(self, event):
        """Move the circle if the drag event has been initiated.
        """
        if event.inaxes != self.ax_img.axes or self.intensity is None: return
        x = int(event.xdata)
        y = int(event.ydata)
        self.img3d[self.plane_n, y, x] = self.intensity
        print("changed intensity to {} at x,y,z = {},{},{}"
              .format(self.intensity, x, y, self.plane_n))

        # restore the saved background and redraw the circle at its new position
        canvas = self.ax_img.figure.canvas
        ax = self.ax_img.axes
        canvas.restore_region(self.background)
        self.ax_img.set_data(self.img3d[self.plane_n])
        ax.draw_artist(self.ax_img)
        canvas.blit(ax.bbox)
    
    def on_release(self, event):
        """Finalize the circle and segment's position after a drag event
        is completed with a button release.
        """
        # turn off animation property, reset background
        PlotEditor.lock = None
        self.ax_img.set_animated(False)
        self.background = None
        self.intensity = None
        self.ax_img.figure.canvas.draw()
        print("released!")
    
    def disconnect(self):
        """Disconnect event listeners.
        """
        if self.cidpress:
            self.ax_img.figure.canvas.mpl_disconnect(self.cidpress)
        if self.cidrelease:
            self.ax_img.figure.canvas.mpl_disconnect(self.cidrelease)
        if self.cidmotion:
            self.ax_img.figure.canvas.mpl_disconnect(self.cidmotion)
