# 2D plots from stacks of imaging data
# Author: David Young, 2017
"""Plots 2D views through multiple levels of a 3D stack for
comparison with 3D visualization.

Attributes:
    colormap_2d: The Matplotlib colormap for 2D plots.
    savefig: Extension of the file in which to automatically save the
        window as a figure (eg "pdf"). If None, figures will not be
        automatically saved.
    verify: If true, verification mode is turned on, which for now
        simply turns on interior borders as the picker remains on
        by default.
    padding: Padding in pixels (x, y), or planes (z) in which to show
        extra segments.
"""

import copy
import os
import math
from time import time

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH
from skimage import exposure
from skimage import img_as_float
from scipy import stats

from clrbrain import detector
from clrbrain import config
from clrbrain import lib_clrbrain

colormap_2d = cm.inferno
CMAP_GRBK = LinearSegmentedColormap.from_list("Green_black", ['black', 'green'])
#colormap_2d = cm.gray
savefig = None
verify = False
# TODO: may want to base on scaling factor instead
padding = (5, 5, 3) # human (x, y, z) order

SEG_LINEWIDTH = 0.5
ZOOM_COLS = 9
Z_LEVELS = ("bottom", "middle", "top")
PLANE = ("xy", "xz", "yz")
plane = None
CIRCLES = ("Circles", "Repeat circles", "No circles", "Full annotation")
vmax_overview = 1.0
_DOWNSAMPLE_THRESH = 1000

# need to store DraggableCircles objects to prevent premature garbage collection
_draggable_circles = []
_circle_last_picked = []
_CUT = "cut"
_COPY = "copy"

segs_color_dict = {
    -1: "none",
    0: "r",
    1: "g",
    2: "y"
}

truth_color_dict = {
    -1: None,
    0: "m",
    1: "b"
}

class DraggableCircle:
    def __init__(self, circle, segment, fn_update_seg, color="none"):
        self.circle = circle
        self.circle.set_picker(5)
        self.facecolori = -1
        for key, val in segs_color_dict.items():
            if val == color:
                self.facecolori = key
        self.press = None
        self.segment = segment
        self.fn_update_seg = fn_update_seg
    
    def connect(self):
        """Connect events to functions.
        """
        self.cidpress = self.circle.figure.canvas.mpl_connect(
            "button_press_event", self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect(
            "button_release_event", self.on_release)
        self.cidmotion = self.circle.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion)
        self.cidpick = self.circle.figure.canvas.mpl_connect(
            "pick_event", self.on_pick)
        #print("connected circle at {}".format(self.circle.center))
    
    def remove_self(self):
        self.disconnect()
        self.circle.remove()
        #segi = self.get_vis_segments_index(self.segment)
        #self.vis_segments.remove(segi)
    
    def on_press(self, event):
        """Initiate drag events with Shift-click inside a circle.
        """
        if (event.key != "shift" and event.key != "alt" 
            or event.inaxes != self.circle.axes):
            return
        contains, attrd = self.circle.contains(event)
        if not contains: return
        print("pressed on {}".format(self.circle.center))
        x0, y0 = self.circle.center
        self.press = x0, y0, event.xdata, event.ydata
    
    def on_motion(self, event):
        """Move the circle if the drag event has been initiated.
        """
        if self.press is None: return
        if event.inaxes != self.circle.axes: return
        x0, y0, xpress, ypress = self.press
        dx = None
        dy = None
        if event.key == "shift":
            dx = event.xdata - xpress
            dy = event.ydata - ypress
            self.circle.center = x0 + dx, y0 + dy
        elif event.key == "alt":
            dx = abs(event.xdata - x0)
            dy = abs(event.ydata - y0)
            self.circle.radius = max([dx, dy])
        print("initial position: {}, {}; change thus far: {}, {}"
              .format(x0, y0, dx, dy))

        self.circle.figure.canvas.draw()
    
    def on_release(self, event):
        """Finalize the circle and segment's position after a drag event
        is completed with a button release.
        """
        if self.press is None: return
        print("released on {}".format(self.circle.center))
        print("segment moving from {}...".format(self.segment))
        seg_old = np.copy(self.segment)
        self.segment[1:3] += np.subtract(
            self.circle.center, self.press[0:2]).astype(np.int)[::-1]
        rad_sign = -1 if self.segment[3] < config.POS_THRESH else 1
        self.segment[3] = rad_sign * self.circle.radius
        print("...to {}".format(self.segment))
        self.fn_update_seg(self.segment, seg_old)
        self.press = None
        self.circle.figure.canvas.draw()
    
    def on_pick(self, event):
        """Select the verification flag with unmodified (no Ctrl of Shift)
        button press on a circle.
        """
        if (event.mouseevent.key == "control" 
            or event.mouseevent.key == "shift" 
            or event.mouseevent.key == "alt" 
            or event.artist != self.circle):
            return
        #print("color: {}".format(self.facecolori))
        if event.mouseevent.key == "x":
            # "cut" segment
            _circle_last_picked.append((self, _CUT))
            self.remove_self()
            print("cut seg: {}".format(self.segment))
        elif event.mouseevent.key == "c":
            # "copy" segment
            _circle_last_picked.append((self, _COPY))
            print("copied seg: {}".format(self.segment))
        elif event.mouseevent.key == "d":
            # delete segment
            self.remove_self()
            self.fn_update_seg(self.segment, remove=True)
            print("deleted seg: {}".format(self.segment))
        else:
            seg_old = np.copy(self.segment)
            i = self.facecolori + 1
            if i > max(segs_color_dict.keys()):
                if self.segment[3] < config.POS_THRESH:
                    _circle_last_picked.append((self, _CUT))
                    self.remove_self()
                i = -1
            self.circle.set_facecolor(segs_color_dict[i])
            self.facecolori = i
            self.segment[4] = i
            self.fn_update_seg(self.segment, seg_old)
            print("picked segment: {}".format(self.segment))

    def disconnect(self):
        """Disconnect event listeners.
        """
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)

def _get_radius(seg):
    """Gets the radius for a segments, defaulting to 5 if the segment's
    radius is close to 0.
    
    Args:
        seg: The segments, where seg[3] is the radius.
    
    Returns:
        The radius, defaulting to 0 if the given radius value is close 
        to 0 by numpy.allclose.
    """
    radius = seg[3]
    if radius < config.POS_THRESH:
        radius *= -1
    return radius

def _circle_collection(segments, edgecolor, facecolor, linewidth):
    """Draws a patch collection of circles for segments.
    
    Args:
        segments: Numpy array of segments, generally as an (n, 4)
            dimension array, where each segment is in (z, y, x, radius).
        edgecolor: Color of patch borders.
        facecolor: Color of patch interior.
        linewidth: Width of the border.
    
    Returns:
        The patch collection.
    """
    seg_patches = []
    for seg in segments:
        seg_patches.append(patches.Circle((seg[2], seg[1]), radius=_get_radius(seg)))
    collection = PatchCollection(seg_patches)
    collection.set_edgecolor(edgecolor)
    collection.set_facecolor(facecolor)
    collection.set_linewidth(linewidth)
    return collection

def _plot_circle(ax, segment, edgecolor, linewidth, linestyle, fn_update_seg):
    """Draw a DraggableCircle from the given segment.
    
    Args:
        segment: Numpy array of segments, generally as an (n, 4)
            dimension array, where each segment is in (z, y, x, radius).
        edgecolor: Color of patch borders.
        facecolor: Color of patch interior.
        fn_update_seg: Function to call from DraggableCircle.
    
    Returns:
        The DraggableCircle object.
    """
    facecolor = segs_color_dict[segment[4]]
    circle = patches.Circle(
        (segment[2], segment[1]), radius=_get_radius(segment), 
        edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth, 
        linestyle=linestyle)
    ax.add_patch(circle)
    draggable_circle = DraggableCircle(
        circle, segment, fn_update_seg, facecolor)
    draggable_circle.connect()
    _draggable_circles.append(draggable_circle)
    return draggable_circle

def add_scale_bar(ax):
    """Adds a scale bar to the plot.
    
    Uses the x resolution value and assumes that it is in microns per pixel.
    
    Args:
        ax: The plot that will show the bar.
    """
    scale_bar = ScaleBar(detector.resolutions[0][2], u'\u00b5m', SI_LENGTH, 
                         box_alpha=0, color="w", location=3)
    ax.add_artist(scale_bar)

def show_subplot(fig, gs, row, col, image5d, channel, roi_size, offset, 
                 fn_update_seg, segs_in, segs_out, segs_cmap, alpha, 
                 highlight=False, border=None, plane="xy", roi=None, 
                 z_relative=-1, labels=None, blobs_truth=None, circles=None, 
                 aspect=None, grid=False):
    """Shows subplots of the region of interest.
    
    Args:
        fig: Matplotlib figure.
        gs: Gridspec layout.
        row: Row number of the subplot in the layout.
        col: Column number of the subplot in the layout.
        image5d: Full Numpy array of the image stack.
        channel: Channel of the image to display.
        roi_size: List of x,y,z dimensions of the ROI.
        offset: Tuple of x,y,z coordinates of the ROI.
        segs_in: Numpy array of segments within the ROI to display in the 
            subplot, which can be None. Segments are generally given as an 
            (n, 4) dimension array, where each segment is in (z, y, x, radius).
        segs_out: Subset of segments that are adjacent to rather than
            inside the ROI, which will be drawn in a different style.
        segs_cmap: Colormap for segments.
        alpha: Opacity level.
        highlight: If true, the plot will be highlighted; defaults 
            to False.
            Defaults to None.
        plane: The plane to show in each 2D plot, with "xy" to show the 
            XY plane (default) and "xz" to show XZ plane.
        roi: A denoised region of interest, to show in place of image5d for the
            zoomed images. Defaults to None, in which case image5d will be
            used instead.
        z_relative: Index of the z-plane relative to the start of the ROI; 
            defaults to -1.
    """
    ax = plt.subplot(gs[row, col])
    _hide_axes(ax)
    size = image5d.shape
    # swap columns if showing a different plane
    plane_axis = "z"
    image5d_shape_offset = 1 if image5d.ndim >= 4 else 0
    if plane == PLANE[1]:
        # "xz" planes
        size = lib_clrbrain.swap_elements(size, 0, 1, image5d_shape_offset)
        plane_axis = "y"
    elif plane == PLANE[2]:
        # "yz" planes
        size = lib_clrbrain.swap_elements(size, 0, 2, image5d_shape_offset)
        size = lib_clrbrain.swap_elements(size, 0, 1, image5d_shape_offset)
        plane_axis = "x"
    z = offset[2]
    ax.set_title("{}={}".format(plane_axis, z))
    if border is not None:
        # boundaries of border region, with xy point of corner in first 
        # elements and [width, height] in 2nd, allowing flipping for yz plane
        border_bounds = np.array(
            [border[0:2], 
            [roi_size[0] - 2 * border[0], roi_size[1] - 2 * border[1]]])
    if z < 0 or z >= size[image5d_shape_offset]:
        print("skipping z-plane {}".format(z))
        plt.imshow(np.zeros(roi_size[0:2]))
    else:
        # show the zoomed in 2D region
        
        # calculate the region depending on whether given ROI directly
        if roi is None:
            region = [0, offset[2], 
                      slice(offset[1], offset[1] + roi_size[1]), 
                      slice(offset[0], offset[0] + roi_size[0])]
            roi = image5d
            #print("roi shape: {}".format(roi.shape))
            #print("region: {}".format(region))
        else:
            region = [0, z_relative, slice(0, roi_size[1]), 
                      slice(0, roi_size[0])]
        # swap columns if showing a different plane
        if plane == PLANE[1]:
            region = lib_clrbrain.swap_elements(region, 1, 2)
        elif plane == PLANE[2]:
            region = lib_clrbrain.swap_elements(region, 1, 3)
            region = lib_clrbrain.swap_elements(region, 1, 2)
        # get the zoomed region
        if roi.ndim >= 5:
            roi = roi[tuple(region + [channel])]
        elif roi.ndim == 4:
            roi = roi[tuple(region)]
        else:
            roi = roi[tuple(region[1:])]
        
        # show labels if provided and within ROI; for some reason if added
        # after imshow, the main image gets squeezed to show full patches
        if (labels is not None and z_relative >= 0 
            and z_relative < labels.shape[0]):
            try:
                ax.contour(labels[z_relative])
            except ValueError as e:
                print(e)
                print("could not show label:\n{}".format(labels[z_relative]))
            #ax.imshow(labels[z_relative])
        
        if highlight:
            # highlight borders of z plane at bottom of ROI
            for spine in ax.spines.values():
                spine.set_edgecolor("yellow")
        if grid:
            # draw grid lines by directly editing copy of image
            grid_intervals = (roi_size[0] // 4, roi_size[1] // 4)
            roi = np.copy(roi)
            roi[::grid_intervals[0], :] = roi[::grid_intervals[0], :] / 2
            roi[:, ::grid_intervals[1]] = roi[:, ::grid_intervals[1]] / 2
        plt.imshow(roi, cmap=colormap_2d, alpha=alpha, aspect=aspect)
        
        if ((segs_in is not None or segs_out is not None) 
            and not circles == CIRCLES[2].lower()):
            segs_in = np.copy(segs_in)
            if circles is None or circles == CIRCLES[0].lower():
                # zero radius of all segments outside of current z to preserve 
                # the order of segments for the corresponding colormap order 
                # while hiding outside segments
                segs_in[segs_in[:, 0] != z_relative, 3] = 0
            
            # show segments from all z's as circles with colored outlines
            if segs_in is not None and segs_cmap is not None:
                if circles in (CIRCLES[1].lower(), CIRCLES[3].lower()):
                    z_diff = np.abs(np.subtract(segs_in[:, 0], z_relative))
                    r_orig = np.abs(np.copy(segs_in[:, 3]))
                    segs_in[:, 3] = np.subtract(
                        r_orig, np.divide(z_diff, 3))
                    # make circles below 3/4 of their original radius 
                    # invisible but not removed to preserve their corresponding 
                    # colormap index
                    segs_in[np.less(
                        segs_in[:, 3], np.multiply(r_orig, 3/4)), 3] = 0
                collection = _circle_collection(
                    segs_in, segs_cmap.astype(float) / 255.0, "none", 
                    SEG_LINEWIDTH)
                ax.add_collection(collection)
            
            # segments outside the ROI shown in black dotted line only for 
            # their corresponding z
            if segs_out is not None:
                collection_adj = _circle_collection(
                    segs_out[segs_out[:, 0] == z_relative], "k", "none", 
                    SEG_LINEWIDTH)
                collection_adj.set_linestyle("--")
                ax.add_collection(collection_adj)
            
            # overlay segments with dotted line patch and make pickable for 
            # verifying the segment
            segments_z = segs_in[segs_in[:, 3] > 0] # full annotation
            if circles == CIRCLES[3].lower():
                # when showing full annotation, include all segments in the ROI
                for i in range(len(segments_z)):
                    seg = segments_z[i]
                    if seg[0] != z_relative and seg[3] > 0:
                        # add segments to Visualizer table
                        seg[0] = z_relative
                        roi_offset = np.copy(offset)
                        roi_offset[2] -= z_relative
                        segments_z[i] = fn_update_seg(seg, offset=roi_offset)
            else:
                # apply only to segments in their current z
                segments_z = segs_in[segs_in[:, 0] == z_relative]
            if segments_z is not None:
                for seg in segments_z:
                    _plot_circle(
                        ax, seg, "w", SEG_LINEWIDTH, ":", fn_update_seg)
            
            # shows truth blobs as small, solid circles
            if blobs_truth is not None:
                for blob in blobs_truth:
                    ax.add_patch(patches.Circle(
                        (blob[2], blob[1]), radius=3, 
                        facecolor=truth_color_dict[blob[5]], alpha=1))
        
        # adds a simple border to highlight the border of the ROI
        if border is not None:
            #print("border: {}, roi_size: {}".format(border, roi_size))
            ax.add_patch(patches.Rectangle(border_bounds[0], 
                                           border_bounds[1, 0], 
                                           border_bounds[1, 1], 
                                           fill=False, edgecolor="yellow",
                                           linestyle="dashed", 
                                           linewidth=SEG_LINEWIDTH))
        
    return ax

def plot_roi(img, segments, channel, show=True, title=""):
    """Plot ROI as sequence of z-planes containing only the ROI itself.
    
    Args:
        img: The ROI image as a 3D array in (z, y, x) order.
        segments: Numpy array of segments to display in the subplot, which 
            can be None. Segments are generally given as an (n, 4)
            dimension array, where each segment is in (z, y, x, radius).
            All segments are assumed to be within the ROI for display.
        channel: Channel of the image to display.
        show: True if the plot should be displayed to screen; defaults 
            to True.
        title: String used as basename of output file. Defaults to "" 
            and only used if :attr:``savefig`` is set to a file 
            extension.
    """
    fig = plt.figure()
    #fig.suptitle(title)
    # total number of z-planes
    z_planes = img.shape[0]
    # wrap plots after reaching max, but tolerates additional column
    # if it will fit all the remainder plots from the last row
    zoom_plot_rows = math.ceil(z_planes / ZOOM_COLS)
    col_remainder = z_planes % ZOOM_COLS
    zoom_plot_cols = ZOOM_COLS
    if col_remainder > 0 and col_remainder < zoom_plot_rows:
        zoom_plot_cols += 1
        zoom_plot_rows = math.ceil(z_planes / zoom_plot_cols)
        col_remainder = z_planes % zoom_plot_cols
    roi_size = img.shape[::-1]
    zoom_offset = [0, 0, 0]
    gs = gridspec.GridSpec(
        zoom_plot_rows, zoom_plot_cols, wspace=0.1, hspace=0.1)
    
    # plot the fully zoomed plots
    for i in range(zoom_plot_rows):
        # adjust columns for last row to number of plots remaining
        cols = zoom_plot_cols
        if i == zoom_plot_rows - 1 and col_remainder > 0:
            cols = col_remainder
        # show zoomed in plots and highlight one at offset z
        for j in range(cols):
            # z relative to the start of the ROI, since segs are relative to ROI
            z = i * zoom_plot_cols + j
            zoom_offset[2] = z
            
            # shows the zoomed subplot with scale bar for the current z-plane 
            # with all segments
            ax_z = show_subplot(
                fig, gs, i, j, img, channel, roi_size, zoom_offset, None,
                segments, None, None, 1.0, circles=CIRCLES[0], z_relative=z)
            if i == 0 and j == 0:
                add_scale_bar(ax_z)
    gs.tight_layout(fig, pad=0.5)
    if show:
        plt.show()
    if savefig is not None:
        plt.savefig(title + "." + savefig)
    

def plot_2d_stack(fn_update_seg, title, filename, image5d, channel, roi_size, 
                  offset, segments, mask_in, 
                  segs_cmap, border=None, plane="xy", padding_stack=None,
                  zoom_levels=2, single_zoom_row=False, z_level=Z_LEVELS[0], 
                  roi=None, labels=None, blobs_truth=None, circles=None, 
                  mlab_screenshot=None, grid=False, zoom_cols=ZOOM_COLS):
    """Shows a figure of 2D plots to compare with the 3D plot.
    
    Args:
        title: Figure title.
        image5d: Full Numpy array of the image stack.
        channel: Channel of the image to display.
        roi_size: List of x,y,z dimensions of the ROI.
        offset: Tuple of x,y,z coordinates of the ROI.
        segments: Numpy array of segments to display in the subplot, which 
            can be None. Segments are generally given as an (n, 4)
            dimension array, where each segment is in (z, y, x, radius), and 
            coordinates are relative to ``offset``.
            This array can include adjacent segments as well.
        mask_in: Boolean mask of ``segments`` within the ROI.
        segs_cmap: Colormap for segments inside the ROI.
        border: Border dimensions in pixels given as (x, y, z); defaults
            to None.
        plane: The plane to show in each 2D plot, with "xy" to show the 
            XY plane (default) and "xz" to show XZ plane.
        padding_stack: The amount of padding in pixels, defaulting to the 
            padding attribute.
        zoom_levels: Number of zoom levels to include, with n - 1 levels
            included at the overview level, and the last one viewed
            as the series of ROI-sized plots; defaults to 2.
        single_zoom_row: True if the ROI-sized zoomed plots should be
            displayed on a single row; defaults to False.
        z_level: Position of the z-plane shown in the overview plots,
            based on the Z_LEVELS attribute constant; defaults to 
            Z_LEVELS[0].
        roi: A denoised region of interest for display in fully zoomed plots. 
            Defaults to None, in which case image5d will be used instead.
        zoom_cols: Number of columns per row to reserve for zoomed plots; 
            defaults to :attr:``ZOOM_COLS``.
    """
    time_start = time()
    fig = plt.figure()
    # black text with transluscent background the color of the figure
    # background in case the title is a 2D plot
    fig.suptitle(title, color="black", 
                 bbox=dict(facecolor=fig.get_facecolor(), edgecolor="none", 
                           alpha=0.5))
    
    # adjust array order based on which plane to show
    border_full = np.copy(border)
    border[2] = 0
    if plane == PLANE[1]:
        # "xz" planes; flip y-z to give y-planes instead of z
        roi_size = lib_clrbrain.swap_elements(roi_size, 1, 2)
        offset = lib_clrbrain.swap_elements(offset, 1, 2)
        border = lib_clrbrain.swap_elements(border, 1, 2)
        border_full = lib_clrbrain.swap_elements(border_full, 1, 2)
        segments[:, [0, 1]] = segments[:, [1, 0]]
    elif plane == PLANE[2]:
        # "yz" planes; roll backward to flip x-z and x-y
        roi_size = lib_clrbrain.roll_elements(roi_size, -1)
        offset = lib_clrbrain.roll_elements(offset, -1)
        border = lib_clrbrain.roll_elements(border, -1)
        border_full = lib_clrbrain.roll_elements(border_full, -1)
        print("orig segments:\n{}".format(segments))
        # roll forward since segments in zyx order
        segments[:, [0, 2]] = segments[:, [2, 0]]
        segments[:, [1, 2]] = segments[:, [2, 1]]
        print("rolled segments:\n{}".format(segments))
    print("2D border: {}".format(border))
    
    # total number of z-planes
    z_start = offset[2]
    z_planes = roi_size[2]
    if padding_stack is None:
        padding_stack = padding
    z_planes_padding = padding_stack[2] # additional z's above/below
    print("padding: {}, savefig: {}".format(padding, savefig))
    z_planes = z_planes + z_planes_padding * 2
    z_overview = z_start
    if z_level == Z_LEVELS[1]:
        z_overview = (2 * z_start + z_planes) // 2
    elif z_level == Z_LEVELS[2]:
        z_overview = z_start + z_planes
    print("z_overview: {}".format(z_overview))
    
    # pick image based on chosen orientation
    img2d, aspect, origin = extract_plane(image5d, z_overview, plane)
    
    # plot layout depending on number of z-planes
    if single_zoom_row:
        # show all plots in single row
        zoom_plot_rows = 1
        col_remainder = 0
        zoom_plot_cols = z_planes
    else:
        # wrap plots after reaching max, but tolerates additional column
        # if it will fit all the remainder plots from the last row
        zoom_plot_rows = math.ceil(z_planes / zoom_cols)
        col_remainder = z_planes % zoom_cols
        zoom_plot_cols = zoom_cols
        if col_remainder > 0 and col_remainder < zoom_plot_rows:
            zoom_plot_cols += 1
            zoom_plot_rows = math.ceil(z_planes / zoom_plot_cols)
            col_remainder = z_planes % zoom_plot_cols
    # overview plots is 1 > levels, but last spot is taken by screenshot
    top_cols = zoom_levels
    height_ratios = (3, zoom_plot_rows)
    if mlab_screenshot is None:
        # remove column for screenshot
        top_cols -= 1
        if img2d.shape[1] > 2 * img2d.shape[0]:
            # for wide ROIs, prioritize the fully zoomed plots, especially 
            # if only one overview column
            height_ratios = (1, 1) if top_cols >= 2 else (1, 2)
    gs = gridspec.GridSpec(2, top_cols, wspace=0.7, hspace=0.4,
                           height_ratios=height_ratios)
    
    
    
    # overview images taken from the bottom plane of the offset, with
    # progressively zoomed overview images if set for additional zoom levels
    overview_cols = zoom_plot_cols // zoom_levels
    for i in range(zoom_levels - 1):
        ax = plt.subplot(gs[0, i])
        _hide_axes(ax)
        img2d_zoom = img2d
        patch_offset = offset[0:2]
        print("patch_offset: {}".format(patch_offset))
        if i > 0:
            # move origin progressively closer withe ach zoom level
            origin = np.floor(np.multiply(offset[0:2], zoom_levels + i - 1) 
                              / (zoom_levels + i)).astype(int)
            zoom_shape = np.flipud(img2d.shape)[0:2]
            # progressively decrease size, zooming in for each level
            size = np.floor(zoom_shape / (i + 3)).astype(int)
            end = np.add(origin, size)
            # keep the zoomed area within the full 2D image
            for j in range(len(origin)):
                if end[j] > zoom_shape[j]:
                    origin[j] -= end[j] - zoom_shape[j]
            img2d_zoom = img2d_zoom[origin[1]:end[1], origin[0]:end[0]]
            print(img2d_zoom.shape, origin, size)
            patch_offset = np.subtract(patch_offset, origin)
        # show the zoomed 2D image along with rectangle showing ROI, 
        # downsampling by using threshold as mask
        downsample = np.max(np.divide(img2d_zoom.shape, _DOWNSAMPLE_THRESH)).astype(np.int)
        if downsample < 1: 
            downsample = 1
        ax.imshow(
            img2d_zoom[::downsample, ::downsample], cmap=colormap_2d, 
            aspect=aspect, origin=origin, vmin=0.0, vmax=vmax_overview)
        ax.add_patch(patches.Rectangle(
            np.divide(patch_offset, downsample), 
            *np.divide(roi_size[0:2], downsample), 
            fill=False, edgecolor="yellow"))
        add_scale_bar(ax)
    
    # zoomed-in views of z-planes spanning from just below to just above ROI
    ax_z_list = []
    segs_in = None
    segs_out = None
    if segments is not None and len(segments) > 0:
        # separate segments inside from outside the ROI
        if mask_in is not None:
            segs_in = segments[mask_in]
            segs_out = segments[np.invert(mask_in)]
        # separate out truth blobs
        if segments.shape[1] >= 6:
            if blobs_truth is None:
                blobs_truth = segments[segments[:, 5] >= 0]
            print("blobs_truth:\n{}".format(blobs_truth))
            # non-truth blobs have truth flag unset (-1)
            segs_in = segs_in[segs_in[:, 5] == -1]
            segs_out = segs_out[segs_out[:, 5] == -1]
            #print("segs_in:\n{}".format(segs_in))
        
    # selected or newly added patches since difficult to get patch from collection,
    # and they don't appear to be individually editable
    seg_patch_dict = {}
    
    # sub-gridspec for fully zoomed plots to allow flexible number of columns
    gs_zoomed = gridspec.GridSpecFromSubplotSpec(zoom_plot_rows, zoom_plot_cols, 
                                                 gs[1, :],
                                                 wspace=0.1, hspace=0.1)
    # plot the fully zoomed plots
    #zoom_plot_rows = 0 # TESTING: show no fully zoomed plots
    for i in range(zoom_plot_rows):
        # adjust columns for last row to number of plots remaining
        cols = zoom_plot_cols
        if i == zoom_plot_rows - 1 and col_remainder > 0:
            cols = col_remainder
        # show zoomed in plots and highlight one at offset z
        for j in range(cols):
            # z relative to the start of the ROI, since segs are relative to ROI
            z_relative = i * zoom_plot_cols + j - z_planes_padding
            # absolute z value, relative to start of image5d
            z = z_start + z_relative
            zoom_offset = (offset[0], offset[1], z)
            
            # fade z-planes outside of ROI and show only image5d
            if z < z_start or z >= z_start + roi_size[2]:
                alpha = 0.5
                roi_show = None
            else:
                alpha = 1
                roi_show = roi
            
            # collects truth blobs within the given z-plane
            blobs_truth_z = None
            if blobs_truth is not None:
                blobs_truth_z = blobs_truth[np.all([
                    blobs_truth[:, 0] == z_relative, 
                    blobs_truth[:, 4] > 0], axis=0)]
            #print("blobs_truth_z:\n{}".format(blobs_truth_z))
            
            # shows border outlining area that will be saved if in verify mode
            show_border = (verify and z_relative >= border[2] 
                           and z_relative < roi_size[2] - border[2])
            
            # shows the zoomed subplot with scale bar for the current z-plane
            ax_z = show_subplot(
                fig, gs_zoomed, i, j, image5d, channel, roi_size, zoom_offset, 
                fn_update_seg,
                segs_in, segs_out, segs_cmap, alpha, z == z_overview, 
                border_full if show_border else None, plane, roi_show, 
                z_relative, labels, blobs_truth_z, circles=circles, 
                aspect=aspect, grid=grid)
            if i == 0 and j == 0:
                add_scale_bar(ax_z)
            ax_z_list.append(ax_z)
    
    # add points that were not segmented by ctrl-clicking on zoom plots
    def on_btn_release(event):
        ax = event.inaxes
        if event.key == "control":
            try:
                axi = ax_z_list.index(ax)
                if (axi != -1 and axi >= z_planes_padding 
                    and axi < z_planes - z_planes_padding):
                    
                    seg = np.array([axi - z_planes_padding, 
                                     event.ydata.astype(int), 
                                     event.xdata.astype(int), -5, 1, -1])
                    seg = fn_update_seg(seg, offset=offset)
                    # adds a circle to denote the new segment
                    patch = _plot_circle(
                        ax, seg, "none", SEG_LINEWIDTH, "-", fn_update_seg)
            except ValueError as e:
                print(e)
                print("not on a plot to select a point")
        elif event.key == "v":
            _circle_last_picked_len = len(_circle_last_picked)
            if _circle_last_picked_len < 1:
                print("No previously picked circle to paste")
                return
            moved_item = _circle_last_picked[_circle_last_picked_len - 1]
            circle, move_type = moved_item
            axi = ax_z_list.index(ax)
            dz = axi - z_planes_padding - circle.segment[0]
            seg_old = np.copy(circle.segment)
            seg_new = np.copy(circle.segment)
            seg_new[0] += dz
            if move_type == _CUT:
                print("Pasting a cut segment")
                _draggable_circles.remove(circle)
                _circle_last_picked.remove(moved_item)
                seg_new = fn_update_seg(seg_new, seg_old)
            else:
                print("Pasting a copied in segment")
                seg_new = fn_update_seg(seg_new, offset=offset)
            _plot_circle(
                ax, seg_new, "w", SEG_LINEWIDTH, ":", fn_update_seg)
       
    fig.canvas.mpl_connect("button_release_event", on_btn_release)
    
    # show 3D screenshot if available
    if mlab_screenshot is not None:
        img3d = mlab_screenshot
        ax = plt.subplot(gs[0, zoom_levels - 1])
        # auto to adjust size with less overlap
        ax.imshow(img3d)
        ax.set_aspect(img3d.shape[1] / img3d.shape[0])
        _hide_axes(ax)
    gs.tight_layout(fig, pad=0.5)
    #gs_zoomed.tight_layout(fig, pad=0.5)
    plt.ion()
    plt.show()
    fig.set_size_inches(*(fig.get_size_inches() * 1.5), True)
    if savefig is not None:
        name = "{}_offset{}x{}.{}".format(
            os.path.basename(filename), offset, tuple(roi_size), 
            savefig).replace(" ", "")
        print("saving figure as {}".format(name))
        plt.savefig(name)
    print("2D plot time: {}".format(time() - time_start))
    
def _hide_axes(ax):
    """Hides x- and y-axes.
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def extract_plane(image5d, plane_n, plane=None, channel=0, savefig=None, 
                  name=None):
    """Extracts a single 2D plane and saves to file.
    
    Args:
        image5d: The full image stack.
        plane_n: Slice of planes to extract, which can be a single index 
            or multiple indices such as would be used for an animation.
        channel: Channel to view.
        plane: Type of plane to extract, which should be one of 
            :attribute:`PLANES`.
        name: Name of the resulting file, without the extension.
        savefig: Name of extension to use, which also specifies the file 
            type in which to save.
    """
    origin = None
    aspect = None # aspect ratio
    img3d = None
    if image5d.ndim >= 5:
        img3d = image5d[0, :, :, :, channel]
    elif image5d.ndim == 4:
        img3d = image5d[0, :, :, :]
    else:
        img3d = image5d[:, :, :]
    # extract a single 2D plane or a stack of planes if plane_n is a slice, 
    # which would be used for animations
    if plane == PLANE[1]:
        # xz plane
        aspect = detector.resolutions[0, 0] / detector.resolutions[0, 2]
        origin = "lower"
        img2d = img3d[:, plane_n, :]
        print("img2d.shape: {}".format(img2d.shape))
        if img2d.ndim > 2 and img2d.shape[1] > 1:
            # make y the "z" axis for stack of 2D plots, such as animations
            img2d = np.swapaxes(img2d, 0, 1)
    elif plane == PLANE[2]:
        # yz plane
        aspect = detector.resolutions[0, 0] / detector.resolutions[0, 1]
        origin = "lower"
        img2d = img3d[:, :, plane_n]
        if img2d.ndim > 2 and img2d.shape[2] > 1:
            # make x the "z" axis for stack of 2D plots, such as animations
            img2d = np.moveaxis(img2d, -1, 0)
    else:
        # defaults to "xy"
        aspect = detector.resolutions[0, 1] / detector.resolutions[0, 2]
        img2d = img3d[plane_n, :, :]
    print("aspect: {}, origin: {}".format(aspect, origin))
    if savefig is not None:
        filename = name + "." + savefig
        print("extracting plane as {}".format(filename))
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        _hide_axes(ax)
        ax.imshow(img2d, cmap=CMAP_GRBK, aspect=aspect, origin=origin)
        fig.savefig(filename)
    return img2d, aspect, origin

def cycle_colors(i):
    num_colors = len(config.colors)
    cycle = i // num_colors
    colori = i % num_colors
    color = config.colors[colori]
    '''
    print("num_colors: {}, cycle: {}, colori: {}, color: {}"
          .format(num_colors, cycle, colori, color))
    '''
    upper = 255
    if cycle > 0:
        color = np.copy(color)
        color[0] = color[0] + cycle * 5
        if color[0] > upper:
            color[0] -= upper * (color[0] // upper)
    return np.divide(color, upper)

def plot_roc(stats_dict, name):
    """Plot ROC curve.
    
    Args:
        stats_dict: Dictionary of statistics to plot as given by 
            :func:``mlearn.parse_grid_stats``.
        name: String to display as title.
    """
    fig = plt.figure()
    posi = 1 # position of legend
    for group, iterable_dicts in stats_dict.items():
        lines = []
        colori = 0
        for key, value in iterable_dicts.items():
            fdr = value[0]
            sens = value[1]
            params = value[2]
            line, = plt.plot(
                fdr, sens, label=key, lw=2, color=cycle_colors(colori), 
                linestyle="", marker=".")
            lines.append(line)
            colori += 1
            for i, n in enumerate(params):
                plt.annotate(n, (fdr[i], sens[i]))
        # iterated legend position to avoid overlap from multiple legends
        legend = plt.legend(
            handles=lines, loc=posi, title=group, fancybox=True, 
            framealpha=0.5)
        plt.gca().add_artist(legend)
        posi += 1
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.2])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Discovery Rate")
    plt.ylabel("Sensitivity")
    plt.title("ROC for {}".format(name))
    plt.show()

def _show_overlay(ax, img, plane_i, cmap, out_plane, aspect=1.0, alpha=1.0, 
                  title=None):
    """Shows an image for overlays in the orthogonal plane specified by 
    :attribute:`plane`.
    
    Args:
        ax: Subplot axes.
        img: 3D image.
        plane_i: Plane index of `img` to show.
        cmap: Name of colormap.
        aspect: Aspect ratio; defaults to 1.0.
        alpha: Alpha level; defaults to 1.0.
        title: Subplot title; defaults to None, in which case no title will 
            be shown.
    """
    if out_plane == PLANE[1]:
        # xz plane
        img_2d = img[:, plane_i]
        img_2d = np.flipud(img_2d)
    elif out_plane == PLANE[2]:
        # yz plane, which requires a flip when original orientation is 
        # horizontal section
        # TODO: generalize to other original orientations
        img_2d = img[:, :, plane_i]
        #img_2d = np.swapaxes(img_2d, 1, 0)
        #aspect = 1 / aspect
        img_2d = np.flipud(img_2d)
    else:
        # xy plane (default)
        img_2d = img[plane_i]
    ax.imshow(img_2d, cmap=cmap, aspect=aspect, alpha=alpha)
    _hide_axes(ax)
    if title is not None:
        ax.set_title(title)

def plot_overlays(imgs, z, cmaps, title=None, aspect=1.0):
    """Plot images in a single row, with the final subplot showing an 
    overlay of all images.
    
    Args:
        imgs: List of 3D images to show.
        z: Z-plane to view for all images.
        cmaps: List of colormap names, which should be be the same length as 
            `imgs`, with the colormap applied to the corresponding image.
        title: Figure title; if None, will be given default title.
        aspect: Aspect ratio, which will be applied to all images; 
           defaults to 1.0.
    """
    # TODO: deprecated
    fig = plt.figure()
    fig.suptitle(title)
    imgs_len = len(imgs)
    gs = gridspec.GridSpec(1, imgs_len + 1)
    for i in range(imgs_len):
        print("showing img {}".format(i))
        _show_overlay(plt.subplot(gs[0, i]), imgs[i], z, cmaps[i], aspect)
    ax = plt.subplot(gs[0, imgs_len])
    for i in range(imgs_len):
        _show_overlay(ax, imgs[i], z, cmaps[i], aspect, alpha=0.5)
    if title is None:
        title = "Image overlays"
    gs.tight_layout(fig)
    plt.show()

def plot_overlays_reg(exp, atlas, atlas_reg, labels_reg, cmap_exp, 
                      cmap_atlas, cmap_labels, translation=None, title=None, 
                      out_plane=None):
    """Plot overlays of registered 3D images, showing overlap of atlas and 
    experimental image planes.
    
    Shows the figure on screen. If :attribute:plot_2d:`savefig` is set, 
    the figure will be saved to file with the extensive given by savefig.
    
    Args:
        exp: Experimental image.
        atlas: Atlas image, unregistered.
        atlas_reg: Atlas image, after registration.
        labels_reg: Atlas labels image, also registered.
        cmap_exp: Colormap for the experimental image.
        cmap_atlas: Colormap for the atlas.
        cmap_labels: Colormap for the labels.
        translation: Translation in (z, y, x) order for consistency with 
            operations on Numpy rather than SimpleITK images here; defaults 
            to None, in which case the chosen plane index for the 
            unregistered atlast will be the same fraction of its size as for 
            the registered image.
        title: Figure title; if None, will be given default title.
        out_plane: Output planar orientation.
    """
    fig = plt.figure()
    # give extra space to the first row since the atlas is often larger
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 2])
    resolution = detector.resolutions[0]
    #size_ratio = np.divide(atlas_reg.shape, exp.shape)
    aspect = 1.0
    z = 0
    atlas_z = 0
    plane_frac = 2#5 / 2
    if out_plane is None:
        out_plane = plane
    if out_plane == PLANE[1]:
        # xz plane
        aspect = resolution[0] / resolution[2]
        z = exp.shape[1] // plane_frac
        if translation is None:
            atlas_z = atlas.shape[1] // plane_frac
        else:
            atlas_z = int(z - translation[1])
    elif out_plane == PLANE[2]:
        # yz plane
        aspect = resolution[0] / resolution[1]
        z = exp.shape[2] // plane_frac
        if translation is None:
            atlas_z = atlas.shape[2] // plane_frac
        else:
            # TODO: not sure why needs to be addition here
            atlas_z = int(z + translation[2])
    else:
        # xy plane (default)
        aspect = resolution[1] / resolution[2]
        z = exp.shape[0] // plane_frac
        if translation is None:
            atlas_z = atlas.shape[0] // plane_frac
        else:
            atlas_z = int(z - translation[0])
    print("z: {}, atlas_z: {}, aspect: {}".format(z, atlas_z, aspect))
    
    # invert any neg values (one hemisphere) to minimize range and match other
    # hemisphere
    labels_reg[labels_reg < 0] = np.multiply(labels_reg[labels_reg < 0], -1)
    vmin, vmax = np.percentile(labels_reg, (5, 95))
    print("vmin: {}, vmax: {}".format(vmin, vmax))
    labels_reg = exposure.rescale_intensity(labels_reg, in_range=(vmin, vmax))
    '''
    labels_reg = labels_reg.astype(np.float)
    lib_clrbrain.normalize(labels_reg, 1, 100, background=15000)
    labels_reg = labels_reg.astype(np.int)
    print(labels_reg[290:300, 20, 190:200])
    '''
    
    # experimental image and atlas
    _show_overlay(plt.subplot(gs[0, 0]), exp, z, cmap_exp, out_plane, aspect, 
                              title="Experiment")
    _show_overlay(plt.subplot(gs[0, 1]), atlas, atlas_z, cmap_atlas, out_plane, 
                  alpha=0.5, title="Atlas")
    
    # atlas overlaid onto experiment
    ax = plt.subplot(gs[0, 2])
    _show_overlay(ax, exp, z, cmap_exp, out_plane, aspect, title="Registered")
    _show_overlay(ax, atlas_reg, z, cmap_atlas, out_plane, aspect, 0.5)
    
    # labels overlaid onto atlas
    ax = plt.subplot(gs[1, 0])
    _show_overlay(ax, atlas_reg, z, cmap_atlas, out_plane, aspect, title="Labeled atlas")
    _show_overlay(ax, labels_reg, z, cmap_labels, out_plane, aspect, 0.5)
    
    # labels overlaid onto exp
    ax = plt.subplot(gs[1, 1])
    _show_overlay(ax, exp, z, cmap_exp, out_plane, aspect, title="Labeled experiment")
    _show_overlay(ax, labels_reg, z, cmap_labels, out_plane, aspect, 0.5)
    
    # all overlaid
    ax = plt.subplot(gs[1, 2])
    _show_overlay(ax, exp, z, cmap_exp, out_plane, aspect, title="All overlaid")
    _show_overlay(ax, atlas_reg, z, cmap_atlas, out_plane, aspect, 0.5)
    _show_overlay(ax, labels_reg, z, cmap_labels, out_plane, aspect, 0.3)
    
    if title is None:
        title = "Image Overlays"
    fig.suptitle(title)
    gs.tight_layout(fig)
    if savefig is not None:
        plt.savefig(title + "." + savefig)
    plt.show()

def _bar_plots(ax, lists, errs, list_names, x_labels, colors, width, y_label, title):
    """Generate grouped bar plots from lists, where corresponding elements 
    in the lists are grouped together.
    
    Args:
        ax: Axes.
        lists: Tuple of mean lists to display, with each list getting a 
            separate set of bar plots. All lists should be the same size as one 
            another.
        errs: Tuple of variance lists (eg standard deviation or error) to 
            display, with each list getting a separate
            set of bar plots. All lists should be the same size as one 
            another.
        list_names: List of names of each list, where the list size should 
            be the same size as the size of ``lists``.
        x_labels: List of labels for each bar group, where the list size 
            should be equal to the size of each list in ``lists``.
        width: Width of each bar.
        y_label: Y-axis label.
        title: Graph title.
    """
    bars = []
    if len(lists) < 1: return
    indices = np.arange(len(lists[0]))
    #print("lists: {}".format(lists))
    #print("errs: {}".format(errs))
    for i in range(len(lists)):
        err = None if errs is None else errs[i]
        bars.append(ax.bar(
            indices + width * i, lists[i], width=width, color=colors[i], 
            linewidth=0, yerr=err))
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(indices + width)
    ax.set_xticklabels(x_labels, rotation=80, horizontalalignment="right")
    ax.legend(bars, list_names, loc="best", fancybox=True, framealpha=0.5)

def _get_mean_sem(group_dict, key, vals):
    """Get the mean and standard error of the mean (SEM), storing the SEM 
    in the dictionary.
    
    Args:
        group_dict: Dictionary where the SEM will be stored.
        key: Key at which the SEM will be stored.
        vals: Values from which to calculate the SEM.
    
    Returns:
        The mean of ``vals``.
    """
    group_dict[key].append(stats.sem(vals))
    return np.mean(vals)

def plot_volumes(volumes_dict, ignore_empty=False, title=None, densities=False, 
                 show=True, multiple=False):
    """Plot volumes and densities.
    
    Args:
        volumes_dict: Dictionary of volumes as generated by 
            :func:``register.volumes_by_id``, including values from 
            individual or grouped experiments.
        ignore_empty: True if empty volumes should be ignored; defaults to 
            False.
        title: Title to display for the entire figure; defaults to None, in 
            while case a generic title will be given.
        densities: True if densities should be extracted and displayed from 
            the volumes dictionary; defaults to False.
        show: True if plots should be displayed; defaults to True.
        multiple: True if ``volumes_dict`` contains values from multiple 
            experiments, in which case means and standard errors of mean will 
            be calculated.
    """
    # setup figure layout with single subplot for volumes only or 
    # side-by-side subplots if including densities
    fig = plt.figure()
    subplots_width = 2 if densities else 1
    gs = gridspec.GridSpec(1, subplots_width)
    ax_vols = plt.subplot(gs[0, 0])
    ax_densities = plt.subplot(gs[0, 1]) if densities else None
    
    # measurement units, assuming a base unit of microns
    unit_factor = np.power(1 * 1000.0, 3)
    unit = "mm"
    width = 0.1 # default bar width
    
    SIDE = "side"
    MIR = "mirrored"
    SIDE_SEM = SIDE + "_sem"
    MIR_SEM = MIR + "_sem"
    vol_group = {SIDE: [], MIR: [], SIDE_SEM: [], MIR_SEM: []}
    dens_group = copy.deepcopy(vol_group)
    names = []
    for key in volumes_dict.keys():
        # find negative keys based on the given positive key to show them
        # side-by-side
        if key >= 0:
            name = volumes_dict[key][config.ABA_NAME]
            
            # get volumes in the given unit
            vol_side = np.divide(volumes_dict[key][config.VOL_KEY], unit_factor)
            vol_mirrored = np.divide(
                volumes_dict[-1 * key].get(config.VOL_KEY), unit_factor)
            
            if (ignore_empty and vol_mirrored is not None 
                and abs(np.sum([vol_side, vol_mirrored])) < config.POS_THRESH):
                # skip completely empty regions if flagged
                print("skipping {} as both sides are empty".format(name))
            else:
                # extract names and volumes
                names.append(name)
                if multiple:
                    # calculate 
                    vol_side = _get_mean_sem(vol_group, SIDE_SEM, vol_side)
                    vol_mirrored = _get_mean_sem(vol_group, MIR_SEM, vol_mirrored)
                vol_group[SIDE].append(vol_side)
                if vol_mirrored is None:
                    volume_mirrored = 0
                vol_group[MIR].append(vol_mirrored)
                if densities:
                    # calculate densities based on blobs counts and volumes
                    blobs_side = volumes_dict[key][config.BLOBS_KEY]
                    blobs_mirrored = volumes_dict[-1 * key].get(
                        config.BLOBS_KEY)
                    print("id {}: blobs R {}, L {}".format(
                        key, blobs_side, blobs_mirrored))
                    density_side = np.divide(blobs_side, vol_side)
                    density_mirrored = np.divide(blobs_mirrored, vol_mirrored)
                    if multiple:
                        density_side = _get_mean_sem(dens_group, SIDE_SEM, density_side)
                        density_mirrored = _get_mean_sem(dens_group, MIR_SEM, density_mirrored)
                    dens_group[SIDE].append(density_side)
                    dens_group[MIR].append(density_mirrored)
    
    # generate bar plots
    legend_names = ("Left", "Right")
    bar_colors = ("b", "g")
    errs = (vol_group[MIR_SEM], vol_group[SIDE_SEM]) if multiple else None
    _bar_plots(
        ax_vols, (vol_group[MIR], vol_group[SIDE]), errs, legend_names, names, 
        bar_colors, width, "Volume (cubic {})".format(unit), "Volumes")
    if densities:
        errs = (dens_group[MIR_SEM], dens_group[SIDE_SEM]) if multiple else None
        _bar_plots(
            ax_densities, (dens_group[MIR], dens_group[SIDE]), errs, legend_names, 
            names, bar_colors, width, 
            "Cell density (cells / cubic {})".format(unit), "Densities")
    
    # finalize the image with title and tight layout
    if title is None:
        title = "Regional Volumes"
    fig.suptitle(title)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.97]) # extra padding for title
    if savefig is not None:
        plt.savefig(title + "." + savefig)
    if show:
        plt.show()

if __name__ == "__main__":
    print("Testing plot_2d...")
    stats_dict = { 
        "test1": (np.array([[5, 4, 3], [8, 3, 4]]), [10, 20]),
        "test2": (np.array([[1225, 1200, 95], [235, 93, 230]]), [25, 34])
    }
    plot_roc(stats_dict, "Testing ROC")
    