# ROI Editor with serial 2D viewer and annotator
# Author: David Young, 2018, 2020
"""ROI editing GUI in the MagellanMapper package.

Attributes:
    verify: If true, verification mode is turned on, which for now
        simply turns on interior borders as the picker remains on
        by default.
    padding: Padding in pixels (x, y), or planes (z) in which to show
        extra segments.
"""

from collections import OrderedDict
import math
import os
from enum import Enum
import re
from time import time

import numpy as np
from matplotlib import figure
from matplotlib import gridspec
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

from magmap.gui import pixel_display
from magmap.gui import plot_editor
from magmap.plot import colormaps
from magmap.settings import config
from magmap.cv import detector
from magmap.io import importer
from magmap.io import libmag
from magmap.plot import plot_support

verify = False


class DraggableCircle:
    """Circle representation of a blob to allow the user to manipulate 
    blob position, size, and status.
    
    Attributes:
        BLOB_COLORS (:obj:`dict`): Mapping of integers to ``Matplotlib`` 
            color strings.
        CUT (str): Flag to cut a circle.
        circle (:obj:`patches.Circle`): A circle patch. 
        segment (:obj:`np.ndarray`): Array in the format, 
            `[z, y, x, r, confirmation, truth]` of the blob. 
        fn_update_seg (`meth`): Function that takes 
            `(segment_new, segment_old)` to call when updating the blob.
        picked (:obj:`list`): List of picked, active blobs in the 
            tuple format, `(segment, pick_flag)`.
    
    """
    
    # segment colors based on confirmation status
    BLOB_COLORS = {
        -1: "none",
        0: "r",
        1: "g",
        2: "y"
    }

    CUT = "cut"
    
    #: str: Flag to copy a circle.
    _COPY = "copy"

    picked = None

    def __init__(self, circle, segment, fn_update_seg, picked, color="none"):
        """Initialize a circle from a blob.
        
        Args:
            circle (:obj:`patches.Circle`): A circle patch. 
            segment (:obj:`np.ndarray`): Array in the format, 
                `[z, y, x, r, confirmation, truth]` of the blob. 
            fn_update_seg (`meth`): Function that takes 
                `(segment_new, segment_old)` to call when updating the blob.
            picked (:obj:`list`): List of picked, active blobs in the 
                tuple format, `(segment, pick_flag)`.
            color (str, optional): ``Matplotlib`` color string for the circle; 
                defaults to "none".
        
        """
        self.circle = circle
        self.circle.set_picker(self.circle.radius)
        self._facecolori = -1
        for key, val in self.BLOB_COLORS.items():
            if val == color:
                self._facecolori = key
        self.segment = segment
        self.fn_update_seg = fn_update_seg
        self.picked = picked

        self._press = None  # event position
        self._background = None  # bbox of bkgd for blitting
        
        # event connection objects
        self._cidpress = None
        self._cidrelease = None
        self._cidmotion = None
        self._cidpick = None

    def connect(self):
        """Connect events to functions.
        """
        self._cidpress = self.circle.figure.canvas.mpl_connect(
            "button_press_event", self.on_press)
        self._cidrelease = self.circle.figure.canvas.mpl_connect(
            "button_release_event", self.on_release)
        self._cidmotion = self.circle.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_motion)
        self._cidpick = self.circle.figure.canvas.mpl_connect(
            "pick_event", self.on_pick)
        #print("connected circle at {}".format(self.circle.center))

    def remove_self(self):
        self.disconnect()
        self.circle.remove()

    def on_press(self, event):
        """Initiate drag events with Shift- or Alt-click inside a circle.

        Shift-click to move a circle, and Alt-click to resize a circle's radius.
        """
        if (event.key not in ("shift", "alt")
                or event.inaxes != self.circle.axes):
            # ignore if without the given modifiers or outside of circle's axes
            return
        # ensure that event is within the circle
        contains, attrd = self.circle.contains(event, radius=self.circle.radius)
        if not contains: return
        print("pressed on {}".format(self.circle.center))
        x0, y0 = self.circle.center
        self._press = x0, y0, event.xdata, event.ydata
        DraggableCircle.lock = self

        # draw everywhere except the circle itself, store the pixel buffer
        # in background, and draw the circle
        canvas = self.circle.figure.canvas
        ax = self.circle.axes
        self.circle.set_animated(True)
        canvas.draw_idle()
        self._background = canvas.copy_from_bbox(self.circle.axes.bbox)
        ax.draw_artist(self.circle)
        canvas.blit(ax.bbox)

    def on_motion(self, event):
        """Move the circle if the drag event has been initiated.
        """
        if self._press is None: return
        if event.inaxes != self.circle.axes: return
        x0, y0, xpress, ypress = self._press
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

        # restore the saved background and redraw the circle at its new position
        canvas = self.circle.figure.canvas
        ax = self.circle.axes
        canvas.restore_region(self._background)
        ax.draw_artist(self.circle)
        canvas.blit(ax.bbox)

    def on_release(self, event):
        """Finalize the circle and segment's position after a drag event
        is completed with a button release.
        """
        if self._press is None: return
        print("released on {}".format(self.circle.center))
        print("segment moving from {}...".format(self.segment))
        seg_old = np.copy(self.segment)
        self.segment[1:3] += np.subtract(
            self.circle.center, self._press[0:2]).astype(np.int)[::-1]
        rad_sign = -1 if self.segment[3] < config.POS_THRESH else 1
        self.segment[3] = rad_sign * self.circle.radius
        print("...to {}".format(self.segment))
        self.fn_update_seg(self.segment, seg_old)
        self._press = None

        # turn off animation property, reset background
        DraggableCircle.lock = None
        self.circle.set_animated(False)
        self._background = None
        self.circle.figure.canvas.draw()

    def on_pick(self, event):
        """Select the verification flag with button press on a circle when
        not dragging the circle.
        """
        if (event.mouseevent.key in ("control", "shift", "alt")
                or event.artist != self.circle):
            return
        #print("color: {}".format(self.facecolori))
        if event.mouseevent.key == "x":
            # "cut" segment
            self.picked.append((self, self.CUT))
            self.remove_self()
            print("cut seg: {}".format(self.segment))
        elif event.mouseevent.key == "c":
            # "copy" segment
            self.picked.append((self, self._COPY))
            print("copied seg: {}".format(self.segment))
        elif event.mouseevent.key == "d":
            # delete segment, which will be stored as cut segment to allow
            # undoing the deletion by pasting
            self.picked.append((self, self.CUT))
            self.remove_self()
            self.fn_update_seg(self.segment, remove=True)
            print("deleted seg: {}".format(self.segment))
        else:
            # change verification flag
            seg_old = np.copy(self.segment)
            # "r"-click to change flag in reverse order
            change = -1 if event.mouseevent.key == "r" else 1
            i = self._facecolori + change
            # wrap around keys if exceeding min/max
            if i > max(self.BLOB_COLORS.keys()):
                if self.segment[3] < config.POS_THRESH:
                    # user-added segments simply disappear when exceeding
                    self.picked.append((self, self.CUT))
                    self.remove_self()
                i = -1
            elif i < min(self.BLOB_COLORS.keys()):
                i = max(self.BLOB_COLORS.keys())
            self.circle.set_facecolor(self.BLOB_COLORS[i])
            self._facecolori = i
            self.segment[4] = i
            self.fn_update_seg(self.segment, seg_old)
            print("picked segment: {}".format(self.segment))
        self.circle.figure.canvas.draw()

    def disconnect(self):
        """Disconnect event listeners.
        """
        self.circle.figure.canvas.mpl_disconnect(self._cidpress)
        self.circle.figure.canvas.mpl_disconnect(self._cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self._cidmotion)
        self.circle.figure.canvas.mpl_disconnect(self._cidpick)


class ROIEditor(plot_support.ImageSyncMixin):
    """Graphical interface for viewing and annotating 3D ROIs through
    serial 2D planes.

    Provides overview plots showing context for the ROI at various
    zoom levels, which can be synchronized with the selected 2D plane or
    scrolled to other planes.

    Overlays detected blobs as :class:``DraggableCircle`` objects to
    flag, reposition, or add/subtract annotations.

    :attr:`plot_eds` are dictionaries where keys are zoom levels and values
    are Plot Editors.
    
    Attributes:
        ROI_COLS (int): Default number of columns for the "zoomed-in"
            2D plots, the 2D planes for the ROI.
        ZLevels (:obj:`Enum`): Enum denoting the possible positions of the
            z-plane shown in the overview plots.
        fig (:obj:`figure.figure`): Matplotlib figure.
        image5d (:obj:`np.ndarray`): Main image array in ``t,z,y,x[,c]``
            format; defaults to None.
        labels_img (:obj:`np.ndarray`): Atlas labels image in ``z,y,x`` format;
            defaults to None.
        img_region (:obj:`np.ndarray`): 3D boolean or binary array with the
            selected region labeled as True or 1. Defaults to None, in
            which case the region will be ignored.
        fn_status_bar (func): Function to call during status bar updates
            in :class:`pixel_display.PixelDisplay`; defaults to None.
        max_intens_proj: True to show overview images as local max intensity
            projections through the ROI. Defaults to Faslse.
        zoom_shift (List[float]): Sequence of x,y shift in zoomed plot
            origin when zooming into ROI; defaults to None.
        fn_update_coords (func): Function to call when updating coordinates
            in the overview plots; defaults to None.
        fn_redraw (func): Function to call when double-clicking in an
            overview plot; defaults to None.
        blobs (:obj:`magmap.cv.detector.Blobs`]): Blobs object; defaults
            to None. Blobs should have coordinates relative to the ROI
            and may include blobs from adjacent regions.
    """
    ROI_COLS = 9

    ZLevels = Enum(
        "ZLevels", (
            "BOTTOM", "MIDDLE", "TOP",
        )
    )

    class CircleStyles(Enum):
        CIRCLES = "Blob circles"
        REPEAT_CIRCLES = "Repeat circles"
        NO_CIRCLES = "No circles"
        FULL_ANNOTATION = "Full annotation"

    # segment line styles based on channel
    _BLOB_LINESTYLES = {
        0: ":",
        1: "-.",
        2: "--",
        3: (0, (3, 5, 1, 5, 1, 5)),
        4: "-",
    }

    # face colors for truth flags
    _TRUTH_COLORS = {
        -1: None,
        0: "m",
        1: "b"
    }

    _BLOB_LINEWIDTH = 1

    # divisor for finding array interval to downsample images
    _DOWNSAMPLE_MAX_ELTS = 1000
    
    #: int: padding for ROI within overview plots
    _ROI_PADDING = 10

    def __init__(self, image5d=None, labels_img=None, img_region=None,
                 fn_show_label_3d=None, fn_status_bar=None):
        """Initialize the editor."""
        super().__init__()
        print("Initiating ROI Editor")
        self.image5d = image5d
        self.labels_img = labels_img
        if img_region is not None:
            # invert region selection image to opacify areas outside of the
            # region; note that in MIP mode, will still only show lowest plane
            img_region = np.invert(img_region).astype(float)
            img_region[img_region == 0] = np.nan
        self.img_region = img_region
        self.fn_show_label_3d = fn_show_label_3d
        self.fn_status_bar = fn_status_bar

        # initialize other instance attributes
        self.filename = None
        self.offset = None
        self.roi_size = None
        self.plane = None
        self.max_intens_proj = False
        self.zoom_shift = None
        self.fn_update_coords = None
        self.fn_redraw = None
        self.blobs = None
        self._blobs_coloc_text = None
        self._z_overview = None
        self._channel = None  # list of channel lists
        
        # store DraggableCircles objects to prevent premature garbage collection
        self._draggable_circles = []
        self._circle_last_picked = []
        self._ax_subplots = OrderedDict()  # PlotAxImg for each zoomed plot

        # additional z's above/below
        margin = config.plot_labels[config.PlotLabels.MARGIN]
        if margin is None:
            self._z_planes_padding = 3
        else:
            # assumes x,y,z order
            self._z_planes_padding = libmag.get_if_within(margin, 2, 3)
        print("margin: {}, savefig: {}".format(margin, config.savefig))

    def _show_overview(self, ax_ov, lev, zoom_levels, arrs_3d, cmap_labels,
                       aspect, origin, scaling, max_size):
        """Show overview image with progressive zooming on the ROI for each
        zoom level, displayed in a a :class:`PlotEditor`.
        
        Shifts the zoom based on :attr:`zoom_shift`, defaulting to ``(1, 1)``
        if None.

        Args:
            ax_ov: Subplot axes.
            lev: Zoom level index, where 0 is the original image.
            zoom_levels (List[float]): Sequence of zoom levels.
            arrs_3d (List[:obj:`np.ndarray`]): Sequence of 3D arrays to
                overlay.
            cmap_labels (:obj:`colors.ListedColormap`): Atlas labels colormap.
            aspect (float): Aspect ratio.
            origin (str): Planar orientation, usually either "lower" or None.
            scaling (List[float]): Scaling/spacing in z,y,x.
            max_size (int): Maximum size of either side of the 2D plane shown;
                defaults to None.
        """
        def update_coords(coord, plane):
            # update displayed overview plot for the given coordinates and
            # show preview ROI
            plot_ed.update_coord(coord, show_crosslines=False)
            plot_ed.show_roi(coord[1:], self.roi_size[1::-1], preview=True)
            if self.fn_update_coords:
                # trigger callback with coordinates in z-plane orientation
                coord_zax = libmag.transpose_1d_rev(list(coord), plane)
                self.fn_update_coords(coord_zax)
        
        # main overview image, on which other images may be overlaid
        roi_end = np.add(self.offset, self.roi_size)
        offsets = []  # z,y,x
        sizes = []  # z,y,x
        zoom = zoom_levels[lev]
        if lev > 0:
            # move origin progressively closer with each zoom level,
            # a small fraction less than the offset
            
            # default to shifting origin so that ROI is near upper L corner
            zoom_shift = (1, 1) if self.zoom_shift is None else self.zoom_shift
            ori = np.multiply(
                self.offset[:2],
                np.subtract(zoom, zoom_shift) / zoom).astype(int)
            zoom_shape = np.flipud(arrs_3d[0].shape[1:3])
            # progressively decrease size, zooming in for each level
            size = (zoom_shape / zoom).astype(int)
            end = np.add(ori, size)
            # if ROI exceeds bounds of zoomed plot, shift plot
            for o in range(len(ori)):
                roi_end_padded = roi_end[o] + self._ROI_PADDING
                if end[o] < roi_end_padded:
                    diff = roi_end_padded - end[o]
                    ori[o] += diff
                    end[o] += diff
            # keep the zoomed area within the full 2D image
            for o in range(len(ori)):
                if end[o] > zoom_shape[o]:
                    ori[o] -= end[o] - zoom_shape[o]
            for img_i, img in enumerate(arrs_3d):
                if img is not None:
                    # zoom images based on scaling to main image
                    scale = np.divide(
                        img.shape[1:3], arrs_3d[0].shape[1:3])[::-1]
                    origin_scaled = np.multiply(ori, scale).astype(np.int)
                    end_scaled = np.multiply(end, scale).astype(np.int)
                    offsets.append(origin_scaled[::-1])
                    sizes.append(np.subtract(end_scaled, origin_scaled)[::-1])

        # create a Plot Editor for the overview image
        num_arrs_3d = len(arrs_3d)
        labels_img = None if num_arrs_3d <= 1 else arrs_3d[1]
        img3d_extras = arrs_3d[2:] if num_arrs_3d > 2 else None
        if img3d_extras is not None:
            img3d_extras = [np.array(img) for img in img3d_extras]
        plot_ed = plot_editor.PlotEditor(
            ax_ov, arrs_3d[0], labels_img, cmap_labels,
            self.plane, aspect, origin, update_coords,
            scaling, max_size=max_size, fn_status_bar=self.fn_status_bar,
            img3d_extras=img3d_extras,
            fn_show_label_3d=self.fn_show_label_3d)
        plot_ed.scale_bar = True
        plot_ed.enable_painting = False
        plot_ed.max_intens_proj = self.roi_size[2] if self.max_intens_proj else 0
        update_coords((self._z_overview, *self.offset[1::-1]), self.plane)
        plot_ed.show_roi(self.offset[1::-1], self.roi_size[1::-1])
        if offsets and sizes:
            # zoom toward ROI
            plot_ed.view_subimg(offsets[0], sizes[0])
        self.plot_eds[zoom] = plot_ed
        self._update_overview_title(ax_ov, lev, zoom)
    
    def _update_overview_title(self, ax_ov, lev, zoom):
        """Set title with total zoom including objective and plane number.
        
        Args:
            ax_ov: Subplot axes.
            lev: Zoom level, where 0 is the original image.
            zoom (float): Microscope total zoom. Overridden by
                :attr:`config.zoom` and :attr:`config.magnification` if they
                both exist.

        """
        if config.zoom and config.magnification:
            # calculate total mag from objective zoom and mag
            zoom_components = np.array(
                [config.zoom, config.magnification, zoom]).astype(np.float)
            # use abs since the default mag and zoom were previously -1.0
            tot_zoom = "{}x".format(
                libmag.compact_float(abs(np.prod(zoom_components)), 1))
        elif lev == 0:
            tot_zoom = "original magnification"
        else:
            tot_zoom = "{}x of original".format(zoom)
        plot_support.set_overview_title(
            ax_ov, self.plane, self._z_overview, tot_zoom, lev,
            self.max_intens_proj)
    
    def _redraw(self, event):
        """Trigger :attr:`fn_redraw` if the event was a right button
        double-clck that took place in a Plot Editor.
        
        Args:
            event (:obj:`matplotlib.backend_bases.MouseEvent`): Mouse event.

        """
        if not self.fn_redraw or not event.dblclick or not event.button == 3:
            return
        for ed in self.plot_eds.values():
            if ed.axes == event.inaxes:
                self.fn_redraw()
                break
    
    def plot_2d_stack(self, fn_update_seg, filename, channel,
                      roi_size, offset, mask_in, blobs_cmap,
                      fn_close_listener, border=None, plane=config.PLANE[0],
                      zoom_levels=1, single_roi_row=False,
                      z_level=ZLevels.BOTTOM, roi=None, labels=None,
                      blobs_truth=None, circles=None, mlab_screenshot=None,
                      grid=False, roi_cols=None, fig=None, region_name=None):
        """Shows a figure of 2D plots to compare with the 3D plot.

        Args:
            fn_update_seg (func): Callback when updating a 
                :obj:`DraggableCircle`.
            filename (str): Path to use when saving the plot.
            channel: Channel of the image to display.
            roi_size: List of x,y,z dimensions of the ROI.
            offset: Tuple of x,y,z coordinates of the ROI.
            mask_in: Boolean mask of ``segments`` within the ROI.
            blobs_cmap: Colormap for blobs inside the ROI.
            fn_close_listener: Handle figure close events.
            border: Border dimensions in pixels given as (x, y, z); defaults
                to None.
            plane: The plane to show in each 2D plot, with "xy" to show the
                XY plane (default) and "xz" to show XZ plane.
            zoom_levels (int, List[int]): Number of overview zoom levels to
                include or sequence of zoom multipliers; defaults to 1.
            single_roi_row: True if the ROI-sized plots should be
                displayed on a single row; defaults to False.
            z_level: Position of the z-plane shown in the overview plots,
                based on the Z_LEVELS attribute constant; defaults to
                Z_LEVELS[0].
            roi (:obj:`np.ndarray`): A denoised region of interest for display
                in ROI plots, such as a preprocessed ROI.
                Defaults to None, in which case image5d will be used instead.
            labels (:obj:`np.ndarray`): Segmentation labels of the same shape
                as that of ``image5d``; defaults to None.
            blobs_truth (:obj:`np.ndarray`): Array of blobs to display as
                ground truth; defaults to None.
            circles: :class:``CircleStyles`` enum member; defaults to None.
            mlab_screenshot (:obj:`np.ndarray`): Array from Mayavi screenshot;
                defaults to None.
            grid (bool): True to overlay a grid on all plots.
            roi_cols (int): Number of columns per row to reserve for ROI plots;
                defaults to None, in which case :attr:`ROI_COLS` will be used.
            fig (:obj:`figure.Figure`): Matplotlib figure; defaults to None
                to generate a new figure.
            region_name (str): Name of atlas region for title; defaults to None.
        """
        time_start = time()

        self.filename = filename
        self.offset = offset
        self.roi_size = roi_size
        self.plane = plane
        self._channel = [channel]

        if not roi_cols:
            roi_cols = self.ROI_COLS

        if not np.ndim(zoom_levels):
            # convert scalar to sequence of zoom multipliers for zooming into
            # the ROI in overview plots; scale the zoom to a default of 3x
            # the ROI shape
            size_max = self.image5d.shape[2:4][::-1]
            size_min = np.multiply(roi_size[:2], 3)
            if any(np.greater(size_min, size_max)):
                # fallback to ROI size if default exceeds the full image size
                size_min = roi_size[:2]

            # zoom increasingly faster toward the max zoom for the ROI
            # shape by using a power function scaling output from 1 to max
            # zoom, excluding final value to allow greater zooming with
            # increased zoom levels
            max_zoom = np.amin(np.divide(size_max, size_min))
            zoom_levels = np.power(np.linspace(
                np.power(1 / max_zoom, 1 / 3), 1, zoom_levels, endpoint=False),
                3) * max_zoom
            print("zoom_levels:", zoom_levels, "max_zoom:", max_zoom)

        num_zoom_levels = len(zoom_levels)

        # set up figure
        if fig is None:
            fig = figure.Figure()
        fig.clear()
        self.fig = fig

        # black text with transluscent background the color of the figure
        # background in case the title is a 2D plot
        fig.suptitle(
            ROIEditor._fig_title(
                region_name, os.path.basename(filename), offset, roi_size),
            bbox=dict(
                facecolor="xkcd:silver", edgecolor="none", alpha=0.5))

        # adjust array order based on which plane to show
        border_full = np.copy(border)
        border[2] = 0
        blobs = self.blobs.blobs if self.blobs else None
        if plane == config.PLANE[1]:
            # "xz" planes; flip y-z to give y-planes instead of z
            roi_size = libmag.swap_elements(roi_size, 1, 2)
            offset = libmag.swap_elements(offset, 1, 2)
            border = libmag.swap_elements(border, 1, 2)
            border_full = libmag.swap_elements(border_full, 1, 2)
            if blobs is not None and len(blobs) > 0:
                blobs[:, [0, 1]] = blobs[:, [1, 0]]
        elif plane == config.PLANE[2]:
            # "yz" planes; roll backward to flip x-z and x-y
            roi_size = libmag.roll_elements(roi_size, -1)
            offset = libmag.roll_elements(offset, -1)
            border = libmag.roll_elements(border, -1)
            border_full = libmag.roll_elements(border_full, -1)
            print("orig blobs:\n{}".format(blobs))
            if blobs is not None and len(blobs) > 0:
                # roll forward since segments in zyx order
                blobs[:, [0, 2]] = blobs[:, [2, 0]]
                blobs[:, [1, 2]] = blobs[:, [2, 1]]
                print("rolled blobs:\n{}".format(blobs))
        print("2D border: {}".format(border))

        # mark z-planes to show
        z_start = offset[2]
        z_planes = roi_size[2]
        z_planes = z_planes + self._z_planes_padding * 2

        # position overview at bottom (default), middle, or top of stack
        self._z_overview = z_start  # abs positioning
        if z_level == self.ZLevels.MIDDLE:
            self._z_overview = (2 * z_start + z_planes) // 2
        elif z_level == self.ZLevels.TOP:
            self._z_overview = z_start + z_planes
        print("z_overview: {}".format(self._z_overview))

        # set up images to overlay in overview plots
        arrs3d = [self.image5d[0], self.labels_img]
        if self.img_region is not None:
            arrs3d.append(self.img_region)
        arrs_3d, aspect, origin, scaling = plot_support.setup_images_for_plane(
            plane, arrs3d)
        scaling = config.labels_scaling
        if scaling is not None: scaling = [scaling]
        cmap_labels = None
        if self.labels_img is not None:
            # set up labels image discrete colormap
            cmap_labels = colormaps.setup_labels_cmap(self.labels_img)
        max_sizes = plot_support.get_downsample_max_sizes()
        max_size = max_sizes[plot_support.get_plane_axis(
            plane, get_index=True)] if max_sizes else None

        # plot layout depending on number of z-planes
        if single_roi_row:
            # show all plots in single row
            zoom_plot_rows = 1
            col_remainder = 0
            zoom_plot_cols = z_planes
        else:
            # wrap plots after reaching max, but tolerates additional column
            # if it will fit all the remainder plots from the last row
            zoom_plot_rows = math.ceil(z_planes / roi_cols)
            col_remainder = z_planes % roi_cols
            zoom_plot_cols = roi_cols
            if 0 < col_remainder < zoom_plot_rows:
                zoom_plot_cols += 1
                zoom_plot_rows = math.ceil(z_planes / zoom_plot_cols)
                col_remainder = z_planes % zoom_plot_cols
        # number of columns for top row with overview plots
        top_cols = len(zoom_levels)
        height_ratios = (3, zoom_plot_rows)
        if mlab_screenshot is None:
            main_img_shape = arrs_3d[0].shape[1:]
            if main_img_shape[1] > 2 * main_img_shape[0]:
                # for wide layouts, prioritize the ROI plots, especially
                # if only one overview column
                height_ratios = (1, 1) if top_cols >= 2 else (1, 2)
        else:
            # add column for screenshot
            top_cols += 1
        gs = gridspec.GridSpec(
            2, top_cols, wspace=0.01, hspace=0.01, height_ratios=height_ratios,
            figure=fig, left=0.01, right=0.99, bottom=0.01, top=0.93)

        # overview subplotting
        ax_overviews = []  # overview axes
        self._ax_subplots = OrderedDict()  # zoom plot axes

        def jump(event):
            z_ov = None
            subplots = list(self._ax_subplots.keys())
            if event.inaxes in subplots:
                # right-arrow to jump to z-plane of given zoom plot
                z_ov = (subplots.index(event.inaxes) + z_start
                        - self._z_planes_padding)
            return z_ov

        def scroll_overview(event):
            """Scroll through overview images along their orthogonal axis.

            Args:
                event: Mouse or key event. For mouse events, scroll step sizes
                    will be used for movements. For key events, up/down arrows
                    will be used.
            """
            for edi, plot_ed in enumerate(self.plot_eds.values()):
                plot_ed.scroll_overview(event, only_in_axes=False, fn_jump=jump)
                if edi == 0:
                    # z-plane index should be same for all editors
                    self._z_overview = plot_ed.coord[0]
                self._update_overview_title(
                    plot_ed.axes, edi, zoom_levels[edi])
            update_subplot_border()
            fig.canvas.draw_idle()

        def update_subplot_border():
            # show a colored border around zoomed plot corresponding to
            # overview plots
            for axi, axz in enumerate(self._ax_subplots.keys()):
                if axi + z_start - self._z_planes_padding == self._z_overview:
                    # highlight border
                    axz.patch.set_edgecolor("orange")
                    axz.patch.set_linewidth(3)
                else:
                    # make border invisible
                    axz.patch.set_linewidth(0)

        def key_press(event):
            # respond to key presses
            if event.key == "ctrl+s" or event.key == "cmd+s":
                # support default save shortcuts on multiple platforms;
                # ctrl-s will bring up save dialog from fig, but cmd/win-S
                # will bypass
                self.save_fig(self.get_save_path())
            else:
                # default to scrolling commands for up/down/right arrows
                scroll_overview(event)

        def on_btn_release(event):
            # respond to mouse button presses for DraggableCircle management
            inax = event.inaxes
            print("event key: {}".format(event.key))
            subplots = list(self._ax_subplots.keys())
            if event.key is None:
                # for some reason becomes none if previous event was
                # ctrl combo and this event is control
                pass
            elif event.key == "control" or event.key.startswith("ctrl"):
                blob_channel = None
                if channel:
                    blob_channel = channel[0]
                    num_chls = len(channel)
                    if num_chls > 1:
                        chl_matches = re.search(regex_key_chl, event.key)
                        if chl_matches:
                            # ctrl+n to specify the n-th channel
                            chl = int(chl_matches[0])
                            if chl < num_chls:
                                blob_channel = channel[chl]
                            else:
                                print("selected channel index {} not within"
                                      " range up to index {}"
                                      .format(chl, num_chls - 1))
                                return
                try:
                    axi = subplots.index(inax)
                    if (axi != -1 and self._z_planes_padding <= axi
                            < z_planes - self._z_planes_padding):
                        blob = np.array([[axi - self._z_planes_padding,
                                         event.ydata.astype(int),
                                         event.xdata.astype(int), -5]])
                        blob = detector.format_blobs(blob, blob_channel)
                        detector.shift_blob_abs_coords(blob, offset[::-1])
                        detector.set_blob_confirmed(blob, 1)
                        blob = fn_update_seg(blob[0])
                        # adds a circle to denote the new segment
                        patch = self._plot_circle(
                            inax, blob, self._BLOB_LINEWIDTH, "-",
                            fn_update_seg)
                except ValueError as e:
                    print(e)
                    print("not on a plot to select a point")
                fig.canvas.draw_idle()
            elif event.key == "v":
                _circle_last_picked_len = len(self._circle_last_picked)
                if _circle_last_picked_len < 1:
                    print("No previously picked circle to paste")
                    return
                moved_item = self._circle_last_picked[
                    _circle_last_picked_len - 1]
                circle, move_type = moved_item
                axi = subplots.index(inax)
                dz = axi - self._z_planes_padding - circle.segment[0]
                seg_old = np.copy(circle.segment)
                seg_new = np.copy(circle.segment)
                seg_new[0] += dz
                if move_type == DraggableCircle.CUT:
                    print("Pasting a cut segment")
                    self._draggable_circles.remove(circle)
                    self._circle_last_picked.remove(moved_item)
                    seg_new = fn_update_seg(seg_new, seg_old)
                else:
                    print("Pasting a copied in segment")
                    detector.shift_blob_abs_coords(seg_new, (dz, 0, 0))
                    seg_new = fn_update_seg(seg_new)
                self._plot_circle(
                    inax, seg_new, self._BLOB_LINEWIDTH, None, fn_update_seg)
                fig.canvas.draw_idle()

        # overview images taken from the bottom plane of the offset, with
        # progressively zoomed overview images if set for additional zoom levels
        for level in range(num_zoom_levels):
            ax = fig.add_subplot(gs[0, level])
            ax_overviews.append(ax)
            plot_support.hide_axes(ax)
            self._show_overview(
                ax, level, zoom_levels, arrs_3d, cmap_labels, aspect, origin,
                scaling, max_size)
        
        # attach overview plot navigation handlers: 1) mouse scroll, 2) arrow
        # key, and 3) right-click in zoomed plot to jump to that plane in the
        # overview plots; note that fig/axes lose focus sporadically in lower
        # right canvas on Mac, at which time axes are not associated with
        # key events but are with mouse events
        fig.canvas.mpl_connect("scroll_event", scroll_overview)
        fig.canvas.mpl_connect("key_press_event", key_press)
        fig.canvas.mpl_connect("button_release_event", key_press)
        # fig.canvas.mpl_connect("draw_event", lambda x: print("redraw"))
        
        if self.fn_redraw:
            # handle potential redraws
            fig.canvas.mpl_connect("button_press_event", self._redraw)

        # zoomed-in views of z-planes spanning from just below to just above ROI
        blobs_in = None
        blobs_out = None
        if (circles != self.CircleStyles.NO_CIRCLES and blobs is not None
                and len(blobs) > 0):
            # separate segments inside from outside the ROI
            if mask_in is not None:
                blobs_in = blobs[mask_in]
                blobs_out = blobs[np.invert(mask_in)]
            # separate out truth blobs
            if blobs.shape[1] >= 6:
                if blobs_truth is None:
                    blobs_truth = blobs[blobs[:, 5] >= 0]
                print("blobs_truth:\n{}".format(blobs_truth))
                # non-truth blobs have truth flag unset (-1)
                if blobs_in is not None:
                    blobs_in = blobs_in[blobs_in[:, 5] == -1]
                if blobs_out is not None:
                    blobs_out = blobs_out[blobs_out[:, 5] == -1]
                #print("blobs_in:\n{}".format(blobs_in))

        # selected or newly added patches since difficult to get patch from
        # collection,and they don't appear to be individually editable
        seg_patch_dict = {}

        # sub-gridspec for fully zoomed plots to allow flexible number of cols
        gs_zoomed = gridspec.GridSpecFromSubplotSpec(
            zoom_plot_rows, zoom_plot_cols, gs[1, :], wspace=0.1, hspace=0.1)
        cmap_labels = None
        if labels is not None:
            # background partially transparent to show any mismatch
            cmap_labels = colormaps.get_labels_discrete_colormap(labels, 100)
        # plot the fully zoomed plots
        #zoom_plot_rows = 0 # TESTING: show no fully zoomed plots
        for i in range(zoom_plot_rows):
            # adjust columns for last row to number of plots remaining
            cols = zoom_plot_cols
            if i == zoom_plot_rows - 1 and col_remainder > 0:
                cols = col_remainder
            # show zoomed in plots and highlight one at offset z
            for j in range(cols):
                # z relative to the start of ROI, since blobs are relative to ROI
                z_relative = i * zoom_plot_cols + j - self._z_planes_padding
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

                # show border outlining area that will be saved in verify mode
                show_border = (verify and border[2] <= z_relative
                               < roi_size[2] - border[2])

                # show the zoomed subplot with scale bar for the current z-plane
                ax_z, ax_z_imgs = self.show_subplot(
                    fig, gs_zoomed, i, j, channel, roi_size,
                    zoom_offset, fn_update_seg,
                    blobs_in, blobs_out, blobs_cmap, alpha, z_relative,
                    z == self._z_overview, border_full if show_border else None,
                    plane, roi_show, labels, blobs_truth_z, circles=circles,
                    aspect=aspect, grid=grid, cmap_labels=cmap_labels)
                if (i == 0 and j == 0
                        and config.plot_labels[config.PlotLabels.SCALE_BAR]):
                    plot_support.add_scale_bar(ax_z, plane=plane)
                self._ax_subplots[ax_z] = ax_z_imgs
        update_subplot_border()

        if not circles == self.CircleStyles.NO_CIRCLES:
            # add points that were not segmented by ctrl-clicking on zoom plots
            # as long as not in "no circles" mode
            regex_key_chl = re.compile(r"\+[0-9]+$")
            
            fig.canvas.mpl_connect("button_release_event", on_btn_release)
            # reset circles window flag
            fig.canvas.mpl_connect("close_event", fn_close_listener)

        # show 3D screenshot if available
        if mlab_screenshot is not None:
            img3d = mlab_screenshot
            ax = fig.add_subplot(gs[0, num_zoom_levels])
            # auto to adjust size with less overlap
            ax.imshow(img3d)
            ax.set_aspect(img3d.shape[1] / img3d.shape[0])
            plot_support.hide_axes(ax)
        plt.ion()
        fig.canvas.draw_idle()
        print("2D plot time: {}".format(time() - time_start))
    
    def update_imgs_display(self, imgi, **kwargs):
        """Update images with the given display settings.
        
        Args:
            imgi (int): Index of image group.
            **kwargs: Arguments to pass to the updater.

        Returns:
            :obj:`plot_editor.PlotAxImg`: Updated plotted image.

        """
        # update overview images
        plot_ax_img = super().update_imgs_display(imgi, **kwargs)
        
        if "chl" in kwargs:
            # use channel when getting plotted image
            chl = kwargs["chl"]
            del kwargs["chl"]
        else:
            chl = None
        alpha = kwargs["alpha"] if "alpha" in kwargs else None
        num_subplots = len(self._ax_subplots)
        for i, plot_ax_imgs in enumerate(self._ax_subplots.values()):
            # get zoomed image; halve alpha for ROI padding planes
            img = plot_editor.PlotEditor.get_plot_ax_img(
                plot_ax_imgs, imgi, self._channel, chl)
            alpha_subplot = alpha
            if alpha and (i < self._z_planes_padding
                          or i >= num_subplots - self._z_planes_padding):
                alpha_subplot /= 2
            kwargs["alpha"] = alpha_subplot
            
            # update zoomed image
            plot_editor.PlotEditor.update_plot_ax_img_display(
                img, **kwargs)
        return plot_ax_img
    
    def plot_roi(self, roi, segments, channel, show=True, title=""):
        """Plot ROI as sequence of z-planes containing only the ROI itself.

        Args:
            roi: The ROI image as a 3D array in (z, y, x) order.
            segments: Numpy array of segments to display in the subplot, which
                can be None. Segments are generally given as an (n, 4)
                dimension array, where each segment is in (z, y, x, radius).
                All segments are assumed to be within the ROI for display.
            channel: Channel of the image to display.
            show: True if the plot should be displayed to screen; defaults
                to True.
            title: String used as basename of output file. Defaults to ""
                and only used if :attr:``config.savefig`` is set to a file
                extension.
        """
        fig = plt.figure()
        # fig.suptitle(title)
        # total number of z-planes
        z_planes = roi.shape[0]
        # wrap plots after reaching max, but tolerates additional column
        # if it will fit all the remainder plots from the last row
        zoom_plot_rows = math.ceil(z_planes / self.ROI_COLS)
        col_remainder = z_planes % self.ROI_COLS
        zoom_plot_cols = self.ROI_COLS
        if col_remainder > 0 and col_remainder < zoom_plot_rows:
            zoom_plot_cols += 1
            zoom_plot_rows = math.ceil(z_planes / zoom_plot_cols)
            col_remainder = z_planes % zoom_plot_cols
        roi_size = roi.shape[::-1]
        zoom_offset = [0, 0, 0]
        gs = gridspec.GridSpec(
            zoom_plot_rows, zoom_plot_cols, wspace=0.1, hspace=0.1)
        image5d = importer.roi_to_image5d(roi)

        # plot the fully zoomed plots
        for i in range(zoom_plot_rows):
            # adjust columns for last row to number of plots remaining
            cols = zoom_plot_cols
            if i == zoom_plot_rows - 1 and col_remainder > 0:
                cols = col_remainder
            # show zoomed in plots and highlight one at offset z
            for j in range(cols):
                # z relative to start of ROI, since blobs are relative to ROI
                z = i * zoom_plot_cols + j
                zoom_offset[2] = z

                # shows the zoomed subplot with scale bar for the current
                # z-plane with all segments
                ax_z = self.show_subplot(
                    fig, gs, i, j, image5d, channel, roi_size, zoom_offset,
                    None, segments, None, None, 1.0, z,
                    circles=self.CircleStyles.CIRCLES, roi=roi)
                if (i == 0 and j == 0
                        and config.plot_labels[config.PlotLabels.SCALE_BAR]):
                    plot_support.add_scale_bar(ax_z)
        gs.tight_layout(fig, pad=0.5)
        if show:
            plt.show()
        plot_support.save_fig(title, config.savefig)
    
    def get_save_path(self):
        """Get default figure save based on ROI offset, shape, plane axis,
        z-plane of overview image, and extension based on
        :attr:`config.savefig`, or "png"
        
        Returns:
            str: Figure save path.

        """
        ext = config.savefig if config.savefig else "png"
        return "{}_{}{}.{}".format(plot_support.get_roi_path(
            os.path.basename(self.filename), self.offset, self.roi_size),
            plot_support.get_plane_axis(self.plane),
            self._z_overview, ext)
    
    @staticmethod
    def _fig_title(atlas_region, name, offset, roi_size):
        """Figure title parser.

        Arguments:
            atlas_region: Name of the region in the atlas; if None, the region
                will be ignored.
            offset: (x, y, z) image offset
            roi_size: (x, y, z) region of interest size

        Returns:
            Figure title string.
        """
        region = ""
        if atlas_region is not None:
            region = "{} from ".format(atlas_region)
        # cannot round to decimal places or else tuple will further round
        roi_size_um = np.around(
            np.multiply(roi_size, config.resolutions[0][::-1]))
        series = ""
        if config.series is not None:
            series = " (series {})".format(config.series)
        return "{}{}{} ROI at x={}, y={}, z={}; size {}px ({}{})".format(
            region, name, series, *offset[:3], str(tuple(roi_size)).strip("()"),
            str(tuple(roi_size_um)).strip("()"), u'\u00b5m')

    def show_subplot(self, fig, gs, row, col, channel, roi_size,
                     offset, fn_update_seg, segs_in, segs_out, segs_cmap, alpha,
                     z_relative, highlight=False, border=None, plane="xy",
                     roi=None, labels=None, blobs_truth=None, circles=None,
                     aspect=None, grid=False, cmap_labels=None):
        """Shows subplots of the region of interest.

        Args:
            fig (:obj:`figure.Figure`): Matplotlib figure.
            gs: Gridspec layout.
            row: Row number of the subplot in the layout.
            col: Column number of the subplot in the layout.
            channel: Channel of the image to display.
            roi_size: List of x,y,z dimensions of the ROI.
            offset: Tuple of x,y,z coordinates of the ROI.
            segs_in: Numpy array of segments within the ROI to display in the
                subplot, which can be None. Segments are generally given as an
                ``(n, 4)`` dimension array, where each segment is in
                ``(z, y, x, radius)``.
            segs_out: Subset of segments that are adjacent to rather than
                inside the ROI, which will be drawn in a different style.
                Can be None.
            segs_cmap: Colormap for segments.
            alpha: Opacity level.
            z_relative: Index of the z-plane relative to the start of the ROI.
            highlight: If true, the plot will be highlighted; defaults
                to False.
            border: Border dimensions in pixels given as (x, y, z); defaults
                to None.
            plane: The plane to show in each 2D plot, with "xy" to show the
                XY plane (default) and "xz" to show XZ plane.
            roi: A denoised region of interest, to show in place of image5d
                for the zoomed images. Defaults to None, in which case
                image5d will be used instead.
            labels: Segmentation labels; defaults to None.
            blobs_truth: Truth blobs formatted similarly to ``segs_in``;
                defaults to None.
            circles: :class:``CircleStyles`` enum member; defaults to None.
            aspect: Image aspect; defauls to None.
            grid: True if a grid should be overlaid; defaults to False.
            cmap_labels: :class:``colormaps.DiscreteColormap`` for labels;
                defaults to None.
        """
        def on_motion(event):
            if event.inaxes == ax:
                # update status bar based on position in axes
                self.fn_status_bar(ax.format_coord.get_msg(event))
        
        ax = fig.add_subplot(gs[row, col])
        plot_support.hide_axes(ax)
        size = self.image5d.shape
        # swap columns if showing a different plane
        plane_axis = plot_support.get_plane_axis(plane)
        image5d_shape_offset = 1 if self.image5d.ndim >= 4 else 0
        if plane == config.PLANE[1]:
            # "xz" planes
            size = libmag.swap_elements(size, 0, 1, image5d_shape_offset)
        elif plane == config.PLANE[2]:
            # "yz" planes
            size = libmag.swap_elements(size, 0, 2, image5d_shape_offset)
            size = libmag.swap_elements(size, 0, 1, image5d_shape_offset)
        z = offset[2]
        ax.set_title("{}={}".format(plane_axis, z))
        if border is not None:
            # boundaries of border region, with xy point of corner in first
            # elements and [width, height] in 2nd, allowing flip for yz plane
            border_bounds = np.array(
                [border[0:2],
                [roi_size[0] - 2 * border[0], roi_size[1] - 2 * border[1]]])
        if z < 0 or z >= size[image5d_shape_offset]:
            # draw empty, grey subplot out of image planes just for spacing
            ax_imgs = [[ax.imshow(np.zeros(roi_size[0:2]), alpha=0)]]
        else:
            # show the zoomed in 2D region

            # calculate the region depending on whether given ROI directly and
            # remove time dimension since roi argument does not have it
            if roi is None:
                region = [offset[2],
                          slice(offset[1], offset[1] + roi_size[1]),
                          slice(offset[0], offset[0] + roi_size[0])]
                roi = self.image5d[0]
                #print("region: {}".format(region))
            else:
                region = [z_relative, slice(0, roi_size[1]),
                          slice(0, roi_size[0])]
            # swap columns if showing a different plane
            if plane == config.PLANE[1]:
                region = libmag.swap_elements(region, 0, 1)
            elif plane == config.PLANE[2]:
                region = libmag.swap_elements(region, 0, 2)
                region = libmag.swap_elements(region, 0, 1)
            # get the zoomed region
            if roi.ndim >= 4:
                roi = roi[tuple(region + [slice(None)])]
            else:
                roi = roi[tuple(region)]
            #print("roi shape:", roi.shape)

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

            # show the ROI, which is now a 2D zoomed image
            ax_imgs = [plot_support.imshow_multichannel(
                ax, roi, channel, config.cmaps, aspect, alpha)]
            #print("roi shape: {} for z_relative: {}".format(roi.shape, z_relative))

            # show labels if provided and within ROI
            if labels is not None:
                for i in range(len(labels)):
                    label = labels[i]
                    if z_relative >= 0 and z_relative < label.shape[0]:
                        ax.imshow(
                            label[z_relative], cmap=cmap_labels,
                            norm=cmap_labels.norm)
                        #ax.imshow(label[z_relative]) # showing only threshold

            if ((segs_in is not None or segs_out is not None)
                    and not circles == self.CircleStyles.NO_CIRCLES):

                # shows truth blobs as blue circles
                if blobs_truth is not None:
                    for blob in blobs_truth:
                        ax.add_patch(patches.Circle(
                            (blob[2], blob[1]), radius=blob[3]/2,
                            facecolor=self._TRUTH_COLORS[blob[5]], alpha=0.8))

                segs_in = np.copy(segs_in)
                if circles is None or circles == self.CircleStyles.CIRCLES:
                    # show circles at detection point only mode:
                    # zero radius of all segments outside of current z to
                    # preserve the order of segments for the corresponding
                    # colormap order while hiding outside segments
                    segs_in[segs_in[:, 0] != z_relative, 3] = 0

                if segs_in is not None and segs_cmap is not None:
                    if circles in (self.CircleStyles.REPEAT_CIRCLES,
                                   self.CircleStyles.FULL_ANNOTATION):
                        # repeat circles and full annotation:
                        # show segments from all z's as circles with colored
                        # outlines, gradually decreasing in size when moving
                        # away from the blob's central z-plane
                        z_diff = np.abs(np.subtract(segs_in[:, 0], z_relative))
                        r_orig = np.abs(np.copy(segs_in[:, 3]))
                        segs_in[:, 3] = np.subtract(
                            r_orig, np.divide(z_diff, 3))
                        # make circles below 90% of their original radius
                        # invisible but not removed to preserve their
                        # corresponding colormap index
                        segs_in[np.less(
                            segs_in[:, 3], np.multiply(r_orig, 0.9)), 3] = 0
                    # show colored, non-pickable circles
                    segs_color = segs_in
                    if circles == self.CircleStyles.FULL_ANNOTATION:
                        # zero out circles from other z's in full annotation
                        # mode to minimize crowding and highlight center circle
                        segs_color = np.copy(segs_in)
                        segs_color[segs_color[:, 0] != z_relative, 3] = 0
                    collection = self._circle_collection(
                        segs_color, segs_cmap.astype(float) / 255.0, "none",
                        self._BLOB_LINEWIDTH)
                    ax.add_collection(collection)

                # segments outside the ROI shown in black dotted line only for
                # their corresponding z
                segs_out_z = None
                if segs_out is not None:
                    segs_out_z = segs_out[segs_out[:, 0] == z_relative]
                    collection_adj = self._circle_collection(
                        segs_out_z, "k", "none", self._BLOB_LINEWIDTH)
                    collection_adj.set_linestyle("--")
                    ax.add_collection(collection_adj)

                # for planes within ROI, overlay segments with dotted line
                # patch and make pickable for verifying the segment
                segments_z = segs_in[segs_in[:, 3] > 0] # full annotation
                if circles == self.CircleStyles.FULL_ANNOTATION:
                    # when showing full annotation, show all segments in the
                    # ROI with adjusted radii unless radius is <= 0
                    for i in range(len(segments_z)):
                        seg = segments_z[i]
                        if seg[0] != z_relative:
                            # add segments outside of plane to Visualizer table
                            # since they have been added to the plane,
                            # adjusting rel and abs z coords to the given plane
                            z_diff = z_relative - seg[0]
                            seg[0] = z_relative
                            detector.shift_blob_abs_coords(
                                segments_z[i], (z_diff, 0, 0))
                            segments_z[i] = fn_update_seg(seg)
                else:
                    # apply only to segments in their current z
                    segments_z = segs_in[segs_in[:, 0] == z_relative]
                    if segs_out_z is not None:
                        segs_out_z_confirmed = segs_out_z[
                            detector.get_blob_confirmed(segs_out_z) == 1]
                        if len(segs_out_z_confirmed) > 0:
                            # include confirmed blobs; TODO: show contextual
                            # circles in adjacent planes?
                            segments_z = np.concatenate(
                                (segments_z, segs_out_z_confirmed))
                            print("segs_out_z_confirmed:\n{}"
                                  .format(segs_out_z_confirmed))
                if segments_z is not None:
                    # show pickable circles
                    for seg in segments_z:
                        self._plot_circle(
                            ax, seg, self._BLOB_LINEWIDTH, None, fn_update_seg)
                
                if (self.blobs is not None
                        and self.blobs.blob_matches is not None):
                    # show blob matches by corresponding number labels
                    for i, match in enumerate(self.blobs.blob_matches):
                        for j, blob in enumerate((match.blob1, match.blob2)):
                            if blob[0] != z_relative: continue
                            # add label with number; italicize if 1st blob
                            style = "italic" if j == 0 else "normal"
                            ax.text(blob[2], blob[1], i, color="k",
                                    alpha=0.8, style=style,
                                    horizontalalignment="center",
                                    verticalalignment="center")

            # adds a simple border to highlight the border of the ROI
            if border is not None:
                #print("border: {}, roi_size: {}".format(border, roi_size))
                ax.add_patch(patches.Rectangle(border_bounds[0],
                                               border_bounds[1, 0],
                                               border_bounds[1, 1],
                                               fill=False, edgecolor="yellow",
                                               linestyle="dashed",
                                               linewidth=self._BLOB_LINEWIDTH))
            
            if self.fn_status_bar:
                # set up status bar pixel display for mouseover
                imgs2d = [roi if channel is None or len(roi.shape) < 3
                          else roi[..., tuple(channel)]]
                ax.format_coord = pixel_display.PixelDisplay(
                    imgs2d, ax_imgs, offset=offset[1::-1])
                fig.canvas.mpl_connect("motion_notify_event", on_motion)
        plot_ax_imgs = None
        if ax_imgs and ax_imgs[0]:
            plot_ax_imgs = [[plot_editor.PlotAxImg(img) for img in imgs]
                            for imgs in ax_imgs]
        return ax, plot_ax_imgs

    def _circle_collection(self, segments, edgecolor, facecolor, linewidth):
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
            seg_patches.append(
                patches.Circle((seg[2], seg[1]), radius=self._get_radius(seg)))
        collection = PatchCollection(seg_patches)
        collection.set_edgecolor(edgecolor)
        collection.set_facecolor(facecolor)
        collection.set_linewidth(linewidth)
        return collection

    def _plot_circle(self, ax, segment, linewidth, linestyle, fn_update_seg,
                     alpha=0.5, edgecolor="w"):
        """Create and draw a DraggableCircle from the given segment.

        Args:
            ax: Matplotlib axes.
            segment: Numpy array of segments, generally as an (n, 4)
                dimension array, where each segment is in (z, y, x, radius).
            linewidth: Edge line width.
            linestyle: Edge line style.
            fn_update_seg: Function to call from DraggableCircle.
            alpha: Alpha transparency level; defaults to 0.5.
            edgecolor: String of circle edge color; defaults to "w" for white.

        Returns:
            The DraggableCircle object.
        """
        channel = detector.get_blob_channel(segment)
        facecolor = DraggableCircle.BLOB_COLORS[
            detector.get_blob_confirmed(segment)]
        if linestyle is None:
            linestyle = self._BLOB_LINESTYLES[channel]
        circle = patches.Circle(
            (segment[2], segment[1]), radius=self._get_radius(segment),
            edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth,
            linestyle=linestyle, alpha=alpha)
        ax.add_patch(circle)
        #print("added circle: {}".format(circle))
        draggable_circle = DraggableCircle(
            circle, segment, fn_update_seg, self._circle_last_picked, facecolor)
        draggable_circle.connect()
        self._draggable_circles.append(draggable_circle)
        return draggable_circle

    def _get_radius(self, seg):
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
    
    def show_colocalized_blobs(self, visible):
        """Show blob co-localization by overlaying text showing all the
        channels with signal at each blob's position.
        
        Args:
            visible (bool): True to make the co-localization text visible.

        """
        if self._blobs_coloc_text:
            for text in self._blobs_coloc_text:
                # change existing label's visibility
                text.set_visible(visible)
        
        else:
            if (not visible or self.blobs is None or self.blobs.blobs is None
                    or self.blobs.colocalizations is None):
                return
            # show labels for each blob
            self._blobs_coloc_text = []
            for i, ax in enumerate(self._ax_subplots.keys()):
                # get blobs at given z-val relative to ROI, shifting  for
                # plots in the padding region above and below the ROI
                z = i - self._z_planes_padding
                if i < 0: continue
                mask = self.blobs.blobs[:, 0] == z
                blobs = self.blobs.blobs[mask]
                colocs = self.blobs.colocalizations[mask]
                
                for j, (blob, coloc) in enumerate(zip(blobs, colocs)):
                    # overlay the channels with signal at given blob position
                    self._blobs_coloc_text.append(ax.text(
                        blob[2], blob[1],
                        ",".join([str(c) for c in np.where(coloc > 0)[0]]),
                        color="C{}".format(
                            int(detector.get_blob_channel(blob))),
                        alpha=0.8, horizontalalignment="center",
                        verticalalignment="center"))
        self.fig.canvas.draw_idle()
    
    def set_circle_visibility(self, visible):
        """Set the visibility of detection circles.
        
        Args:
            visible (bool): True to make the circles visible, False for
                invisibility.

        """
        for circle in self._draggable_circles:
            # change the visibility of selectable circles
            circle.circle.set_visible(visible)
        for ax in self._ax_subplots.keys():
            for collection in ax.collections:
                # change the visibility of colored circle collections
                collection.set_visible(visible)
        self.fig.canvas.draw_idle()
