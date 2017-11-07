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

SEG_LINEWIDTH = 2.0
ZOOM_COLS = 9
Z_LEVELS = ("bottom", "middle", "top")
PLANE = ("xy", "xz", "yz")
plane = None
CIRCLES = ("Circles", "Repeat circles", "No circles")

segs_color_dict = {
    -1: None,
    0: "r",
    1: "g",
    2: "y"
}

truth_color_dict = {
    -1: None,
    0: "m",
    1: "b"
}

def _get_radius(seg):
    """Gets the radius for a segments, defaulting to 5 if the segment's
    radius is close to 0.
    
    Args:
        seg: The segments, where seg[3] is the radius.
    
    Returns:
        The radius, defaulting to 0 if the given radius value is close 
        to 0 by numpy.allclose.
    """
    return 5 if np.allclose(seg[3], 0) else seg[3]

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

def add_scale_bar(ax):
    """Adds a scale bar to the plot.
    
    Uses the x resolution value and assumes that it is in microns per pixel.
    
    Args:
        ax: The plot that will show the bar.
    """
    scale_bar = ScaleBar(detector.resolutions[0][2], u'\u00b5m', SI_LENGTH, 
                         box_alpha=0, color="w", location=3)
    ax.add_artist(scale_bar)

def show_subplot(fig, gs, row, col, image5d, channel, roi_size, offset, segments, 
                 segments_z, segs_cmap, alpha, highlight=False, border=None, 
                 segments_adj=None, plane="xy", roi=None, z_relative=-1,
                 labels=None, blobs_truth=None, circles=None, aspect=None, 
                 grid=False):
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
        segments: Numpy array of segments to display in the subplot, which 
            can be None. Segments are generally given as an (n, 4)
            dimension array, where each segment is in (z, y, x, radius).
        segments_z: Subset of segments to highlight in a separate patch
            collection.
        segs_cmap: Colormap for segments.
        alpha: Opacity level.
        highlight: If true, the plot will be highlighted; defaults 
            to False.
        segments_adj: Subset of segments that are adjacent to rather than
            inside the ROI, which will be drawn in a differen style.
            Defaults to None.
        plane: The plane to show in each 2D plot, with "xy" to show the 
            XY plane (default) and "xz" to show XZ plane.
        roi: A denoised region of interest, to show in place of image5d for the
            zoomed images. Defaults to None, in which case image5d will be
            used instead.
        z_relative: Index of the z-plane relative to the start of the ROI, used
            when roi is given and ignored otherwise. Defaults to -1.
    """
    ax = plt.subplot(gs[row, col])
    _hide_axes(ax)
    size = image5d.shape
    # swap columns if showing a different plane
    plane_axis = "z"
    if plane == PLANE[1]:
        # "xz" planes
        size = lib_clrbrain.swap_elements(size, 0, 1, 1 if image5d.ndim >= 4 else 0)
        plane_axis = "y"
    elif plane == PLANE[2]:
        # "yz" planes
        size = lib_clrbrain.swap_elements(size, 0, 2, 1 if image5d.ndim >= 4 else 0)
        size = lib_clrbrain.swap_elements(size, 0, 1, 1 if image5d.ndim >= 4 else 0)
        plane_axis = "x"
    z = offset[2]
    ax.set_title("{}={}".format(plane_axis, z))
    collection_z = None
    if border is not None:
        # boundaries of border region, with xy point of corner in first 
        # elements and [width, height] in 2nd, allowing flipping for yz plane
        border_bounds = np.array(
            [border[0:2], 
            [roi_size[0] - 2 * border[0], roi_size[1] - 2 * border[1]]])
    if z < 0 or z >= size[1]:
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
        
        if not circles == CIRCLES[2].lower():
            segs = segments
            if circles is None or circles == CIRCLES[0].lower():
                # zero radius of all segments outside of current z to preserve 
                # the order of segments for the corresponding colormap order 
                # while hiding outside segments
                segs = np.copy(segs)
                segs[segs[:, 0] != z_relative] = 0
            
            if circles == CIRCLES[1].lower():
                # overlays segments in adjacent regions with dashed line patch
                if segments_adj is not None:
                    collection_adj = _circle_collection(
                        segments_adj, "k", "none", SEG_LINEWIDTH)
                    collection_adj.set_linestyle("--")
                    ax.add_collection(collection_adj)
                
            # show segments as patches
            if segments is not None and segs_cmap is not None:
                collection = _circle_collection(
                    segs, segs_cmap.astype(float) / 255.0, "none", 
                    SEG_LINEWIDTH)
                ax.add_collection(collection)
            
            # overlays segments in current z with dotted line patch and makes
            # pickable for verifying the segment
            if segments_z is not None:
                collection_z = _circle_collection(
                    segments_z, "w", "none", SEG_LINEWIDTH)
                collection_z.set_linestyle(":")
                collection_z.set_picker(5)
                ax.add_collection(collection_z)
            
            # shows truth blobs as small, solid circles
            if blobs_truth is not None:
                for blob in blobs_truth:
                    ax.add_patch(patches.Circle((blob[2], blob[1]), radius=3, 
                                           facecolor=truth_color_dict[blob[5]], 
                                           alpha=1))
        
        # adds a simple border to highlight the border of the ROI
        if border is not None:
            #print("border: {}, roi_size: {}".format(border, roi_size))
            ax.add_patch(patches.Rectangle(border_bounds[0], 
                                           border_bounds[1, 0], 
                                           border_bounds[1, 1], 
                                           fill=False, edgecolor="yellow",
                                           linestyle="dashed"))
        
    return ax, collection_z

def plot_2d_stack(vis, title, filename, image5d, channel, roi_size, offset, segments, 
                  segs_cmap, border=None, plane="xy", padding_stack=None,
                  zoom_levels=2, single_zoom_row=False, z_level=Z_LEVELS[0], 
                  roi=None, labels=None, blobs_truth=None, circles=None, 
                  mlab_screenshot=None, grid=False):
    """Shows a figure of 2D plots to compare with the 3D plot.
    
    Args:
        title: Figure title.
        image5d: Full Numpy array of the image stack.
        channel: Channel of the image to display.
        roi_size: List of x,y,z dimensions of the ROI.
        offset: Tuple of x,y,z coordinates of the ROI.
        segments: Numpy array of segments to display in the subplot, which 
            can be None. Segments are generally given as an (n, 4)
            dimension array, where each segment is in (z, y, x, radius).
            This array can include adjacent segments as well.
        segs_cmap: Colormap for segments.
        border: Border dimensions in pixels given as (x, y, z); defaults
            to None.
        plane: The plane to show in each 2D plot, with "xy" to show the 
            XY plane (default) and "xz" to show XZ plane.
        padding: The amount of padding in pixels, defaulting to the 
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
        zoom_plot_rows = math.ceil(z_planes / ZOOM_COLS)
        col_remainder = z_planes % ZOOM_COLS
        zoom_plot_cols = ZOOM_COLS
        if col_remainder > 0 and col_remainder < zoom_plot_rows:
            zoom_plot_cols += 1
            zoom_plot_rows = math.ceil(z_planes / zoom_plot_cols)
            col_remainder = z_planes % zoom_plot_cols
    #top_rows = 3 if zoom_plot_rows > 1 else 3
    gs = gridspec.GridSpec(2, zoom_levels, wspace=0.7, hspace=0.4,
                           height_ratios=[3, zoom_plot_rows])
    
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
        # show the zoomed 2D image along with rectangle showing ROI
        ax.imshow(img2d_zoom, cmap=colormap_2d, aspect=aspect, origin=origin)
        ax.add_patch(patches.Rectangle(patch_offset, roi_size[0], roi_size[1], 
                                       fill=False, edgecolor="yellow"))
        add_scale_bar(ax)
    
    # zoomed-in views of z-planes spanning from just below to just above ROI
    #print("rows: {}, cols: {}, remainder: {}"
    #      .format(zoom_plot_rows, zoom_plot_cols, col_remainder))
    collection_z_list = []
    segments_z_list = []
    ax_z_list = []
    segs_out = None
    # separate out truth blobs
    if segments.shape[1] >= 6:
        if blobs_truth is None:
            blobs_truth = segments[segments[:, 5] >= 0]
        print("blobs_truth:\n{}".format(blobs_truth))
        segments = segments[segments[:, 5] == -1]
    # finds adjacent segments, outside of the ROI
    if segments is not None:
        mask_in = np.all([segments[:, 0] >= border[2], segments[:, 0] < roi_size[2] - border[2],
                          segments[:, 1] >= border[1], segments[:, 1] < roi_size[1] - border[1],
                          segments[:, 2] >= border[0], segments[:, 2] < roi_size[0] - border[0]], 
                         axis=0)
        segs_out = segments[np.invert(mask_in)]
        print("segs_out:\n{}".format(segs_out))
        
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
            
            # collects the segments within the given z-plane
            segments_z = None
            if segments is not None:
                segments_z = segments[segments[:, 0] == z_relative]
            segments_z_list.append(segments_z)
            
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
            ax_z, collection_z = show_subplot(
                fig, gs_zoomed, i, j, image5d, channel, roi_size, zoom_offset, 
                segments, segments_z, segs_cmap, alpha, z == z_overview, 
                border_full if show_border else None, segs_out, plane, roi_show, 
                z_relative, labels, blobs_truth_z, circles=circles, 
                aspect=aspect, grid=grid)
            if i == 0 and j == 0:
                add_scale_bar(ax_z)
            collection_z_list.append(collection_z)
            ax_z_list.append(ax_z)
            
            # restores saved segment markings as patches, which are pickable
            # from their corresponding segments within their collection
            if circles is None or circles == CIRCLES[0].lower():
                segi = 0
                for seg in segments_z:
                    if seg[4] != -1:
                        key = "{}-{}".format(len(collection_z_list) - 1, segi)
                        #print("key: {}".format(key))
                        patch = patches.Circle(
                            (seg[2], seg[1]), radius=_get_radius(seg), 
                            facecolor=segs_color_dict[seg[4]], alpha=0.5)
                        ax_z.add_patch(patch)
                        seg_patch_dict[key] = patch
                    segi += 1
    
    def _force_seg_refresh(i):
       """Triggers table update by either selecting and reselected the segment
       or vice versa.
       
       Args:
           i: The element in vis.segs_selected, which is simply an index to
              the segment in vis.segments.
       """
       if i in vis.segs_selected:
           vis.segs_selected.remove(i)
           vis.segs_selected.append(i)
       else:
           vis.segs_selected.append(i)
           vis.segs_selected.remove(i)
    
    # record selected segments in the Visualization segments table
    def on_pick(event):
        # ignore ctrl-clicks since used elsewhere
        if event.mouseevent.key == "control":
            return
        if isinstance(event.artist, PatchCollection):
            # segments_z_list is linked to collection list
            collection = event.artist
            collectioni = collection_z_list.index(collection)
            if collection != -1:
                # patch index is linked to segments_z_list
                seg = segments_z_list[collectioni][event.ind[0]]
                segi = np.where((vis.segments == seg).all(axis=1))
                if len(segi) > 0:
                    # must take from vis rather than saved copy in case user 
                    # manually updates the table
                    i = segi[0][0]
                    #seg[4] = 1 if event.mouseevent.button == 1 else 0
                    key = "{}-{}".format(collectioni, event.ind[0])
                    print("key: {}".format(key))
                    if seg[4] == -1:
                        # 1st click selects, which shows a filled green circle
                        # and adds to selected segments list
                        seg[4] = 1
                        if key not in seg_patch_dict:
                            patch = patches.Circle((seg[2], seg[1]), 
                                                   radius=_get_radius(seg), 
                                                   facecolor="g", alpha=0.5)
                            collection.axes.add_patch(patch)
                            seg_patch_dict[key] = patch
                        if not i in vis.segs_selected:
                            vis.segs_selected.append(i)
                    elif seg[4] == 1:
                        # 2nd click changes to yellow circle, setting seg as "maybe"
                        seg[4] = 2
                        seg_patch_dict[key].set_facecolor("y")
                        _force_seg_refresh(i)
                    elif seg[4] == 2:
                        # 3rd click changes to red circle, verifying as not a seg
                        seg[4] = 0
                        seg_patch_dict[key].set_facecolor("r")
                        _force_seg_refresh(i)
                    elif seg[4] == 0:
                        seg[4] = -1
                        # 4th click unselects, which removes from selected
                        # list and removes filled patch
                        _force_seg_refresh(i)
                        if key in seg_patch_dict:
                            seg_patch_dict[key].remove()
                            del seg_patch_dict[key]
                    vis.segments[segi[0][0]] = seg
                    _force_seg_refresh(i)
                print("picked segment: {}".format(seg))
        elif isinstance(event.artist, patches.Circle):
            # new patches added outside of collections
            i = list(seg_patch_dict.keys())[list(seg_patch_dict.values()).index(event.artist)]
            seg = vis.segments[i]
            if seg[4] == 1:
                # 2nd click changes to yellos circle, setting seg as "maybe"
                seg[4] = 2
                event.artist.set_facecolor("y")
            elif seg[4] == 2:
                # 3rd click changes to red circle, verifying as not a seg
                seg[4] = 0
                event.artist.set_facecolor("r")
            elif seg[4] == 0:
                seg[4] = -1
                # 4th click to unselect, which removes from selected
                # list and removes filled patch
                del seg_patch_dict[i]
                event.artist.remove()
                vis.segs_selected.remove(i)
            vis.segments[i] = seg
            _force_seg_refresh(i)
       
    fig.canvas.mpl_connect("pick_event", on_pick)
    
    # add points that were not segmented by ctrl-clicking on zoom plots
    def on_btn_release(event):
        ax = event.inaxes
        if event.key == "control":
            try:
                axi = ax_z_list.index(ax)
                if (axi != -1 and axi >= z_planes_padding 
                    and axi < z_planes - z_planes_padding):
                    
                    seg = np.array([[axi - z_planes_padding, 
                                     event.ydata.astype(int), 
                                     event.xdata.astype(int), 0.0, 1, -1]])
                    seg = np.concatenate(
                        (seg, np.add(seg[:, :3], offset[::-1])), axis=1)
                    print("added segment: {}".format(seg))
                    # concatenate for in-place array update, though append
                    # and re-assigning also probably works
                    vis.segments = np.concatenate((vis.segments, seg))
                    # create a new copy rather than appending to trigger a
                    # full update; otherwise, only last entry gets selected
                    segsi = vis.segments.shape[0] - 1
                    vis.segs_selected = (vis.segs_selected + [segsi])
                    # adds a circle to denote the new segment
                    patch = patches.Circle((seg[0][2], seg[0][1]), radius=5, 
                                           facecolor="g", alpha=0.5, picker=5)
                    seg_patch_dict[segsi] = patch
                    ax.add_patch(patch)
                    '''# shows surrounding area of adding blob; reqs ROI
                    if roi is not None:
                        detector.show_blob_surroundings(seg, roi, 6)
                    '''
            except ValueError as e:
                print(e)
                print("not on a plot to select a point")
       
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
    fig = plt.figure()
    posi = 1
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
        legend = plt.legend(handles=lines, loc=posi, title=group)
        plt.gca().add_artist(legend)
        posi += 1
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.2])
    plt.ylim([0.0, 1.2])
    plt.xlabel("False Discovery Rate")
    plt.ylabel("Sensitivity")
    plt.title("ROC for {}".format(name))
    plt.show()

def _show_overlay(ax, img, plane_i, cmap, aspect=1.0, alpha=1.0, title=None):
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
    if plane == PLANE[1]:
        # xz plane
        img_2d = img[:, plane_i]
    elif plane == PLANE[2]:
        # yz plane, which requires a rotation
        img_2d = img[:, :, plane_i]
        img_2d = np.swapaxes(img_2d, 0, 1)
        aspect = 1 / aspect
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
                      cmap_atlas, cmap_labels, translation=None, title=None):
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
    """
    fig = plt.figure()
    fig.suptitle(title)
    # give extra space to the first row since the atlas is often larger
    gs = gridspec.GridSpec(2, 3, height_ratios=[3, 2])
    resolution = detector.resolutions[0]
    #size_ratio = np.divide(atlas_reg.shape, exp.shape)
    aspect = 1.0
    z = 0
    atlas_z = 0
    if plane == PLANE[1]:
        # xz plane
        aspect = resolution[0] / resolution[2]
        z = exp.shape[1] // 3
        if translation is None:
            atlas_z = atlas.shape[1] // 3
        else:
            atlas_z = int(z - translation[1])
    elif plane == PLANE[2]:
        # yz plane
        aspect = resolution[0] / resolution[1]
        z = exp.shape[2] // 3
        if translation is None:
            atlas_z = atlas.shape[2] // 3
        else:
            # TODO: not sure why needs to be addition here
            atlas_z = int(z + translation[2])
    else:
        # xy plane (default)
        aspect = resolution[1] / resolution[2]
        z = exp.shape[0] // 3
        if translation is None:
            atlas_z = atlas.shape[0] // 3
        else:
            atlas_z = int(z - translation[0])
    print("z: {}, atlas_z: {}, aspect: {}".format(z, atlas_z, aspect))
    
    # experimental image and atlas
    _show_overlay(plt.subplot(gs[0, 0]), exp, z, cmap_exp, aspect, 
                              title="Experiment")
    _show_overlay(plt.subplot(gs[0, 1]), atlas, atlas_z, cmap_atlas, alpha=0.5, 
                              title="Atlas")
    
    # atlas overlaid onto experiment
    ax = plt.subplot(gs[0, 2])
    _show_overlay(ax, exp, z, cmap_exp, aspect, title="Registered")
    _show_overlay(ax, atlas_reg, z, cmap_atlas, aspect, 0.5)
    
    # labels overlaid onto atlas
    ax = plt.subplot(gs[1, 0])
    _show_overlay(ax, atlas_reg, z, cmap_atlas, aspect, title="Labeled atlas")
    _show_overlay(ax, labels_reg, z, cmap_labels, aspect, 0.5)
    
    # labels overlaid onto exp
    ax = plt.subplot(gs[1, 1])
    _show_overlay(ax, exp, z, cmap_exp, aspect, title="Labeled experiment")
    _show_overlay(ax, labels_reg, z, cmap_labels, aspect, 0.5)
    
    # all overlaid
    ax = plt.subplot(gs[1, 2])
    _show_overlay(ax, exp, z, cmap_exp, aspect, title="All overlaid")
    _show_overlay(ax, atlas_reg, z, cmap_atlas, aspect, 0.5)
    _show_overlay(ax, labels_reg, z, cmap_labels, aspect, 0.3)
    
    if title is None:
        title = "Image Overlays"
    gs.tight_layout(fig)
    if savefig is not None:
        plt.savefig(title + "." + savefig)
    plt.show()

if __name__ == "__main__":
    print("Testing plot_2d...")
    stats_dict = { 
        "test1": (np.array([[5, 4, 3], [8, 3, 4]]), [10, 20]),
        "test2": (np.array([[1225, 1200, 95], [235, 93, 230]]), [25, 34])
    }
    plot_roc(stats_dict, "Testing ROC")
    