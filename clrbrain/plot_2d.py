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

import math
import numpy as np
from mayavi import mlab
from tvtk.pyface.scene_model import SceneModelError
from matplotlib import pyplot as plt, cm
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH

from clrbrain import detector
from clrbrain import config

colormap_2d = cm.inferno
#colormap_2d = cm.gray
savefig = None
verify = False
# TODO: may want to base on scaling factor instead
padding = (5, 5, 3) # human (x, y, z) order

SEG_LINEWIDTH = 2.0
ZOOM_COLS = 8
Z_LEVELS = ("bottom", "middle", "top")

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
    
    Params:
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
    
    Params:
        ax: The plot that will show the bar.
    """
    scale_bar = ScaleBar(detector.resolutions[0][2], u'\u00b5m', SI_LENGTH, 
                         box_alpha=0, color="w", location=3)
    ax.add_artist(scale_bar)

def _swap_elements(arr, axis0, axis1, offset=0):
    axis0 += offset
    axis1 += offset
    check_tuple = isinstance(arr, tuple)
    if check_tuple:
        arr = list(arr)
    arr[axis0], arr[axis1] = arr[axis1], arr[axis0]
    if check_tuple:
        arr = tuple(arr)
    return arr

def show_subplot(fig, gs, row, col, image5d, channel, roi_size, offset, segments, 
                 segments_z, segs_cmap, alpha, highlight=False, border=None, 
                 segments_adj=None, plane="xy", roi=None, z_relative=-1,
                 labels=None, blobs_truth=None):
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
    if plane == "xz":
        size = _swap_elements(size, 0, 1, 1 if image5d.ndim >= 4 else 0)
    z = offset[2]
    ax.set_title("z={}".format(z))
    collection_z = None
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
        if plane == "xz":
            region = _swap_elements(region, 1, 2)
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
            ax.contour(labels[z_relative])
            #ax.imshow(labels[z_relative])
        
        # highlight borders of z plane at bottom of ROI
        if highlight:
            for spine in ax.spines.values():
                spine.set_edgecolor("yellow")
        plt.imshow(roi, cmap=colormap_2d, alpha=alpha)
        
        # draws all segments as patches
        if segments is not None and segs_cmap is not None:
            collection = _circle_collection(segments, 
                                            segs_cmap.astype(float) / 255.0,
                                            "none", SEG_LINEWIDTH)
            ax.add_collection(collection)
        
        # overlays segments in adjacent regions with dashed line patch
        if segments_adj is not None:
            collection_adj = _circle_collection(segments_adj, "k", "none", SEG_LINEWIDTH)
            collection_adj.set_linestyle("--")
            ax.add_collection(collection_adj)
        
        # overlays segments in current z with dotted line patch and makes
        # pickable for verifying the segment
        if segments_z is not None:
            collection_z = _circle_collection(segments_z, "w", "none", SEG_LINEWIDTH)
            collection_z.set_linestyle(":")
            collection_z.set_picker(5)
            ax.add_collection(collection_z)
        
        if blobs_truth is not None:
            for blob in blobs_truth:
                ax.add_patch(patches.Circle((blob[2], blob[1]), radius=3, 
                                       facecolor=truth_color_dict[blob[5]], 
                                       alpha=1))
        
        # adds a simple border to highlight the bottom of the ROI
        if border is not None:
            ax.add_patch(patches.Rectangle(border[0:2], 
                                           roi_size[0] - 2 * border[0], 
                                           roi_size[1] - 2 * border[1], 
                                           fill=False, edgecolor="yellow",
                                           linestyle="dashed"))
        
    return ax, collection_z

def plot_2d_stack(vis, title, image5d, channel, roi_size, offset, segments, 
                  segs_cmap, border=None, plane="xy", padding_stack=None,
                  zoom_levels=2, single_zoom_row=False, z_level=Z_LEVELS[0], 
                  roi=None, labels=None, blobs_truth=None):
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
    fig = plt.figure()
    # black text with transluscent background the color of the figure
    # background in case the title is a 2D plot
    fig.suptitle(title, color="black", 
                 bbox=dict(facecolor=fig.get_facecolor(), edgecolor="none", 
                           alpha=0.5))
    
    # adjust array order based on which plane to show
    aspect = 1 # aspect ratio
    if plane == "xz":
        roi_size = _swap_elements(roi_size, 1, 2)
        offset = _swap_elements(offset, 1, 2)
    
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
    if plane == "xz":
        # TODO: base on scaling vs image shape vs nothing?
        #aspect = detector.resolutions[0, 0] / detector.resolutions[0, 2]
        aspect = float(image5d.shape[3]) / image5d.shape[1]
        print(aspect)
        segments[:, [0, 1]] = segments[:, [1, 0]]
        if image5d.ndim >= 5:
            img2d = image5d[0, :, z_overview, :, channel]
        elif image5d.ndim == 4:
            img2d = image5d[0, :, z_overview, :]
        else:
            img2d = image5d[:, z_overview, :]
    else:
        # defaults to "xy"
        if image5d.ndim >= 5:
            img2d = image5d[0, z_overview, :, :, channel]
        elif image5d.ndim == 4:
            img2d = image5d[0, z_overview, :, :]
        else:
            img2d = image5d[z_overview, :, :]
    
    # plot layout depending on number of z-planes
    if single_zoom_row:
        zoom_plot_rows = 1
        col_remainder = 0
        zoom_plot_cols = z_planes
    else:
        zoom_plot_rows = math.ceil(z_planes / ZOOM_COLS)
        col_remainder = z_planes % ZOOM_COLS
        zoom_plot_cols = ZOOM_COLS
    #top_rows = 3 if zoom_plot_rows > 1 else 3
    gs = gridspec.GridSpec(2, zoom_levels, wspace=0.7, hspace=0.4,
                           height_ratios=[3, zoom_plot_rows])
    
    # overview image, with bottom of offset shown as rectangle
    overview_cols = zoom_plot_cols // zoom_levels
    for i in range(zoom_levels - 1):
        ax = plt.subplot(gs[0, i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img2d_zoom = img2d
        patch_offset = offset[0:2]
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
        ax.imshow(img2d_zoom, cmap=colormap_2d, aspect=aspect)
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
        mask_in = np.all([segments[:, 0] >= 0, segments[:, 0] < roi_size[2],
                          segments[:, 1] >= 0, segments[:, 1] < roi_size[1],
                          segments[:, 2] >= 0, segments[:, 2] < roi_size[0]], 
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
            ax_z, collection_z = show_subplot(fig, gs_zoomed, i, j, image5d, 
                                              channel, roi_size,
                                              zoom_offset, segments, segments_z, 
                                              segs_cmap, alpha, z == z_overview,
                                              border if show_border else None,
                                              segs_out, plane, roi_show, 
                                              z_relative, labels, blobs_truth_z)
            if i == 0 and j == 0:
                add_scale_bar(ax_z)
            collection_z_list.append(collection_z)
            ax_z_list.append(ax_z)
            
            # restores saved segment markings as patches, which are pickable
            # from their corresponding segments within their collection
            segi = 0
            for seg in segments_z:
                if seg[4] != -1:
                    key = "{}-{}".format(len(collection_z_list) - 1, segi)
                    #print("key: {}".format(key))
                    patch = patches.Circle((seg[2], seg[1]), radius=_get_radius(seg), 
                                           facecolor=segs_color_dict[seg[4]], 
                                           alpha=0.5)
                    ax_z.add_patch(patch)
                    seg_patch_dict[key] = patch
                segi += 1
    
    def _force_seg_refresh(i):
       """Triggers table update by either selecting and reselected the segment
       or vice versa.
       
       Params:
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
                print("picked segment: {}".format(seg))
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
                            patch = patches.Circle((seg[2], seg[1]), radius=seg[3], 
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
            except ValueError:
                print("not on a plot to select a point")
       
    fig.canvas.mpl_connect("button_release_event", on_btn_release)
    
    # show 3D screenshot if available
    try:
        img3d = mlab.screenshot(antialiased=True)
        ax = plt.subplot(gs[0, zoom_levels - 1])
        # auto to adjust size with less overlap
        ax.imshow(img3d)
        ax.set_aspect(img3d.shape[1] / img3d.shape[0])
        _hide_axes(ax)
    except SceneModelError:
        print("No Mayavi image to screen capture")
    gs.tight_layout(fig, pad=0.5)
    #gs_zoomed.tight_layout(fig, pad=0.5)
    plt.ion()
    plt.show()
    if savefig is not None:
        name = title.replace("\n", "-").replace(" ", "") + "." + savefig
        print("saving figure as {}".format(name))
        plt.savefig(name)
    
def _hide_axes(ax):
    """Hides x- and y-axes.
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def extract_plane(image5d, channel, offset, name):
    """Extracts a single 2D plane and saves to file.
    
    Params:
        image5d: The full image stack.
        channel: Channel to view.
        offset: Offset given as (x, y, z), which will determine the
            z value of the plane to extract; x and y ignored for now.
        name: Name of the resulting file, without the extension, which
            will be chosen based on the savefig attribute.
    """
    # get the z-plane
    z_start = offset[2]
    if image5d.ndim >= 5:
        img2d = image5d[0, z_start, :, :, channel]
        print(img2d.shape)
    elif image5d.ndim == 4:
        img2d = image5d[0, z_start, :, :]
    else:
        img2d = image5d[z_start, :, :]
    if savefig is not None:
        filename = name + "." + savefig
        print("extracting plane as {}".format(filename))
        plt.imsave(filename, img2d, cmap=colormap_2d)

def cycle_colors(i):
    num_colors = len(config.colors)
    cycle = i // num_colors
    colori = i % num_colors
    color = config.colors[colori]
    upper = 255
    if cycle > 0:
        color = np.copy(color)
        color[0] = color[0] + cylce * 5
        if color[0] > upper:
            color[0] -= upper * (val // upper)
    return np.divide(color, upper)

def plot_roc(stats_dict, name):
    fig = plt.figure()
    label = ""
    colori = 0
    for key, value in stats_dict.items():
        stats = np.array(value[0])
        params = value[1]
        # false discovery rate, the inverse of PPV, since don't have a true negs
        fdr = np.subtract(1, np.divide(stats[:, 1], 
                                       np.add(stats[:, 1], stats[:, 2])))
        sens = np.divide(stats[:, 1], stats[:, 0])
        #print(fdr, sens)
        plt.scatter(fdr, sens, label=key, lw=2, color=cycle_colors(colori))
        colori += 1
        print("{}:".format(key))
        for i, n in enumerate(params):
            plt.annotate(n, (fdr[i], sens[i]))
            print("{}: ppv = {}, sens = {}, pos = {}, true pos = {}, "
                  "false pos = {}".format(
                  params[i], 1 - fdr[i], sens[i], *stats[i].astype(int)))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Discovery Rate")
    plt.ylabel("Sensitivity")
    plt.title("ROC for {}".format(name))
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    print("Testing plot_2d...")
    stats_dict = { 
        "test1": (np.array([[0.2, 0.3], [0.6, 0.7]]), [10, 20]),
        "test2": (np.array([[0.4, 0.7], [0.5, 0.8]]), [25, 34])
    }
    plot_roc(stats_dict, "Testing ROC")
    