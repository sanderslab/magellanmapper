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

colormap_2d = cm.inferno
savefig = None
verify = False
# TODO: may want to base on scaling factor instead
padding = (5, 5, 3) # human (x, y, z) order

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
        seg_patches.append(patches.Circle((seg[2], seg[1]), radius=seg[3]))
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

def show_subplot(fig, gs, row, col, image5d, channel, roi_size, offset, segments, 
                 segments_z, segs_cmap, alpha, highlight=False, border=None, 
                 segments_adj=None):
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
    """
    ax = plt.subplot(gs[row, col])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    size = image5d.shape
    z = offset[2]
    ax.set_title("z={}".format(z))
    collection_z = None
    if z < 0 or z >= size[1]:
        print("skipping z-plane {}".format(z))
        plt.imshow(np.zeros(roi_size[0:2]))
    else:
        # show the zoomed in 2D region
        region = [0, offset[2], 
                  slice(offset[1], offset[1] + roi_size[1]), 
                  slice(offset[0], offset[0] + roi_size[0])]
        if image5d.ndim >= 5:
            roi = image5d[tuple(region + [channel])]
        else:
            roi = image5d[tuple(region)]
        # highlight borders of z plane at bottom of ROI
        if highlight:
            for spine in ax.spines.values():
                spine.set_edgecolor("yellow")
        plt.imshow(roi, cmap=colormap_2d, alpha=alpha)
        
        # draws all segments as patches
        if segments is not None and segs_cmap is not None:
            collection = _circle_collection(segments, 
                                            segs_cmap.astype(float) / 255.0,
                                            "none", 3.0)
            ax.add_collection(collection)
        
        # overlays segments in current z with dotted line patch and makes
        # pickable for verifying the segment
        if segments_z is not None:
            collection_z = _circle_collection(segments_z, "w", "none", 2.0)
            collection_z.set_linestyle(":")
            collection_z.set_picker(5)
            ax.add_collection(collection_z)
        
        # overlays segments in adjacent regions with dashed line patch
        if segments_adj is not None:
            collection_adj = _circle_collection(segments_adj, "k", "none", 3.0)
            collection_adj.set_linestyle("--")
            ax.add_collection(collection_adj)
        
        # adds a simple border to highlight the bottom of the ROI
        if border is not None:
            ax.add_patch(patches.Rectangle(border[0:2], 
                                           roi_size[0] - 2 * border[0], 
                                           roi_size[1] - 2 * border[1], 
                                           fill=False, edgecolor="yellow",
                                           linestyle="dashed"))
    return ax, collection_z

def plot_2d_stack(vis, title, image5d, channel, roi_size, offset, segments, 
                  segs_cmap, border=None):
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
    """
    fig = plt.figure()
    fig.suptitle(title, color="navajowhite")
    
    # total number of z-planes
    z_planes = roi_size[2]
    z_planes_padding = padding[2] # addition z's on either side
    z_planes = z_planes + z_planes_padding * 2
    
    # plot layout depending on number of z-planes
    max_cols = 9
    zoom_plot_rows = math.ceil(z_planes / max_cols)
    col_remainder = z_planes % max_cols
    zoom_plot_cols = max(col_remainder, max_cols)
    top_rows = 3
    gs = gridspec.GridSpec(top_rows + zoom_plot_rows, zoom_plot_cols, 
                           wspace=0.7, hspace=0.5)
    
    # overview image, with bottom of offset shown as rectangle
    half_cols = zoom_plot_cols // 2
    ax = plt.subplot(gs[0:top_rows, :half_cols])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    z_start = offset[2]
    if image5d.ndim >= 5:
        img2d = image5d[0, z_start, :, :, channel]
    else:
        img2d = image5d[0, z_start, :, :]
    plt.imshow(img2d, cmap=colormap_2d)
    ax.add_patch(patches.Rectangle(offset[0:2], roi_size[0], roi_size[1], 
                                   fill=False, edgecolor="yellow"))
    add_scale_bar(ax)
    
    # zoomed-in views of z-planes spanning from just below to just above ROI
    #print("rows: {}, cols: {}, remainder: {}"
    #      .format(zoom_plot_rows, zoom_plot_cols, col_remainder))
    collection_z_list = []
    segments_z_list = []
    ax_z_list = []
    segs_out = None
    # finds adjacent segments, outside of the ROI
    if segments is not None:
        mask_in = np.all([segments[:, 0] >= 0, segments[:, 0] < roi_size[2],
                          segments[:, 1] >= 0, segments[:, 1] < roi_size[1],
                          segments[:, 2] >= 0, segments[:, 2] < roi_size[0]], 
                         axis=0)
        segs_out = segments[np.invert(mask_in)]
        print("segs_out:\n{}".format(segs_out))
    for i in range(zoom_plot_rows):
        # adjust columns for last row to number of plots remaining
        cols = max_cols
        if i == zoom_plot_rows - 1 and col_remainder > 0:
            cols = col_remainder
        # show zoomed in plots and highlight one at offset z
        for j in range(cols):
            # z relative to the start of the ROI, since segs are relative to ROI
            z_relative = i * max_cols + j - z_planes_padding
            # absolute z value, relative to start of image5d
            z = z_start + z_relative
            zoom_offset = (offset[0], offset[1], z)
            # fade z-planes outside of ROI
            alpha = 0.5 if z < z_start or z >= z_start + roi_size[2] else 1
            #print("row: {}".format(i * max_cols + j))
            segments_z = None
            if segments is not None:
                segments_z = segments[segments[:, 0] == z_relative]
            segments_z_list.append(segments_z)
            show_border = (verify and z_relative >= border[2] 
                           and z_relative < roi_size[2] - border[2])
            ax_z, collection_z = show_subplot(fig, gs, i + top_rows, j, image5d, 
                                              channel, roi_size,
                                              zoom_offset, segments, segments_z, 
                                              segs_cmap, alpha, z == z_start,
                                              border if show_border else None,
                                              segs_out)
            if i == 0 and j == 0:
                add_scale_bar(ax_z)
            collection_z_list.append(collection_z)
            ax_z_list.append(ax_z)
    
    # record selected segments in the Visualization segments table
    def on_pick(event):
        # ignore right-clicks
        if event.mouseevent.key == "control":
            return
        # segments_z_list is linked to collection list
        collection = event.artist
        collectioni = collection_z_list.index(collection)
        if collection != -1:
            # patch index is linked to segments_z_list
            seg = segments_z_list[collectioni][event.ind[0]]
            print("picked segment: {}".format(seg))
            segi = np.where((vis.segments == seg).all(axis=1))
            print(segi)
            if len(segi) > 0:
                # must take from vis rather than saved copy in case user 
                # manually updates the table
                i = segi[0][0]
                seg[4] = 1 if event.mouseevent.button == 1 else 0
                vis.segments[segi[0][0]] = seg
                # change selection simply to trigger table update in 
                # separate window
                if event.mouseevent.button == 1:
                    if not i in vis.segs_selected:
                        vis.segs_selected.append(i)
                else:
                    if not i in vis.segs_selected:
                        vis.segs_selected.append(i)
                    vis.segs_selected.remove(i)
       
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
                                     event.ydata, event.xdata, 0.0, 1]])
                    print("added segment: {}".format(seg))
                    # concatenate for in-place array update, though append
                    # and re-assigning also probably works
                    vis.segments = np.concatenate((vis.segments, seg))
                    # create a new copy rather than appending to trigger a
                    # full update; otherwise, only last entry gets selected
                    vis.segs_selected = (vis.segs_selected 
                                         + [vis.segments.shape[0] - 1])
            except ValueError:
                print("not on a plot to select a point")
       
    fig.canvas.mpl_connect("button_release_event", on_btn_release)
    
    # show 3D screenshot if available
    try:
        img3d = mlab.screenshot(antialiased=True)
        ax = plt.subplot(gs[0:top_rows, half_cols:zoom_plot_cols])
        ax.imshow(img3d)
        _hide_axes(ax)
    except SceneModelError:
        print("No Mayavi image to screen capture")
    gs.tight_layout(fig, pad=0.5)
    plt.ion()
    plt.show()
    if savefig is not None:
        name = title.replace("\n", "-").replace(" ", "") + "." + savefig
        print("saving figure as {}".format(name))
        plt.savefig(name)
    
    '''
    # demo 2D segmentation methods
    plt.figure()
    plt.imshow(img2d <= filters.threshold_otsu(img2d))
    #plt.imshow(image5d[0, offset[2], :, :], cmap=cm.gray)
    '''

def _hide_axes(ax):
    """Hides x- and y-axes.
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
