# 2D plots from stacks of imaging data
# Author: David Young, 2017
"""Plots 2D views through multiple levels of a 3D stack for
comparison with 3D visualization.

Attributes:
    colormap_2d: The Matplotlib colormap for 2D plots.
"""

import math
import numpy as np
from mayavi import mlab
from tvtk.pyface.scene_model import SceneModelError
from matplotlib import pyplot as plt, cm
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

colormap_2d = cm.inferno
savefig = None

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

def show_subplot(fig, gs, row, col, image5d, channel, roi_size, offset, segments, 
                 segments_z, segs_cmap, alpha, highlight=False):
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
        if image5d.ndim >= 5:
            roi = image5d[0, offset[2], 
                          slice(offset[1], offset[1] + roi_size[1]), 
                          slice(offset[0], offset[0] + roi_size[0]), channel]
        else:
            roi = image5d[0, offset[2], 
                          slice(offset[1], offset[1] + roi_size[1]), 
                          slice(offset[0], offset[0] + roi_size[0])]
        if highlight:
            for spine in ax.spines.values():
                spine.set_edgecolor("yellow")
        plt.imshow(roi, cmap=colormap_2d, alpha=alpha)
        if segments is not None and segs_cmap is not None:
            collection = _circle_collection(segments, 
                                            segs_cmap.astype(float) / 255.0,
                                            "none", 3.0)
            ax.add_collection(collection)
            
        if segments_z is not None:
            collection_z = _circle_collection(segments_z, "w", "none", 1.0)
            collection_z.set_linestyle(":")
            collection_z.set_picker(5)
            ax.add_collection(collection_z)
    return ax, collection_z
   
def plot_2d_stack(vis, title, image5d, channel, roi_size, offset, segments, 
                  segs_cmap):
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
        segs_cmap: Colormap for segments.
    """
    fig = plt.figure()
    fig.suptitle(title, color="navajowhite")
    
    # total number of z-planes
    z_planes = roi_size[2]
    if z_planes % 2 == 0:
        z_planes = z_planes + 1
    z_planes_padding = 3 # addition z's on either side
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
    
    # zoomed-in views of z-planes spanning from just below to just above ROI
    #print("rows: {}, cols: {}, remainder: {}"
    #      .format(zoom_plot_rows, zoom_plot_cols, col_remainder))
    collection_z_list = []
    segments_z_list = []
    ax_z_list = []
    for i in range(zoom_plot_rows):
        # adjust columns for last row to number of plots remaining
        cols = max_cols
        if i == zoom_plot_rows - 1 and col_remainder > 0:
            cols = col_remainder
        # show zoomed in plots and highlight one at offset z
        for j in range(cols):
            z_relative = i * max_cols + j - z_planes_padding
            z = z_start + z_relative
            zoom_offset = (offset[0], offset[1], z)
            # fade z-planes outside of ROI
            alpha = 0.5 if z < z_start or z >= z_start + roi_size[2] else 1
            #print("row: {}".format(i * max_cols + j))
            segments_z = None
            if segments is not None:
                segments_z = segments[segments[:, 0] == z_relative]
            segments_z_list.append(segments_z)
            ax_z, collection_z = show_subplot(fig, gs, i + top_rows, j, image5d, 
                                        channel, roi_size,
                                        zoom_offset, segments, segments_z, 
                                        segs_cmap, alpha, z == z_start)
            collection_z_list.append(collection_z)
            ax_z_list.append(ax_z)
    
    # record selected segments in the Visualization segments table
    def on_pick(event):
        # ignore right-clicks
        if event.mouseevent.button == 3:
            return
        # segments_z_list is linked to collection list
        collection = event.artist
        collectioni = collection_z_list.index(collection)
        if collection != -1:
            # patch index is linked to segments_z_list
            seg = segments_z_list[collectioni][event.ind[0]]
            print("picked segment: {}".format(seg))
            segi = np.where((segments == seg).all(axis=1))
            if len(segi) > 0:
                # must take from vis rather than saved copy in case user 
                # manually updates the table
                vis.segs_selected.append(segi[0][0])
       
    fig.canvas.mpl_connect("pick_event", on_pick)
    
    # add points that were not segmented by right-clicking on zoom plots
    def on_btn_release(event):
        ax = event.inaxes
        if event.button == 3:
            try:
                axi = ax_z_list.index(ax)
                if (axi != -1 and axi >= z_planes_padding 
                    and axi < z_planes - z_planes_padding):
                    
                    seg = np.array([[axi - z_planes_padding, 
                                     event.ydata, event.xdata, 0.0]])
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
