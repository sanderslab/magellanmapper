
import math
from mayavi import mlab
from tvtk.pyface.scene_model import SceneModelError
from matplotlib import pyplot as plt, cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pylab as pylab

colormap_2d = cm.inferno

def show_subplot(gs, row, col, image5d, offset, roi_size, highlight=False):
    #ax = plt.subplot2grid((2, 7), (1, subploti))
    ax = plt.subplot(gs[row, col])
    #ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    size = image5d.shape
    z = offset[2]
    ax.set_title("z={}".format(z))
    if z < 0 or z >= size[1]:
        print("skipping z-plane {}".format(z))
        plt.imshow(np.zeros(roi_size[0:2]))
    else:
        #i = subploti - 1
        #plt.imshow(image5d[0, i * nz//6, :, :, 0], cmap=cm.rainbow)
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
        """
            for i in range(roi.shape[0]):
                print("row {}: {}".format(i, " ".join(str(s) for s in roi[i])))
        roi_rgb = np.zeros((roi.shape[0:2], 3))
        roi_rgb[
        """
        plt.imshow(roi, cmap=colormap_2d)
   
def plot_2d_stack(title, image5d, roi_size, offset):
    fig = plt.figure()
    fig.suptitle(title, color="navajowhite")
    
    # total number of z-planes
    z_planes = roi_size[2]
    if z_planes % 2 == 0:
        z_planes = z_planes + 1
    z_planes_padding = 3 # addition z's on either side
    z_planes = z_planes + z_planes_padding * 2
    
    # plot layout depending on number of z-planes
    max_cols = 15
    zoom_plot_rows = math.ceil(z_planes / max_cols)
    col_remainder = z_planes % max_cols
    zoom_plot_cols = max(col_remainder, max_cols)
    top_rows = 4
    gs = gridspec.GridSpec(top_rows + zoom_plot_rows, zoom_plot_cols, 
                           wspace=0.5, hspace=0)
    
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
    print("rows: {}, cols: {}, remainder: {}"
          .format(zoom_plot_rows, zoom_plot_cols, col_remainder))
    for i in range(zoom_plot_rows):
    	# adjust columns for last row to number of plots remaining
    	cols = max_cols
    	if i == zoom_plot_rows - 1 and col_remainder > 0:
    	    cols = col_remainder
    	# show zoomed in plots and highlight one at offset z
    	for j in range(cols):
            z = z_start - z_planes_padding + i * max_cols + j
            zoom_offset = (offset[0], offset[1], z)
            show_subplot(gs, i + top_rows, j, image5d, zoom_offset, 
                         roi_size, z == z_start)
    
    # show 3D screenshot if available
    try:
        img3d = mlab.screenshot(antialiased=True)
        ax = plt.subplot(gs[0:top_rows, half_cols:zoom_plot_cols])
        ax.imshow(img3d)
        _hide_axes(ax)
    except SceneModelError as err:
        print("No Mayavi image to screen capture")
    gs.tight_layout(fig, pad=0)
    plt.ion()
    plt.show()
    
    '''
    # demo 2D segmentation methods
    plt.figure()
    plt.imshow(img2d <= filters.threshold_otsu(img2d))
    #plt.imshow(image5d[0, offset[2], :, :], cmap=cm.gray)
    '''

def _hide_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
