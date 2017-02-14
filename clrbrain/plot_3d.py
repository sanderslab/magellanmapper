# 3D plots from stacks of imaging data
# Author: David Young, 2017
"""Plots the image stack in 3D.

Provides options for drawing as surfaces or points.

Attributes:
    intensity_min: The minimum intensity threshold for points
        viewing. Raising this threshold will remove more points.
"""

from time import time
import numpy as np
import math
from skimage import restoration

intensity_min = 0.2
mask_dividend = 100000.0

def denoise(roi):
    """Denoises an image.
    
    Args:
        roi: Region of interest.
    
    Returns:
        Denoised region of interest.
    """
    # enhance contrast and normalize to 0-1 scale
    vmin, vmax = np.percentile(roi, (20, 99.5))
    #print("vmin: {}, vmax: {}".format(vmin, vmax))
    denoised = np.clip(roi, vmin, vmax)
    denoised = (denoised - vmin) / (vmax - vmin)
    
    # additional simple thresholding
    denoised = np.clip(denoised, intensity_min, 1)
    
    # total variation denoising
    time_start = time()
    denoised = restoration.denoise_tv_chambolle(denoised, weight=0.2)
    print('time for total variation: %f' %(time() - time_start))
    
    '''
    # downgrade to uint16, which requires adjusting intensity 
    # thresholds (not quite complete here)
    from skimage import img_as_uint
    denoised = img_as_uint(denoised)
    global intensity_min
    intensity_min = img_as_uint(intensity_min)
    print(denoised)
    '''
    return denoised

def plot_3d_surface(roi, vis):
    """Plots areas with greater intensity as 3D surfaces.
    
    Args:
        roi: Region of interest.
        vis: Visualization object on which to draw the contour. Any 
            current image will be cleared first.
    """
    # Plot in Mayavi
    #mlab.figure()
    vis_mlab = vis.scene.mlab
    pipeline = vis_mlab.pipeline
    vis_mlab.clf()
    
    # ROI is in (z, y, x) order, so need to transpose or swap x,z axes
    #roi = np.flipud(roi)
    roi = np.transpose(roi)
    #roi = np.swapaxes(roi, 0, 2)
    #roi = np.fliplr(roi)
    
    # prepare the data source
    #np.transpose(roi, (0, 1, 3, 2, 4))
    scalars = pipeline.scalar_field(roi)
    
    # create the surface
    contour = pipeline.contour(scalars)
    # TESTING: use when excluding further processing
    #surf = pipeline.surface(contour)
    
    # removes many more extraneous points
    smooth = pipeline.user_defined(contour, 
                                   filter="SmoothPolyDataFilter")
    smooth.filter.number_of_iterations = 400
    smooth.filter.relaxation_factor = 0.015
    # holes within cells?
    curv = pipeline.user_defined(smooth, 
                                 filter="Curvatures")
    vis_mlab.pipeline.surface(curv)
    # colorizes
    module_manager = curv.children[0]
    module_manager.scalar_lut_manager.data_range = np.array([-0.6, 0.5])
    module_manager.scalar_lut_manager.lut_mode = "RdBu"
    
    # based on Surface with contours enabled
    #contour = pipeline.contour_surface(scalars)
    
    # uses unique IsoSurface module but appears to have 
    # similar output to contour_surface
    #contour = pipeline.iso_surface(scalars)
    
def plot_3d_points(roi, vis):
    """Plots all pixels as points in 3D space.
    
    Points falling below the "intensity_min" attribute will be
    removed, allowing the viewer to see through the presumed
    background to masses within the region of interest.
    
    Args:
        roi: Region of interest.
        vis: Visualization object on which to draw the contour. Any 
            current image will be cleared first.
    """
    print("plotting as 3D points")
    '''
    scalars = vis.scene.mlab.pipeline.scalar_scatter(roi)
    vis.scene.mlab.points3d(scalars)
    '''
    vis.scene.mlab.clf()
    shape = roi.shape
    z = np.ones((shape[0], shape[1] * shape[2]))
    for i in range(shape[0]):
        z[i] = z[i] * i
    y = np.ones((shape[0] * shape[1], shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            y[i * shape[1] + j] = y[i * shape[1] + j] * j
    x = np.ones((shape[0] * shape[1], shape[2]))
    for i in range(shape[0] * shape[1]):
        x[i] = np.arange(shape[2])
    x = np.reshape(x, roi.size)
    y = np.reshape(y, roi.size)
    z = np.reshape(z, roi.size)
    roi_1d = np.reshape(roi, roi.size)
    
    # clear background points to see remaining structures, starting with
    # the denoising threshold to approximately visualize what the segmenter 
    # will see and going slightly over for further clarity
    remove = np.where(roi_1d < intensity_min * 2)
    x = np.delete(x, remove)
    y = np.delete(y, remove)
    z = np.delete(z, remove)
    roi_1d = np.delete(roi_1d, remove)
    points_len = roi_1d.size
    time_start = time()
    mask = math.ceil(points_len / mask_dividend)
    print("points: {}, mask: {}".format(points_len, mask))
    vis.scene.mlab.points3d(x, y, z, roi_1d, 
                            mode="sphere", colormap="inferno", 
                            scale_mode="none", mask_points=mask, 
                            line_width=1.0, vmax=1.0, 
                            vmin=(intensity_min * 0.5), transparent=True)
    print("time for 3D points display: {}".format(time() - time_start))
    """
    for i in range(roi_1d.size):
        print("x: {}, y: {}, z: {}, s: {}".format(x[i], y[i], z[i], roi_1d[i]))
    """

def prepare_roi(image5d, channel, roi_size, offset=(0, 0, 0)):
    """Finds and shows the region of interest.
    
    This region will be denoised and displayed in Mayavi.
    
    Args:
        image5d: Image array.
        channel: Channel to view; wil be ignored if image5d has no
            channel dimension.
        roi_size: Size of the region of interest as (x, y, z).
        offset: Tuple of offset given as (x, y, z) for the region 
            of interest. Defaults to (0, 0, 0).
    
    Returns:
        The region of interest, including denoising.
    """
    cube_slices = []
    for i in range(len(offset)):
        cube_slices.append(slice(offset[i], offset[i] + roi_size[i]))
    print(offset, roi_size, cube_slices)
    
    # cube with corner at offset, side of cube_len
    if image5d.ndim >= 5:
        roi = image5d[0, cube_slices[2], cube_slices[1], cube_slices[0], channel]
    elif image5d.ndim == 4:
        roi = image5d[0, cube_slices[2], cube_slices[1], cube_slices[0]]
    else:
        roi = image5d[cube_slices[2], cube_slices[1], cube_slices[0]]
    
    return roi

def show_surface_labels(segments, vis):
    """Shows 3D surface segments from labels generated by segmentation
    methods such as Random-Walker.
    
    Args:
        segments: Labels from segmentation method.
        vis: Visualization GUI.
    """
    '''
    # Drawing options:
    # 1) draw iso-surface around segmented regions
    scalars = vis.scene.mlab.pipeline.scalar_field(labels)
    surf2 = vis.scene.mlab.pipeline.iso_surface(scalars)
    '''
    # 2) draw a contour or points directly from labels
    vis.scene.mlab.contour3d(segments)
    #surf2 = vis.scene.mlab.points3d(labels)
    return None

def show_blobs(segments, vis):
    """Shows 3D blob segments.
    
    Args:
        segments: Labels from 3D blob detection method.
        vis: Visualization GUI.
    
    Returns:
        The random colormap generated with a color for each blob.
    """
    scale = 1.5 * max(segments[:, 3])# * scaling_factor
    print("blob point scaling: {}".format(scale))
    # colormap has to be at least 2 colors
    num_colors = segments.shape[0] if segments.shape[0] >= 2 else 2
    cmap = (np.random.random((num_colors, 4)) * 255).astype(np.uint8)
    cmap[:, -1] = 170
    cmap_indices = np.arange(segments.shape[0])
    pts = vis.scene.mlab.points3d(segments[:, 2], segments[:, 1], 
                            segments[:, 0], cmap_indices, #segments[:, 3],
                            scale_mode="none", scale_factor=scale) 
    pts.module_manager.scalar_lut_manager.lut.table = cmap
    return cmap
