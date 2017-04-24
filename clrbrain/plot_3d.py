# 3D plots from stacks of imaging data
# Author: David Young, 2017
"""Plots the image stack in 3D.

Provides options for drawing as surfaces or points.

Attributes:
    mask_dividend: Maximum number of points to show.
"""

from time import time
import numpy as np
import math
from skimage import restoration
from skimage import img_as_float
from skimage import filters

from clrbrain import config

INTENSITY_MIN = 0.2 # clip below this threshold
mask_dividend = 100000.0
MLAB_3D_TYPES = ("surface", "point")
mlab_3d = MLAB_3D_TYPES[1]

def denoise(roi):
    """Denoises an image.
    
    Args:
        roi: Region of interest.
    
    Returns:
        Denoised region of interest.
    """
    # enhance contrast and normalize to 0-1 scale
    vmin, vmax = np.percentile(roi, (5, 99.5))
    #print("vmin: {}, vmax: {}".format(vmin, vmax))
    denoised = np.clip(roi, vmin, vmax)
    denoised = (denoised - vmin) / (vmax - vmin)
    
    # additional simple thresholding
    denoised = np.clip(denoised, INTENSITY_MIN, 1)
    '''
    # total variation denoising
    time_start = time()
    denoised = restoration.denoise_tv_chambolle(denoised, weight=0.2)
    #denoised = restoration.denoise_nl_means(denoised, patch_size=10, multichannel=False)
    print('time for total variation: %f' %(time() - time_start))
    
    # sharpening
    '''
    unsharp_strength = 0.3
    blur_size = 8
    denoised = img_as_float(denoised)
    blurred = filters.gaussian(denoised, blur_size)
    high_pass = denoised - unsharp_strength * blurred
    denoised = denoised + high_pass
    
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

def deconvolve(roi):
    #shape = roi.shape
    psf = np.ones((5, 5, 5)) / 125
    roi_deconvolved = restoration.richardson_lucy(roi, psf, iterations=30)
    #roi_deconvolved = restoration.unsupervised_wiener(roi, psf)
    return roi_deconvolved

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
    
def _shadow_img2d(img2d, shape, axis, vis):
    img2d = np.swapaxes(img2d, 0, 1)
    img2d[img2d < 1] = 0
    extra_z = (shape[axis] - shape[0]) // 2
    if extra_z > 0:
        img2d_full = np.zeros(shape[1] * shape[2])
        img2d_full = np.reshape(img2d_full, [shape[1], shape[2]])
        print(img2d_full.shape)
        print(img2d.shape)
        print(extra_z, extra_z+img2d.shape[1])
        img2d_full[:, extra_z:extra_z+img2d.shape[1]] = img2d
        img2d = img2d_full
    return vis.scene.mlab.imshow(img2d, opacity=0.5, colormap="gray")

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
    
    # clear background points to see remaining structures
    remove = np.where(roi_1d < 1.3)
    x = np.delete(x, remove)
    y = np.delete(y, remove)
    z = np.delete(z, remove)
    roi_1d = np.delete(roi_1d, remove)
    # adjust range from 0-1 to region of colormap to use
    roi_1d = normalize(roi_1d, 0.4, 0.9)
    points_len = roi_1d.size
    time_start = time()
    mask = math.ceil(points_len / mask_dividend)
    print("points: {}, mask: {}".format(points_len, mask))
    if points_len > 0:
        vis.scene.mlab.points3d(x, y, z, roi_1d, 
                                mode="sphere", colormap="inferno", 
                                scale_mode="none", mask_points=mask, 
                                line_width=1.0, vmax=1.0, 
                                vmin=0.0, transparent=True)
        print("time for 3D points display: {}".format(time() - time_start))
    '''
    for i in range(roi_1d.size):
        print("x: {}, y: {}, z: {}, s: {}".format(x[i], y[i], z[i], roi_1d[i]))
    '''
    
    # 2D overlays on boders
    
    # xy-plane from bottom z
    #roi_xy = np.swapaxes(roi, 1, 2)
    img2d = np.copy(roi[shape[0] // 2, :, :])
    img2d_mlab = _shadow_img2d(img2d, shape, 0, vis)
    img2d_mlab.actor.position = [10, 10, -10]
    
    # yz-plane from top y
    '''
    img2d = np.add(img2d, roi_xy[:, shape[1] // 2, :])
    img2d = np.add(img2d, roi_xy[:, shape[1] - 1, :])
    img2d = np.divide(img2d, 2)
    '''
    img2d = np.copy(roi[:, shape[1] // 2, :])
    img2d_mlab = _shadow_img2d(img2d, shape, 2, vis)
    img2d_mlab.actor.position = [-10, 10, 5]
    img2d_mlab.actor.orientation = [90, 90, 0]
    
    # xz-plane from top x
    img2d = np.copy(roi[:, :, shape[2] // 2])
    img2d_mlab = _shadow_img2d(img2d, shape, 1, vis)
    img2d_mlab.actor.position = [10, -10, 5]
    img2d_mlab.actor.orientation = [90, 0, 0]

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
        The region of interest, including denoising, as a 3-dimensional
           array, without separate time or channel dimensions.
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
    if segments.shape[0] <= 0:
        return None, None, 0
    radii = segments[:, 3]
    scale = 5 if radii is None else 1.3 * np.mean(np.mean(radii) + np.amax(radii))
    print("blob point scaling: {}".format(scale))
    # colormap has to be at least 2 colors
    num_colors = segments.shape[0] if segments.shape[0] >= 2 else 2
    cmap = (np.random.random((num_colors, 4)) * 255).astype(np.uint8)
    cmap[:, -1] = 170
    if num_colors >= 4:
        # initial default colors using 7-color palatte for color blindness
        # (Wong, B. (2011) Nature Methods 8:441)
        cmap[0, 0:3] = 213, 94, 0 # vermillion
        cmap[1, 0:3] = 0, 114, 178 # blue
        cmap[2, 0:3] = 204, 121, 167 # reddish purple
        cmap[3, 0:3] = 0, 0, 0 # black
    cmap_indices = np.arange(segments.shape[0])
    pts = vis.scene.mlab.points3d(segments[:, 2], segments[:, 1], 
                            segments[:, 0], cmap_indices, #segments[:, 3],
                            scale_mode="none", scale_factor=scale) 
    pts.module_manager.scalar_lut_manager.lut.table = cmap
    return pts, cmap, scale

def normalize(array, minimum, maximum):
    array += -(np.min(array))
    array /= np.max(array)
    array += minimum
    return array
