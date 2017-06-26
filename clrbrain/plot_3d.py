# 3D plots from stacks of imaging data
# Author: David Young, 2017
"""Plots the image stack in 3D.

Provides options for drawing as surfaces or points.

Attributes:
    mask_dividend: Maximum number of points to show.
    MLAB_3D_TYPES: Tuple of types of 3D visualizations for both the image
        and segmentation.
        * "surface": Renders surfaces in Mayavi contour and
          surface.
        * "point": Renders as raw points, minus points under
          the intensity_min threshold.
    mlab_3d: The chosen type.
"""

from time import time
import numpy as np
import math
from skimage import restoration
from skimage import img_as_float
from skimage import filters
from skimage import morphology

from clrbrain import config
from clrbrain import detector

_INTENSITY_MIN = 0.2 # clip below this threshold
_MASK_DIVIDEND = 100000.0 # 3D max points
MLAB_3D_TYPES = ("surface", "point")
mlab_3d = MLAB_3D_TYPES[1]

def denoise(roi):
    """Denoises an image.
    
    Args:
        roi: Region of interest.
    
    Returns:
        Denoised region of interest.
    """
    settings = config.process_settings
    # enhance contrast and normalize to 0-1 scale
    vmin, vmax = np.percentile(roi, (5, settings["clip_vmax"]))
    #print("vmin: {}, vmax: {}".format(vmin, vmax))
    denoised = np.clip(roi, vmin, vmax)
    denoised = (denoised - vmin) / (vmax - vmin)
    
    # additional simple thresholding
    denoised = np.clip(denoised, _INTENSITY_MIN, 1)
    
    if settings["tot_var_denoise"]:
        # total variation denoising
        time_start = time()
        denoised = restoration.denoise_tv_chambolle(denoised, weight=0.1)
        #denoised = restoration.denoise_nl_means(denoised, patch_size=10, 
        #                                        multichannel=False)
        print('time for total variation: %f' %(time() - time_start))
    
    # sharpening
    unsharp_strength = settings["unsharp_strength"]
    blur_size = 8
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

def threshold(roi):
    """Thresholds the ROI, with options for various techniques as well as
    post-thresholding morphological filtering.
    
    Params:
        roi: Region of interest, given as [z, y, x].
    
    Returns:
        The thresholded region.
    """
    thresh_type = config.process_settings["thresholding"]
    size = config.process_settings["thresholding_size"]
    thresholded = roi
    roi_thresh = 0
    
    # various thresholding model
    if thresh_type == "otsu":
        try:
            roi_thresh = filters.threshold_otsu(roi, size)
            thresholded = roi > roi_thresh
            '''
            thresholded = morphology.closing(thresholded, morphology.ball(1))
            '''
        except ValueError as e:
            # np.histogram may give an error apparently if any NaN, so 
            # workaround is set all elements in ROI to False
            print(e)
            roi_thresh = roi > np.max(roi)
    elif thresh_type == "local":
        roi_thresh = np.copy(roi)
        for i in range(roi_thresh.shape[0]):
            roi_thresh[i] = filters.threshold_local(
                roi_thresh[i], size, mode="wrap")
        thresholded = roi > roi_thresh
    elif thresh_type == "local-otsu":
        # TODO: not working yet
        selem = morphology.disk(15)
        print(np.min(roi), np.max(roi))
        roi_thresh = np.copy(roi)
        roi_thresh = normalize(roi_thresh, -1.0, 1.0)
        print(roi_thresh)
        print(np.min(roi_thresh), np.max(roi_thresh))
        for i in range(roi.shape[0]):
            roi_thresh[i] = filters.rank.otsu(
                roi_thresh[i], selem)
        thresholded = roi > roi_thresh
    elif thresh_type == "random_walker":
        _, thresholded = detector.segment_rw(roi, size)
    
    # dilation/erosion
    thresholded = morphology.dilation(
        thresholded, morphology.ball(2))
    thresholded = morphology.erosion(
        thresholded, morphology.cube(4))
    return thresholded

def deconvolve(roi):
    """Deconvolves the image.
    
    Params:
        roi: ROI given as a (z, y, x) subset of image5d.
    
    Returns:
        The ROI deconvolved.
    """
    # currently very simple with a generic point spread function
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
    roi = np.transpose(roi)
    
    # prepare the data source
    #roi = morphology.dilation(roi) # fill in holes to smooth surfaces
    roi = restoration.denoise_tv_chambolle(roi, weight=0.3)
    surface = pipeline.scalar_field(roi)
    
    '''
    # create the surface
    surface = pipeline.contour(surface)
    
    # removes many more extraneous points
    surface = pipeline.user_defined(surface, filter="SmoothPolyDataFilter")
    surface.filter.number_of_iterations = 400
    surface.filter.relaxation_factor = 0.015
    # holes within cells?
    surface = pipeline.user_defined(surface, filter="Curvatures")
    vis_mlab.pipeline.surface(surface)
    
    # colorizes
    module_manager = surface.children[0]
    module_manager.scalar_lut_manager.data_range = np.array([-0.6, 0.5])
    module_manager.scalar_lut_manager.lut_mode = "Greens"
    '''
    
    # based on Surface with contours enabled
    #surface = pipeline.contour_surface(surface)
    
    # uses unique IsoSurface module but appears to have 
    # similar output to contour_surface
    surface = pipeline.iso_surface(surface, color=(0.7, 1, 0.7))
    
def plot_3d_points(roi, vis):
    """Plots all pixels as points in 3D space.
    
    Points falling below a given threshold will be
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
    remove = np.where(roi_1d < config.process_settings["points_3d_thresh"])
    x = np.delete(x, remove)
    y = np.delete(y, remove)
    z = np.delete(z, remove)
    roi_1d = np.delete(roi_1d, remove)
    # adjust range from 0-1 to region of colormap to use
    roi_1d = normalize(roi_1d, 0.3, 0.6)
    #print(roi_1d)
    points_len = roi_1d.size
    time_start = time()
    mask = math.ceil(points_len / _MASK_DIVIDEND)
    print("points: {}, mask: {}".format(points_len, mask))
    if points_len > 0:
        vis.scene.mlab.points3d(x, y, z, roi_1d, 
                                mode="sphere", colormap="Greens", 
                                scale_mode="none", mask_points=mask, 
                                line_width=1.0, vmax=1.0, 
                                vmin=0.0, transparent=True)
        print("time for 3D points display: {}".format(time() - time_start))
    '''
    for i in range(roi_1d.size):
        print("x: {}, y: {}, z: {}, s: {}".format(x[i], y[i], z[i], roi_1d[i]))
    '''

def _shadow_img2d(img2d, shape, axis, vis):
    """Shows a plane along the given axis as a shadow parallel to
    the 3D visualization.
    
    Params:
        img2d: The plane to show.
        shape: Shape of the ROI.
        axis: Axis along which the plane lies.
        vis: Visualization object.
    
    Returns:
        The displayed plane.
    """
    img2d = np.swapaxes(img2d, 0, 1)
    img2d[img2d < 1] = 0
    # expands the plane to match the size of the xy plane, with this
    # plane in the middle
    extra_z = (shape[axis] - shape[0]) // 2
    if extra_z > 0:
        img2d_full = np.zeros(shape[1] * shape[2])
        img2d_full = np.reshape(img2d_full, [shape[1], shape[2]])
        img2d_full[:, extra_z:extra_z+img2d.shape[1]] = img2d
        img2d = img2d_full
    return vis.scene.mlab.imshow(img2d, opacity=0.5, colormap="gray")

def plot_2d_shadows(roi, vis):
    """Plots 2D shadows in each axis around the 3D visualization.
    
    Params:
        roi: Region of interest.
        vis: Visualization object on which to draw the contour. Any 
            current image will be cleared first.
    """ 
    # 2D overlays on boders
    shape = roi.shape
    
    # xy-plane
    #roi_xy = np.swapaxes(roi, 1, 2)
    img2d = np.copy(roi[shape[0] // 2, :, :])
    img2d_mlab = _shadow_img2d(img2d, shape, 0, vis)
    img2d_mlab.actor.position = [10, 10, -10]
    
    # xz-plane
    img2d = np.copy(roi[:, shape[1] // 2, :])
    img2d_mlab = _shadow_img2d(img2d, shape, 2, vis)
    img2d_mlab.actor.position = [-10, 10, 5]
    img2d_mlab.actor.orientation = [90, 90, 0]
    
    # yz-plane
    img2d = np.copy(roi[:, :, shape[2] // 2])
    img2d_mlab = _shadow_img2d(img2d, shape, 1, vis)
    img2d_mlab.actor.position = [10, -10, 5]
    img2d_mlab.actor.orientation = [90, 0, 0]

def prepare_roi(image5d, channel, roi_size, offset):
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
    # segments are in (z, y, x) order, so need to transpose or swap x,z axes
    # since Mayavi in (x, y, z)
    segments = np.transpose(segments)
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

def _shadow_blob(x, y, z, cmap_indices, cmap, scale, vis):
    """Shows blobs as shadows projected parallel to the 3D visualization.
    
    Parmas:
        x: Array of x-coordinates of blobs.
        y: Array of y-coordinates of blobs.
        z: Array of z-coordinates of blobs.
        cmap_indices: Indices of blobs for the colormap, usually given as a
            simple ascending sequence the same size as the number of blobs.
        cmap: The colormap, usually the same as for the segments.
        scale: Array of scaled size of each blob.
        vis: The Visualization object.
    """
    pts_shadows = vis.scene.mlab.points3d(x, y, z, cmap_indices, 
                                          mode="2dcircle", scale_mode="none", 
                                          scale_factor=scale*0.8, resolution=20)
    pts_shadows.module_manager.scalar_lut_manager.lut.table = cmap
    return pts_shadows

def show_blobs(segments, vis, show_shadows=False):
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
    # prioritize default colors
    for i in range(len(config.colors)):
        if i >= num_colors:
            break
        cmap[i, 0:3] = config.colors[i]
    cmap_indices = np.arange(segments.shape[0])
    
    if show_shadows:
        # show projections onto side planes
        segs_ones = np.ones(segments.shape[0])
        # xy
        _shadow_blob(segments[:, 2], segments[:, 1], segs_ones * -10, cmap_indices,
                     cmap, scale, vis)
        # xz
        shadows = _shadow_blob(segments[:, 2], segments[:, 0], segs_ones * -10, cmap_indices,
                     cmap, scale, vis)
        shadows.actor.actor.orientation = [90, 0, 0]
        shadows.actor.actor.position = [0, -20, 0]
        # yz
        shadows = _shadow_blob(segments[:, 1], segments[:, 0], segs_ones * -10, cmap_indices,
                     cmap, scale, vis)
        shadows.actor.actor.orientation = [90, 90, 0]
        shadows.actor.actor.position = [0, 0, 0]
        
    # show the blobs
    points_len = len(segments)
    mask = math.ceil(points_len / _MASK_DIVIDEND)
    print("points: {}, mask: {}".format(points_len, mask))
    pts = vis.scene.mlab.points3d(segments[:, 2], segments[:, 1], 
                            segments[:, 0], cmap_indices, mask_points=mask, 
                            scale_mode="none", scale_factor=scale, resolution=20) 
    pts.module_manager.scalar_lut_manager.lut.table = cmap
    
    
    return pts, cmap, scale

def normalize(array, minimum, maximum):
    """Normalizes an array to fall within the given min and max.
    
    Params:
        min: Minimum value for the array.
        max: Maximum value for the array.
    
    Returns:
        The normalized array.
    """
    if len(array) <= 0:
        return array
    array += -(np.min(array))
    array /= np.max(array) / (maximum - minimum)
    array += minimum
    return array
