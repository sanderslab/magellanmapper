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
from scipy import ndimage
from skimage import draw
from skimage import restoration
from skimage import img_as_uint
from skimage import img_as_float
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import transform

from clrbrain import config
from clrbrain import detector
from clrbrain import lib_clrbrain

_MASK_DIVIDEND = 10000.0 # 3D max points
_MAYAVI_COLORMAPS = ("Greens", "Reds", "Blues", "Oranges")
MLAB_3D_TYPES = ("surface", "point")
mlab_3d = MLAB_3D_TYPES[1]

def setup_channels(roi, channel, dim_channel):
    """Setup channels array for the given ROI dimensions.
    
    Args:
        roi: Region of interest, which is either a 3D or 4D array of 
            [[z, y, x, (c)], ...].
        channel: Channel to select, which can be None to indicate all 
            channels.
        dim_channel: Index of the channel dimension.
    
    Returns:
        A tuple of ``multichannel``, a boolean value where True indicates that 
        the ROI is multichannel (ie 4D), and ``channels``, an array of the 
        channel indices of ROI to include.
    """
    multichannel = roi.ndim > dim_channel
    channels = [0]
    if multichannel:
        channels = range(roi.shape[dim_channel]) if channel is None else [channel]
    '''
    lib_clrbrain.printv(
        "multichannel: {}, channels: {}, roi shape: {}, channel: {}"
        .format(multichannel, channels, roi.shape, channel))
    '''
    return multichannel, channels

def saturate_roi(roi, clip_vmax=-1, channel=None):
    """Saturates an image, clipping extreme values and stretching remaining
    values to fit the full range.
    
    Args:
        roi: Region of interest.
    
    Returns:
        Saturated region of interest.
    """
    multichannel, channels = setup_channels(roi, channel, 3)
    roi_out = None
    for i in channels:
        roi_show = roi[..., i] if multichannel else roi
        settings = config.get_process_settings(i)
        if clip_vmax == -1:
            clip_vmax = settings["clip_vmax"]
        # enhance contrast and normalize to 0-1 scale
        vmin, vmax = np.percentile(roi_show, (5, clip_vmax))
        lib_clrbrain.printv("vmin: {}, vmax: {}".format(vmin, vmax))
        # ensures that vmax is at least 50% of near max value of image5d
        lib_clrbrain.printv("near_max: {}".format(config.near_max[i]))
        max_thresh = config.near_max[i] * 0.5
        if vmax < max_thresh:
            vmax = max_thresh
            lib_clrbrain.printv("adjusted vmax to {}".format(vmax))
        saturated = np.clip(roi_show, vmin, vmax)
        saturated = (saturated - vmin) / (vmax - vmin)
        if multichannel:
            if roi_out is None:
                roi_out = np.zeros(roi.shape, dtype=saturated.dtype)
            roi_out[..., i] = saturated
        else:
            roi_out = saturated
    return roi_out
    
def denoise_roi(roi, channel=None):
    """Denoises an image.
    
    Args:
        roi: Region of interest as a 3D (z, y, x) array. Note that 4D arrays 
            with channels are not allowed as the Scikit-Image gaussian filter 
            only accepts specifically 3 channels, presumably for RGB.
    
    Returns:
        Denoised region of interest.
    """
    multichannel, channels = setup_channels(roi, channel, 3)
    roi_out = None
    for i in channels:
        roi_show = roi[..., i] if multichannel else roi
        settings = config.get_process_settings(i)
        # find gross density
        saturated_mean = np.mean(roi_show)
        lib_clrbrain.printv("saturated_mean: {}".format(saturated_mean))
        
        # additional simple thresholding
        denoised = np.clip(roi_show, settings["clip_min"], settings["clip_max"])
        
        if settings["tot_var_denoise"]:
            # total variation denoising
            #time_start = time()
            denoised = restoration.denoise_tv_chambolle(denoised, weight=0.1)
            #denoised = restoration.denoise_tv_bregman(denoised, weight=0.1)
            #print('time for total variation: %f' %(time() - time_start))
        
        # sharpening
        unsharp_strength = settings["unsharp_strength"]
        blur_size = 8
        # turn off multichannel since assume operation on single channel at 
        # a time and to avoid treating as multichannel if 3D ROI happens to 
        # have x size of 3
        blurred = filters.gaussian(denoised, blur_size, multichannel=False)
        high_pass = denoised - unsharp_strength * blurred
        denoised = denoised + high_pass
        
        # further erode denser regions to decrease overlap among blobs
        if saturated_mean > settings["erosion_threshold"]:
            #print("denoising for saturated mean of {}".format(saturated_mean))
            denoised = morphology.erosion(denoised, morphology.octahedron(1))
        if multichannel:
            if roi_out is None:
                roi_out = np.zeros(roi.shape, dtype=denoised.dtype)
            roi_out[..., i] = denoised
        else:
            roi_out = denoised
    return roi_out

def threshold(roi):
    """Thresholds the ROI, with options for various techniques as well as
    post-thresholding morphological filtering.
    
    Args:
        roi: Region of interest, given as [z, y, x].
    
    Returns:
        The thresholded region.
    """
    settings = config.process_settings
    thresh_type = settings["thresholding"]
    size = settings["thresholding_size"]
    thresholded = roi
    roi_thresh = 0
    
    # various thresholding model
    if thresh_type == "otsu":
        try:
            roi_thresh = filters.threshold_otsu(roi, size)
            thresholded = roi > roi_thresh
        except ValueError as e:
            # np.histogram may give an error apparently if any NaN, so 
            # workaround is set all elements in ROI to False
            print(e)
            thresholded = roi > np.max(roi)
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
        roi_thresh = lib_clrbrain.normalize(roi_thresh, -1.0, 1.0)
        print(roi_thresh)
        print(np.min(roi_thresh), np.max(roi_thresh))
        for i in range(roi.shape[0]):
            roi_thresh[i] = filters.rank.otsu(
                roi_thresh[i], selem)
        thresholded = roi > roi_thresh
    elif thresh_type == "random_walker":
        _, thresholded = detector.segment_rw(roi, size)
    
    # dilation/erosion, adjusted based on overall intensity
    thresh_mean = np.mean(thresholded)
    print("thresh_mean: {}".format(thresh_mean))
    selem_dil = None
    selem_eros = None
    if thresh_mean > 0.45:
        thresholded = morphology.erosion(thresholded, morphology.cube(1))
        selem_dil = morphology.ball(1)
        selem_eros = morphology.octahedron(1)
    elif thresh_mean > 0.35:
        thresholded = morphology.erosion(thresholded, morphology.cube(2))
        selem_dil = morphology.ball(2)
        selem_eros = morphology.octahedron(1)
    elif thresh_mean > 0.3:
        selem_dil = morphology.ball(1)
        selem_eros = morphology.cube(5)
    elif thresh_mean > 0.1:
        selem_dil = morphology.ball(1)
        selem_eros = morphology.cube(4)
    elif thresh_mean > 0.05:
        selem_dil = morphology.octahedron(2)
        selem_eros = morphology.octahedron(2)
    else:
        selem_dil = morphology.octahedron(1)
        selem_eros = morphology.octahedron(2)
    if selem_dil is not None:
        thresholded = morphology.dilation(thresholded, selem_dil)
    if selem_eros is not None:
        thresholded = morphology.erosion(thresholded, selem_eros)
    return thresholded

def deconvolve(roi):
    """Deconvolves the image.
    
    Args:
        roi: ROI given as a (z, y, x) subset of image5d.
    
    Returns:
        The ROI deconvolved.
    """
    # currently very simple with a generic point spread function
    psf = np.ones((5, 5, 5)) / 125
    roi_deconvolved = restoration.richardson_lucy(roi, psf, iterations=30)
    #roi_deconvolved = restoration.unsupervised_wiener(roi, psf)
    return roi_deconvolved

def in_paint(roi, to_fill):
    """In-paint to interpolate values into pixels to fill from nearest 
    neighbors.
    
    Args:
        roi: ROI in which to fill pixels.
        to_fill: Boolean array of same shape as ``roi`` where True values 
            designate the pixels to fill.
    
    Returns:
        Copy of ROI with pixels corresponding to ``to_fill`` filled with 
        nearest neighbors.
    """
    indices = ndimage.distance_transform_edt(
        to_fill, return_distances=False, return_indices=True)
    filled = roi[tuple(indices)]
    return filled

def carve(roi, thresh=None, holes_area=None, return_unfilled=False):
    # erase below simple threshold
    roi_carved = np.copy(roi)
    if thresh is None:
        thresh = filters.threshold_mean(roi_carved)
    mask = roi_carved < thresh
    roi_carved[mask] = 0
    
    # fill small holes from original ROI
    roi_uint = roi_carved
    if not isinstance(roi[..., 0], np.int):
        print("converting image to uint to remove small holes")
        roi_uint = img_as_uint(roi_carved)
    labels = measure.label(roi_uint)
    to_fill = np.logical_and(
        morphology.remove_small_holes(labels, holes_area), mask)
    roi_unfilled = np.copy(roi_carved) if return_unfilled else None
    roi_carved[to_fill] = roi[to_fill]
    if return_unfilled:
        return roi_carved, roi_unfilled
    return roi_carved

def rotate_nd(img_np, angle, axis=0, order=1):
    rotated = np.copy(img_np)
    slices = [slice(None)] * img_np.ndim
    for i in range(img_np.shape[axis]):
        #slices[axis] = slice(i, i + 1)
        slices[axis] = i
        img2d = img_np[slices]
        img2d = transform.rotate(
            img2d, angle, order=order, mode="constant", preserve_range=True)
        rotated[slices] = img2d
    return rotated

def perimeter_nd(img_np):
    """Get perimeter of image subtracting eroded image from given image.
    
    Args:
        img_np: Numpy array of arbitrary dimensions.
    
    Returns:
        The perimeter as the border that would have been eroded.
    """
    interior = morphology.binary_erosion(img_np)
    img_border = np.logical_xor(img_np, interior)
    #print("perimeter:\n{}".format(img_border))
    return img_border

def borders_distance(borders_orig, borders_shifted, filter_size=None):
    """Measure distance between borders.
    
    Args:
        borders_orig: Original borders as a boolean mask.
        borders_shifted: Shifted borders as a boolean mask, which should 
            match the shape of ``borders_orig``.
        filter_size: Size of structuring element to use in filter for 
            smoothing the original border with a closing filter before 
            finding distances. Defaults to None, in which case no filter 
            will be applied.
    
    Returns:
        Tuple of ``dist_to_orig``, a Numpy array the same shape as 
        ``borders_orig`` with distances generated from a Euclidean 
        distance transform to the original borders, or to the smoothed 
        borders if ``filter_size`` is given; ``indices``, the distance 
        transform indices in ``borders_orig`` corresponding to each pixel 
        in ``borders_smoothed``; and ``borders_orig`` to 
        allow accessing the smoothed borders.
    """
    if filter_size is not None:
        # find the pixels surrounding the original border
        borders_orig = morphology.binary_closing(
            borders_orig, morphology.ball(filter_size))
    # find distances to the original borders, inverted to become background
    borders_dist, indices = ndimage.distance_transform_edt(
        ~borders_orig, return_indices=True)
    # gather the distances corresponding to the shifted border
    dist_to_orig = np.zeros_like(borders_dist)
    dist_to_orig[borders_shifted] = borders_dist[borders_shifted]
    lib_clrbrain.print_compact(borders_orig, "borders orig", True)
    lib_clrbrain.print_compact(borders_dist, "borders dist", True)
    lib_clrbrain.print_compact(dist_to_orig, "dist_to_orig", True)
    #print(borders_orig)
    return dist_to_orig, indices, borders_orig

def radial_dist(borders, centroid):
    """Measure radial distance from borders to given center.
    
    Args:
        borders: Original borders as a boolean mask.
        centroid: Coordinates corresponding to chosen reference point.
    
    Returns:
        A Numpy array the same shape as ``borders`` with distances measured 
        from each point in ``borders`` to ``centroid`` point.
    """
    center_img = np.ones(borders.shape)
    center_img[tuple(int(n) for n in centroid)] = 0
    radial_dist = ndimage.distance_transform_edt(center_img)
    borders_dist = np.zeros_like(radial_dist)
    borders_dist[borders] = radial_dist[borders]
    return borders_dist

def radial_dist_diff(radial_orig, radial_shifted, indices):
    """Measure the difference between corresponding points in radial 
    distance arrays to get the relative distance from one set of borders 
    to another with reference to the centroid from which the radial 
    distances were calculated.
    
    Shifted points with positive distances are farther from the reference 
    point than the closest original point is.
    
    Args:
        radial_orig: Radial original distances as a Numpy array.
        radial_shifted: Radial shifted distances as a Numpy array.
    
    Returns:
        A Numpy array the same shape as ``radial_orig`` with differences 
        in distance from ``radial_shifted`` to the corresponding points 
        in ``radial_orig``.
    """
    dist_at_nearest_orig = radial_orig[tuple(indices)]
    dist_at_nearest_orig[radial_shifted <= 0] = 0
    radial_diff = np.subtract(radial_shifted, dist_at_nearest_orig)
    '''
    print(radial_orig)
    print(radial_shifted)
    print("indices:\n{}".format(indices))
    print(dist_at_nearest_orig)
    print(radial_diff)
    '''
    return radial_diff

def get_bbox_region(bbox, padding=0, img_shape=None):
    dims = len(bbox) // 2 # bbox has min vals for each dim, then maxes
    shape = [bbox[i + dims] - bbox[i] for i in range(dims)]
    slices = []
    for i in range(dims):
        # add padding for slices and update shape
        start = bbox[i] - padding
        stop = bbox[i] + shape[i] + padding
        if img_shape is not None:
            if start < 0: start = 0
            if stop >= img_shape[i]: stop = img_shape[i]
        slices.append(slice(start, stop))
        shape[i] = stop - start
    #print("shape: {}, slices: {}".format(shape, slices))
    return shape, slices

def get_label_bbox(labels_img_np, label_id):
    label_mask = labels_img_np == label_id
    props = measure.regionprops(label_mask.astype(np.int))
    bbox = None
    if len(props) >= 1: bbox = props[0].bbox
    return bbox

def calc_isotropic_factor(scale):
    res = detector.resolutions[0]
    resize_factor = np.divide(res, np.amin(res))
    resize_factor *= scale
    #print("isotropic resize factor: {}".format(resize_factor))
    return resize_factor
    #return np.array((1, 1, 1))

def make_isotropic(roi, scale):
    resize_factor = calc_isotropic_factor(scale)
    isotropic_shape = np.array(roi.shape)
    isotropic_shape[:3] = (isotropic_shape[:3] * resize_factor).astype(np.int)
    lib_clrbrain.printv("original ROI shape: {}, isotropic: {}"
                        .format(roi.shape, isotropic_shape))
    mode = "reflect"
    if np.any(np.array(roi.shape) == 1):
        # may crash with floating point exception if 1px thick (see 
        # https://github.com/scikit-image/scikit-image/issues/3001, which 
        # causes multiprocessing Pool to hang since the exception isn't 
        # raised), so need to change mode in this case
        mode = "edge"
    return transform.resize(
        roi, isotropic_shape, preserve_range=True, mode=mode, 
        anti_aliasing=True)

def plot_3d_surface(roi, scene_mlab, channel, segment=False):
    """Plots areas with greater intensity as 3D surfaces.
    
    Args:
        roi: Region of interest.
        scene_mlab: ``MayaviScene.mlab`` attribute to draw the contour. Any 
            current image will be cleared first.
        segment: True to denoise and segment ``roi`` before displaying, 
            which may remove artifacts that might otherwise lead to 
            spurious surfaces. Defaults to False.
    """
    # Plot in Mayavi
    #mlab.figure()
    print("viewing 3D surface")
    pipeline = scene_mlab.pipeline
    scene_mlab.clf()
    # saturate to remove noise and normalize values
    roi = saturate_roi(roi, 99, channel=channel)
    #roi = np.clip(roi, 0.2, 0.8)
    #roi = restoration.denoise_tv_chambolle(roi, weight=0.1)
        
    settings = config.process_settings
    
    # turn off segmentation if ROI too big (arbitrarily set here as 
    # > 10 million pixels) to avoid performance hit and since likely showing 
    # large region of downsampled image anyway, where don't need hi res
    num_pixels = np.prod(roi.shape)
    to_segment = num_pixels < 10000000
    
    time_start = time()
    multichannel, channels = setup_channels(roi, channel, 3)
    for i in channels:
        roi_show = roi[..., i] if multichannel else roi
        
        # clip to minimize sub-nuclear variation
        roi_show = np.clip(roi_show, 0.2, 0.8)
        
        if segment:
            # denoising makes for much cleaner images but also seems to allow 
            # structures to blend together. TODO: consider segmented individual 
            # structures and rendering them as separate surfaces to avoid blending
            roi_show = restoration.denoise_tv_chambolle(roi_show, weight=0.1)
            
            # build surface from segmented ROI
            if to_segment:
                _, walker = detector.segment_rw(roi_show, i, vmin=0.6, vmax=0.7)
                roi_show *= np.subtract(walker[0], 1)
            else:
                print("deferring segmentation as {} px is above threshold"
                      .format(num_pixels))
        
        # ROI is in (z, y, x) order, so need to transpose or swap x,z axes
        roi_show = np.transpose(roi_show)
        surface = pipeline.scalar_field(roi_show)
        
        '''
        # create the surface
        surface = pipeline.contour(surface)
        
        # removes many more extraneous points
        surface = pipeline.user_defined(surface, filter="SmoothPolyDataFilter")
        surface.filter.number_of_iterations = 400
        surface.filter.relaxation_factor = 0.015
        # holes within cells?
        surface = pipeline.user_defined(surface, filter="Curvatures")
        scene_mlab.pipeline.surface(surface)
        
        # colorizes
        module_manager = surface.children[0]
        module_manager.scalar_lut_manager.data_range = np.array([-0.6, 0.5])
        module_manager.scalar_lut_manager.lut_mode = "Greens"
        '''
        
        # based on Surface with contours enabled
        '''
        surface = pipeline.contour_surface(
            surface, color=(0.7, 1, 0.7), line_width=6.0)
        surface.actor.property.representation = 'wireframe'
        #surface.actor.property.line_width = 6.0
        surface.actor.mapper.scalar_visibility = False
        '''
        
        # uses unique IsoSurface module but appears to have 
        # similar output to contour_surface
        surface = pipeline.iso_surface(surface, colormap=_MAYAVI_COLORMAPS[i])
        try:
            surface.contour.minimum_contour = 0.5
            surface.contour.maximum_contour = 1.0
        except Exception as e:
            print(e)
            print("ignoring min/max contour for now")
        isotropic = settings["isotropic_vis"]
        if isotropic is not None:
            surface.actor.actor.scale = isotropic[::-1]
    
    print("time to render 3D surface: {}".format(time() - time_start))
    
def plot_3d_points(roi, scene_mlab, channel):
    """Plots all pixels as points in 3D space.
    
    Points falling below a given threshold will be
    removed, allowing the viewer to see through the presumed
    background to masses within the region of interest.
    
    Args:
        roi: Region of interest either as a 3D (z, y, x) or 
            4D (z, y, x, channel) ndarray.
        scene_mlab: ``MayaviScene.mlab`` attribute to draw the contour. Any 
            current image will be cleared first.
        channel: Channel to select, which can be None to indicate all 
            channels.
    
    Returns:
        True if points were rendered, False if no points to render.
    """
    print("plotting as 3D points")
    scene_mlab.clf()
    settings = config.process_settings
    
    # streamline the image
    roi = saturate_roi(roi, 98.5, channel)
    roi = np.clip(roi, 0.2, 0.8)
    roi = restoration.denoise_tv_chambolle(roi, weight=0.1)
    
    # separate parallel arrays for each dimension of all coordinates for
    # Mayavi input format, with the ROI itself given as a 1D scalar array 
    time_start = time()
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
    multichannel, channels = setup_channels(roi, channel, 3)
    for i in channels:
        roi_show = roi[..., i] if multichannel else roi
        roi_show_1d = roi_show.reshape(roi_show.size)
        if i == 0:
            x = np.reshape(x, roi_show.size)
            y = np.reshape(y, roi_show.size)
            z = np.reshape(z, roi_show.size)
        
        # clear background points to see remaining structures
        thresh = 0
        if len(np.unique(roi_show)) > 1:
            # need > 1 val to threshold
            thresh = (filters.threshold_otsu(roi_show, 64) 
                      * settings["points_3d_thresh"])
        print("removing 3D points below threshold of {}".format(thresh))
        remove = np.where(roi_show_1d < thresh)
        roi_show_1d = np.delete(roi_show_1d, remove)
        
        # adjust range from 0-1 to region of colormap to use
        roi_show_1d = lib_clrbrain.normalize(roi_show_1d, 0.6, 1.0)
        points_len = roi_show_1d.size
        if points_len == 0:
            print("no 3D points to display")
            return False
        mask = math.ceil(points_len / _MASK_DIVIDEND)
        print("points: {}, mask: {}".format(points_len, mask))
        # TODO: better performance if manually interval the points rather than 
        # through mask flag?
        #roi_show_1d = roi_show_1d[::mask]
        pts = scene_mlab.points3d(
            np.delete(x, remove), np.delete(y, remove), np.delete(z, remove), 
            roi_show_1d, mode="sphere", colormap=_MAYAVI_COLORMAPS[i], 
            scale_mode="scalar", mask_points=mask, line_width=1.0, vmax=1.0, 
            vmin=0.0, transparent=True)
        isotropic = settings["isotropic_vis"]
        if isotropic is not None:
            pts.actor.actor.scale = isotropic[::-1]
    
    print("time for 3D points display: {}".format(time() - time_start))
    return True

def _shadow_img2d(img2d, shape, axis, vis):
    """Shows a plane along the given axis as a shadow parallel to
    the 3D visualization.
    
    Args:
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
    
    Args:
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

def prepare_roi(image5d, roi_size, offset):
    """Extracts a region of interest from a larger image.
    
    Args:
        image5d: Image array as a 5D array (t, z, y, x, c), or 4D if  
            no separate channel dimension exists as with most one channel 
            images.
        roi_size: Size of the region of interest as (x, y, z).
        offset: Tuple of offset given as (x, y, z) for the region 
            of interest. Defaults to (0, 0, 0).
    
    Returns:
        The region of interest without separate time dimension as a 3D 
        if ``image5d`` is 4D, without a separate channel dimension, or 4-D 
        array if channel dimension exists.
    """
    cube_slices = []
    for i in range(len(offset)):
        cube_slices.append(slice(offset[i], offset[i] + roi_size[i]))
    lib_clrbrain.printv("preparing ROI at offset: {}, size: {}, slices: {}"
                        .format(offset, roi_size, cube_slices))
    
    # cube with corner at offset, side of cube_len
    if image5d.ndim >= 5:
        roi = image5d[0, cube_slices[2], cube_slices[1], cube_slices[0], :]
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

def _shadow_blob(x, y, z, cmap_indices, cmap, scale, mlab):
    """Shows blobs as shadows projected parallel to the 3D visualization.
    
    Parmas:
        x: Array of x-coordinates of blobs.
        y: Array of y-coordinates of blobs.
        z: Array of z-coordinates of blobs.
        cmap_indices: Indices of blobs for the colormap, usually given as a
            simple ascending sequence the same size as the number of blobs.
        cmap: The colormap, usually the same as for the segments.
        scale: Array of scaled size of each blob.
        mlab: Mayavi object.
    """
    pts_shadows = mlab.points3d(x, y, z, cmap_indices, 
                                          mode="2dcircle", scale_mode="none", 
                                          scale_factor=scale*0.8, resolution=20)
    pts_shadows.module_manager.scalar_lut_manager.lut.table = cmap
    return pts_shadows

def show_blobs(segments, mlab, segs_in_mask, show_shadows=False):
    """Shows 3D blob segments.
    
    Args:
        segments: Labels from 3D blob detection method.
        mlab: Mayavi object.
        segs_in_mask: Boolean mask for segments within the ROI; all other 
            segments are assumed to be from padding and border regions 
            surrounding the ROI.
        show_shadows: True if shadows of blobs should be depicted on planes 
            behind the blobs; defaults to False.
    
    Returns:
        A 3-element tuple containing ``pts_in``, the 3D points within the 
        ROI; ``cmap'', the random colormap generated with a color for each 
        blob, and ``scale``, the current size of the points.
    """
    if segments.shape[0] <= 0:
        return None, None, 0
    settings = config.process_settings
    segs = segments
    isotropic = settings["isotropic_vis"]
    if isotropic is not None:
        # adjust position based on isotropic factor
        segs = np.copy(segs)
        segs[:, :3] = np.multiply(segs[:, :3], isotropic)
    
    radii = segs[:, 3]
    scale = 5 if radii is None else np.mean(np.mean(radii) + np.amax(radii))
    print("blob point scaling: {}".format(scale))
    # colormap has to be at least 2 colors
    segs_in = segs[segs_in_mask]
    num_colors = segs_in.shape[0] if segs_in.shape[0] >= 2 else 2
    cmap = lib_clrbrain.discrete_colormap(num_colors, alpha=170)
    cmap_indices = np.arange(segs_in.shape[0])
    
    if show_shadows:
        # show projections onto side planes
        segs_ones = np.ones(segs.shape[0])
        # xy
        _shadow_blob(
            segs_in[:, 2], segs_in[:, 1], segs_ones * -10, cmap_indices,
            cmap, scale, mlab)
        # xz
        shadows = _shadow_blob(
            segs_in[:, 2], segs_in[:, 0], segs_ones * -10, cmap_indices,
            cmap, scale, mlab)
        shadows.actor.actor.orientation = [90, 0, 0]
        shadows.actor.actor.position = [0, -20, 0]
        # yz
        shadows = _shadow_blob(
            segs_in[:, 1], segs_in[:, 0], segs_ones * -10, cmap_indices,
            cmap, scale, mlab)
        shadows.actor.actor.orientation = [90, 90, 0]
        shadows.actor.actor.position = [0, 0, 0]
        
    # show the blobs
    points_len = len(segs)
    mask = math.ceil(points_len / _MASK_DIVIDEND)
    print("points: {}, mask: {}".format(points_len, mask))
    # show segs within the ROI
    pts_in = mlab.points3d(
        segs_in[:, 2], segs_in[:, 1], 
        segs_in[:, 0], cmap_indices, 
        mask_points=mask, scale_mode="none", scale_factor=scale, resolution=50) 
    # show segments within padding or boder region as more transparent
    segs_out_mask = np.logical_not(segs_in_mask)
    pts_out = mlab.points3d(
        segs[segs_out_mask, 2], segs[segs_out_mask, 1], 
        segs[segs_out_mask, 0], color=(0, 0, 0), 
        mask_points=mask, scale_mode="none", scale_factor=scale/2, resolution=50, 
        opacity=0.2) 
    pts_in.module_manager.scalar_lut_manager.lut.table = cmap
    
    return pts_in, cmap, scale

def build_ground_truth(size, blobs):
    """Build ground truth volumetric image from blobs.
    
    Attributes:
        size: Size given with user-defined dimensions, (x, y, z).
        blobs: Numpy array of segments to display, given as an 
            (n, 4) dimension array, where each segment is in (z, y, x, radius).
    
    Returns:
        3D image binary image array, where 0 is background, and 1 is 
        foreground.
    """
    #print("generating ground truth image of size {}".format(size))
    img3d = np.zeros(size[::-1], dtype=np.uint8)
    for i in range(size[2]):
        blobs_in = blobs[blobs[:, 0] == i]
        for blob in blobs_in:
            rr, cc = draw.circle(*blob[1:4], img3d[i].shape)
            #print("drawing circle of {} x {}".format(rr, cc))
            img3d[i, rr, cc] = 1
    #lib_clrbrain.show_full_arrays()
    #print(img3d)
    return img3d
