# 3D plots from stacks of imaging data
# Author: David Young, 2017
"""Plots the image stack in 3D.

Provides options for drawing as surfaces or points.

Attributes:
    mask_dividend: Maximum number of points to show.
"""

import math
from time import time

import numpy as np
from scipy import interpolate
from scipy import ndimage
from skimage import draw
from skimage import restoration
from skimage import img_as_uint
from skimage import img_as_float
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import transform

from clrbrain import colormaps
from clrbrain import config
from clrbrain import detector
from clrbrain import lib_clrbrain
from clrbrain import plot_support
from clrbrain import segmenter

_MASK_DIVIDEND = 10000.0 # 3D max points

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
        channels = (range(roi.shape[dim_channel]) 
                    if channel is None else [channel])
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
        lib_clrbrain.printv(
            "vmin:", vmin, "vmax:", vmax, "near max:", config.near_max[i])
        # ensures that vmax is at least 50% of near max value of image5d
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
        thresholded = segmenter.segment_rw(roi, size)
    
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
    """Carve image by thresholding and filling in small holes.
    
    Args:
        roi: Image as Numpy array.
        thresh: Value by which to threshold. Defaults to None, in which 
            case a mean threshold will be applied.
        holes_area: Maximum area of holes to fill.
        return_unfilled: True to return the thresholded by unfilled image.
    
    Returns:
        Tuple of ``roi_carved``, the carved image; ``maks``, the mask 
        used to carve; and, if ``return_unfilled`` is True, ``roi_unfilled``, 
        the image after carving but before filling in holes.
    """
    roi_carved = np.copy(roi)
    if thresh is None:
        thresh = filters.threshold_mean(roi_carved)
    mask = roi_carved > thresh
    if holes_area:
        pxs_orig = np.sum(mask)
        mask = morphology.remove_small_holes(mask, holes_area)
        print("{} pxs in holes recovered".format(np.sum(mask) - pxs_orig))
        roi_unfilled = np.copy(roi_carved) if return_unfilled else None
    roi_carved[~mask] = 0
    
    if holes_area and return_unfilled:
        return roi_carved, mask, roi_unfilled
    return roi_carved, mask

def rotate_nd(img_np, angle, axis=0, order=1):
    """Rotate an image of arbitrary dimensions.
    
    This function is essentially a wrapper of 
    :func:``skimage.transform.rotate``, applied to each 2D plane along a 
    given axis for volumetric rotation.
    
    Args:
        img_np: Numpy array.
        angle: Angle by which to rotate.
        axis: Axis along which to rotate, given as an int in standard 
            Numpy axis convention; defaults to 0
        order: Spline interpolation order; defaults to 1.
    
    Returns:
        The rotated image.
    """
    rotated = np.copy(img_np)
    slices = [slice(None)] * img_np.ndim
    for i in range(img_np.shape[axis]):
        # rotate each 2D image in the stack along the given axis
        slices[axis] = i
        img2d = img_np[tuple(slices)]
        img2d = transform.rotate(
            img2d, angle, order=order, mode="constant", preserve_range=True)
        rotated[tuple(slices)] = img2d
    return rotated


def affine_nd(img_np, axis_along, axis_shift, shift, bounds, axis_attach=None, 
              attach_far=False):
    """Affine transform an image of arbitrary dimensions.
    
    Args:
        img_np: Numpy array.
        axis_along: Axis along which to shear, given as an int in standard 
            Numpy axis convention.
        axis_shift: Axis giving the direction in which to shear.
        shift: Tuple of ``(shift_start, shift_end)``, giving the distance 
            in pixels by which to shear while progressing along ``axis_along``.
        bounds: Tuple given as ``((z_start, z_end), (y_start, ...) ...)``, 
            with each value given in pixels demarcating the bounds of the 
            ROI to transform.
        axis_atttach: Axis along which the sheared region will remain 
            attached to the original image to provide a smooth transition 
            in the case of selective affines. Another affine will be 
            performed to along this axis, starting with 0 shift at the 
            point of attachment to the full shift for the given plane at 
            the opposite side. The direction of shearing is based on 
            the corresponding ``bounds`` for this axis. Defaults to None, 
            in which case this affine will be ignored.
    
    Returns:
        The transformed image.
    """
    affined = np.copy(img_np)
    
    def get_shift(sl, shift_n, stop=False):
        # get bounds from a slice and shift while ensuring the indices 
        # stay within bounds
        n_dest = sl.stop if stop else sl.start
        maxn = img_np.shape[axis_shift]
        if n_dest is None:
            n_dest = maxn if stop else 0
        n_img = n_dest
        n_dest += shift_n
        if n_dest < 0:
            # clip dest at 0 while cropping image start
            n_img = abs(n_dest)
            n_dest = 0
        elif n_dest > maxn:
            # clip dest at max while cropping image end
            n_img = -(n_dest - maxn)
            n_dest = maxn
        return n_dest, n_img
    
    
    def affine(axes, shifts_bounds, slices):
        # recursively perform affine transformations for an ROI within an 
        # image along each axis within axes so that sheared regions can 
        # stay connected along at least one axis
        axis = axes[0]
        steps = bounds[axis][1] - bounds[axis][0]
        shifts = np.linspace(*shifts_bounds, steps, dtype=int)
        for j in range(steps):
            # reduce dimension while maintaing the same number/order of axes
            slices_shift = np.copy(slices)
            slices_shift[axis] = bounds[axis][0] + j
            if len(axes) > 1:
                # recursively call next affine transformation with shift 
                # boundaries based on current max shift
                shifts_bounds_next = (0, shifts[j])
                if axes[1] == axis_attach and attach_far:
                    # attach at the far end (higher index) by reversing 
                    # the amount of shift while progressing deeper into 
                    # the image, starting with large shift and decreasing 
                    # to no shift at the attachment point
                    shifts_bounds_next = shifts_bounds_next[::-1]
                affine(axes[1:], shifts_bounds_next, slices_shift)
            else:
                # end-case, where the portion of image that will fit into 
                # its new destination is shifted
                sl = slices_shift[axis_shift]
                start, start_img = get_shift(sl, shifts[j])
                stop, stop_img = get_shift(sl, shifts[j], True)
                slices_img = np.copy(slices_shift)
                slices_img[axis_shift] = slice(start_img, stop_img)
                img = np.copy(img_np[tuple(slices_img)])
                affined[tuple(slices_shift)] = 0
                slices_shift[axis_shift] = slice(start, stop)
                #print(slices_shift)
                affined[tuple(slices_shift)] = img
    
    # list of axes along which to affine transform recursively
    axes = [axis_along]
    if axis_attach is not None: axes.append(axis_attach)
    slices = [slice(bound[0], bound[1]) for bound in bounds]
    print("affine transformations along axes {}".format(axes))
    affine(axes, shift, slices)
    
    return affined

def perimeter_nd(img_np):
    """Get perimeter of image subtracting eroded image from given image.
    
    Args:
        img_np: Numpy array of arbitrary dimensions.
    
    Returns:
        The perimeter as a boolean array where True represents the 
        border that would have been eroded.
    """
    interior = morphology.binary_erosion(img_np)
    img_border = np.logical_xor(img_np, interior)
    #print("perimeter:\n{}".format(img_border))
    return img_border

def exterior_nd(img_np):
    """Get the exterior surrounding foreground, the pixels just beyond 
    the foreground's border, which can be used to find connected neighbors.
    
    Args:
        img_np: Numpy array of arbitrary dimensions with foreground to dilate.
    
    Returns:
        The pixels just outside the image as a boolean array where 
        True represents the border that would have been eroded.
    """
    dilated = morphology.binary_dilation(img_np)
    exterior = np.logical_xor(dilated, img_np)
    return exterior

def compactness(mask_borders, mask_object):
    """Compute the classical compactness, currently supported for 2D or 3D.
    
    For 2D, the equation is given by: perimeter^2 / area. 
    For 3D: area^3 / vol^2.
    
    Args:
        mask_borders: Mask of the borders to find the perimeter (2D) or 
            surface area (3D).
        mask_object: Mask of the object to find the area (2D) or 
            volume (3D). The dimensions of this mask will be used to 
            determine whether to use the 2D or 3D compactness formula.
    
    Returns:
        Compactness metric value. If the sum of ``mask_object`` is 0, 
        return NaN instead.
    """
    # TODO: consider supporting higher dimensions, if available
    n = 1 if mask_object.ndim == 2 else 2
    size_object = np.sum(mask_object).item()
    if size_object > 0:
        # convert to native Python scalars since default Numpy int appears to 
        # overflow for large sums
        compactness = (np.sum(mask_borders).item() ** (n + 1) 
                       / size_object ** n)
    else:
        compactness = np.nan
    return compactness

def signed_distance_transform(borders, mask=None, return_indices=False, 
                              spacing=None):
    """Signed version of Euclidean distance transform where values within a 
    given mask are considered negative.
    
    This version can be applied to the case of a border where 
    distances inside the border, defined as those within the original 
    image from which the borders were construcuted, are considered negative.
    
    Args:
        borders: Borders as a boolean mask, typically where borders are 
            False and non-borders are True.
        mask: Mask where distances will be considered neg, such as a 
            mask of the full image for the borders. Defaults to None, in 
            which case distances will be given as-is.
        return_indices: True if distance transform indices should be given; 
            defaults to False.
        spacing: Grid spacing sequence of same length as number of image 
            axis dimensions; defaults to None.
    
    Returns:
        ``transform`` as the signed distances. If ``return_indices`` is True, 
        ``indices`` is also returned. If no ``mask`` is given, returns 
        the same output as from :meth:``ndimage.distance_transform_edt``.
    """
    if borders is None and mask is not None:
        borders = (1 - perimeter_nd(mask)).astype(bool)
    if return_indices:
        transform, indices = ndimage.distance_transform_edt(
            borders, return_indices=return_indices, sampling=spacing)
    else:
        transform = ndimage.distance_transform_edt(borders, sampling=spacing)
    if mask is not None: transform[mask] *= -1
    if return_indices: return transform, indices
    return transform

def borders_distance(borders_orig, borders_shifted, mask_orig=None, 
                     filter_size=None, gaus_sigma=None, spacing=None):
    """Measure distance between borders.
    
    Args:
        borders_orig: Original borders as a boolean mask.
        borders_shifted: Shifted borders as a boolean mask, which should 
            match the shape of ``borders_orig``.
        mask_orig: Mask of original image for signed distances; defaults 
            to None, in which case all distances will be >= 0.
        filter_size: Size of structuring element to use in filter for 
            smoothing the original border with a closing filter before 
            finding distances. Defaults to None, in which case no filter 
            will be applied.
        gaus_sigma: Low-pass filter distances with Gaussian kernel; defaults 
            to None.
        spacing: Grid spacing sequence of same length as number of image 
            axis dimensions; defaults to None.
    
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
    # find distances to the original borders, inverted to become background, 
    # where neg dists are within mask_orig (or >= 0 if no mask_orig given)
    borders_dist, indices = signed_distance_transform(
        ~borders_orig, mask=mask_orig, return_indices=True, spacing=spacing)
    if gaus_sigma is not None:
        # low-pass filter the distances between borders to remove shifts 
        # likely resulting from appropriate compaction
        borders_dist = filters.gaussian(
            borders_dist, sigma=gaus_sigma, multichannel=False, 
            preserve_range=True)
    # gather the distances corresponding to the shifted border
    dist_to_orig = np.zeros_like(borders_dist)
    dist_to_orig[borders_shifted] = borders_dist[borders_shifted]
    '''
    lib_clrbrain.print_compact(borders_orig, "borders orig", True)
    lib_clrbrain.print_compact(borders_dist, "borders dist", True)
    lib_clrbrain.print_compact(dist_to_orig, "dist_to_orig", True)
    '''
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

def replace_vol(img, vol, center, vol_as_mask=None):
    """Replace a volume within an image, centering on the given coordinates 
    and cropping the input volume to fit.
    
    Args:
        img: Image as a Numpy array into which ``vol`` will be placed. 
            ``img`` will be updated in-place.
        vol: Volume to place in ``img``.
        center: Coordinates of the center of volume, given in z,y,x order.
        vol_as_mask: If ``vol`` should be taken as a mask, where only 
            its True values will replace the corresponding pixels in 
            ``img``, assign this value to the mask locations. Defaults to 
            None, in which case the entire ``vol`` will be assigned.
    
    Returns:
        ``img`` with ``vol`` centered on ``center``.
    """
    dims = vol.ndim
    slices_img = []
    slices_vol = []
    for i in range(dims):
        start_vol = 0
        stop_vol = int(vol.shape[i])
        # center volumes with odd-numbered length, and skew slightly 
        # toward lower values for even-numbered length
        start = int(center[i] - vol.shape[i] // 2)
        stop = start + stop_vol
        # ensure that slices do not exceed bounds of img, also cropping 
        # volume if so
        if start < 0:
            start_vol = abs(start)
            start = 0
        if stop >= img.shape[i]:
            stop_vol -= stop - img.shape[i]
            stop = img.shape[i]
        slices_img.append(slice(start, stop))
        slices_vol.append(slice(start_vol, stop_vol))
    if vol_as_mask is not None:
        # replace vol as a mask
        img[tuple(slices_img)][vol[tuple(slices_vol)]] = vol_as_mask
    else:
        # replace complete vol
        img[tuple(slices_img)] = vol[tuple(slices_vol)]
    return img

def get_label_props(labels_img_np, label_id):
    """Get region properties for a label or set of labels.
    
    Args:
        labels_img_np: Image as Numpy array.
        label_id: Scalar or sequence of scalars of the label IDs to include 
            in the bounding box.
    
    Returns:
        Region properties from :func:``measure.regionprops``.
    """
    if isinstance(label_id, (tuple, list)):
        # sequence of IDs
        label_mask = np.isin(labels_img_np, label_id)
        print("label mask", np.sum(label_mask))
    else:
        # single ID
        label_mask = labels_img_np == label_id
    props = measure.regionprops(label_mask.astype(np.int))
    return props

def get_label_bbox(labels_img_np, label_id):
    """Get bounding box for a label or set of labels.
    
    Assumes that only one set of properties will be found for a given label, 
    returning only the first set.
    
    Args:
        labels_img_np: Image as Numpy array.
        label_id: Scalar or sequence of scalars of the label IDs to include 
            in the bounding box.
    
    Returns:
        Bounding box from :func:``measure.regionprops``. If more than 
        one set of properties are found, only the box from the first 
        property will be returned.
    """
    props = get_label_props(labels_img_np, label_id)
    bbox = None
    if len(props) >= 1: bbox = props[0].bbox
    return bbox

def get_thresholded_regionprops(img_np, threshold=10, sort_reverse=False):
    """Get the region properties for a thresholded image.
    
    Args:
        img_np: Image as a Numpy array.
        threshold: Threshold level; defaults to 10. If None, assume 
            ``img_np`` is already binary.
        sort_reverse: Sort properties from largest to smallest area; 
            defaults to False, in which case sorting is from smallest to 
            largest.
    
    Returns:
        List of ``(prop, area)`` sorted by area.
    """
    thresholded = img_np
    if threshold is not None:
        # threshold the image, removing any small object
        thresholded = img_np > threshold
        thresholded = morphology.remove_small_objects(thresholded, 200)
    # make labels for foreground and get label properties
    labels_props = measure.regionprops(measure.label(thresholded))
    num_props = len(labels_props)
    if num_props < 1:
        return None
    props_sizes = []
    for prop in labels_props:
        props_sizes.append((prop, prop.area))
    props_sizes.sort(key=lambda x: x[1], reverse=sort_reverse)
    return props_sizes

def extend_edge(region, region_ref, threshold, plane_region, planei, 
                largest_only=False):
    """Recursively extend the nearest plane with labels based on the 
    underlying atlas.
    
    Assume that each nearer plane is the same size or smaller than the next 
    farther plane, such as the tapering edge of a specimen. The number of 
    objects to extend is set by the first plane, after which only the 
    largest object in each region will be followed.
    
    Args:
        region: Labels region, which will be updated in-place.
        region_ref: Corresponding reference atlas region.
        threshold: Threshold intensity for ``region_ref``.
        plane_region: Labels template that will be resized for current plane.
        planei: Plane index.
        largest_only: True to only use the property with the largest area; 
            defaults to False.
    """
    if planei < 0: return
    
    # find the bounds of the reference image in the given plane to resize 
    # the corresponding section of the labels image to the bounds of the 
    # reference image in the next plane closer to the edge
    prop_sizes = get_thresholded_regionprops(
        region_ref[planei], threshold=threshold, sort_reverse=largest_only)
    if prop_sizes is None: return
    if largest_only:
        # keep only largest property
        num_props = len(prop_sizes)
        if num_props > 1:
            print("ignoring smaller {} prop(s) in plane {}"
                  .format(num_props - 1, planei))
        prop_sizes = prop_sizes[:1]
    for prop_size in prop_sizes:
        # get the region from the property
        _, slices = get_bbox_region(prop_size[0].bbox)
        prop_region_ref = region_ref[:, slices[0], slices[1]]
        prop_region = region[:, slices[0], slices[1]]
        if plane_region is None:
            # set up the labels in the region to use as template for next 
            # plane; remove ventricular space using empirically determined 
            # selem, which appears to be very sensitive to radius since 
            # values above or below lead to square shaped artifact along 
            # outer sample edges
            prop_plane_region = prop_region[planei]
            prop_plane_region = morphology.closing(
                prop_plane_region, morphology.square(12))
        else:
            # resize prior plane's labels to region's shape and replace region
            prop_plane_region = transform.resize(
                plane_region, prop_region[planei].shape, preserve_range=True, 
                order=0, anti_aliasing=False, mode="reflect")
            prop_region[planei] = prop_plane_region
        # recursively call for each region to follow in next plane, but 
        # only get largest region for subsequent planes in case 
        # new regions appear, where the labels would be unknown
        extend_edge(
            prop_region, prop_region_ref, threshold, prop_plane_region, 
            planei - 1, True)

def crop_to_labels(img_labels, img_ref, mask=None, padding=2):
    """Crop images to match labels volume.
    
    Both labels and reference images will be cropped to match the extent of 
    labels with a small padding region. Reference image pixels outside 
    of a small dilation of the labels mask will be turned to zero.
    
    Args:
        img_labels: Labels image as Numpy array.
        img_ref: Reference image as Numpy array of same shape as that 
            of ``img_labels``.
        mask: Binary Numpy array of same shape as that of ``img_labels`` 
            to use in place of it for determining the extent of cropping. 
            Defaults to None.
        padding: Int size of structuring element for padding the crop 
            region; defaults to 2.
    
    Returns:
        Tuple of ``extracted_labels``, the cropped labels, and 
        ``extracted_ref``, the cropped reference image, extracting only 
        pixels corresponding to the labels.
    """
    if mask is None:
        # default to get bounding box of all labels, assuming 0 is background
        mask = img_labels != 0
    props = measure.regionprops(mask.astype(np.int))
    if not props or len(props) < 1: return
    shape, slices = get_bbox_region(
        props[0].bbox, padding=5, img_shape=img_labels.shape)
    
    # crop images to bbox and erase reference pixels just outside of 
    # corresponding labels; dilate to include immediate neighborhood
    extracted_labels = img_labels[tuple(slices)]
    extracted_ref = img_ref[tuple(slices)]
    extracted_mask = mask[tuple(slices)]
    mask_dil = morphology.binary_dilation(
        extracted_mask, morphology.ball(padding))
    extracted_ref[~mask_dil] = 0
    return extracted_labels, extracted_ref

def interpolate_contours(bottom, top, fracs):
    """Interpolate contours between two planes.
    
    Args:
        bottom: bottom plane as an binary mask.
        top: top plane as an binary mask.
        fracs: List of fractions between 0 and 1, inclusive, at which to 
            interpolate contours. 0 corresponds to the bottom plane, while 
            1 is the top.
    
    Returns:
        Array with each plane corresponding to the interpolated plane at the 
        given fraction.
    """
    # convert planes to contour distance maps, where pos distances are 
    # inside the original image
    bottom = -1 * signed_distance_transform(None, bottom.astype(bool))
    top = -1 * signed_distance_transform(None, top.astype(bool))
    r, c = top.shape

    # merge dist maps into an array with shape (2, r, c) and prep 
    # meshgrid and output array
    stack = np.stack((bottom, top))
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    grid = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r * c, 2))
    interpolated = np.zeros((len(fracs), r, c), dtype=bool)
    
    for i, frac in enumerate(fracs):
        # interpolate plane at given fraction between bottom and top planes
        xi = np.c_[np.full(r * c, frac + 1), grid]
        out = interpolate.interpn(points, stack, xi)
        out = out.reshape((r, c))
        out = out > 0
        interpolated[i] = out
    
    return interpolated

def interpolate_label_between_planes(labels_img, label_id, axis, bounds):
    """Interpolate between two planes for the given labeled region. 
    
    Assume that the given label ID has only been extended and not erased 
    within the given image. This image will be updated in-place.
    
    Args:
        labels_img: Labels image as a Numpy array in z,y,x dimensions.
        label_id: ID of label whose planes will be replaced.
        axis: Axis along which planes will be selected.
        bounds: 2-element list demarcating the planes between which to 
            interpolate. The list will be sorted, and the lower bound 
            will mark the starting plane, while the upper bound will mark 
            the ending plane, inclusive.
    """
    # get bounding box for label to limit the volume needed to resize
    bbox = get_label_bbox(labels_img, label_id)
    if bbox is None: return
    shape, slices = get_bbox_region(bbox)
    region = labels_img[tuple(slices)]
    
    # prep region to interpolate by taking only the planes at bounds
    bounds_sorted = np.copy(bounds)
    bounds_sorted.sort()
    offset = slices[axis].start
    bounds_sorted -= offset
    slices_planes = [slice(None)] * labels_img.ndim
    slices_planes[axis] = slice(
        bounds_sorted[0], bounds_sorted[1] + 1, 
        bounds_sorted[1] - bounds_sorted[0])
    
    # interpolate contours of each plane between bounds (inclusive)
    region_planes = region[tuple(slices_planes)]
    region_planes_mask = np.zeros_like(region_planes)
    region_planes_mask[region_planes == label_id] = 1
    slices_plane = [slice(None)] * 3
    slices_plane[axis] = 0
    start = region_planes_mask[tuple(slices_plane)]
    slices_plane[axis] = 1
    end = region_planes_mask[tuple(slices_plane)]
    interpolated = interpolate_contours(
        start, end, np.linspace(0, 1, bounds_sorted[1] - bounds_sorted[0] + 1))
    # interpolate_contours puts the bounded planes at the ends of a z-stack, 
    # so need to transform back to the original orientation
    if axis == 1:
        interpolated = np.swapaxes(interpolated, 0, 1)
    elif axis == 2:
        interpolated = np.moveaxis(interpolated, 0, -1)
    
    # fill interpolated areas with label and replace corresponding sub-images; 
    # could consider in-painting if labels were removed, but could slightly 
    # decrease other areas
    slices_planes[axis] = slice(bounds_sorted[0], bounds_sorted[1] + 1)
    region_within_bounds = region[tuple(slices_planes)]
    region_within_bounds[interpolated] = label_id
    region[tuple(slices_planes)] = region_within_bounds
    labels_img[tuple(slices)] = region

def build_heat_map(shape, coords):
    """Build a heat map for an image based on point placement within it.
    
    The heat map is scaled at the level of ``shape``, generally assuming 
    that ``coords`` have been scaled from a larger size. In other words, 
    the heat map will show the density at the level of its pixels and 
    can be further rescaled/resized to show density at different resolutions.
    
    Args:
        shape: Shape of image that contains ``coords``.
        coords: Array of coordinates of points. The array 
            should have shape (n, m), where n = number of coordinate sets, 
            and m = number of coordinate dimensions.
    
    Returns:
        An image of shape ``shape`` with values corresponding 
        to the number of point occurrences at each pixel.
    """
    # get counts of points at the same coordinate as a measure of density
    coords_unique, coords_count = np.unique(
        coords, return_counts=True, axis=0)
    coordsi = lib_clrbrain.coords_for_indexing(coords_unique)
    dtype = lib_clrbrain.dtype_within_range(
        0, np.amax(coords_count), True, False)
    heat_map = np.zeros(shape, dtype=dtype)
    heat_map[tuple(coordsi)] = coords_count
    return heat_map

def laplacian_of_gaussian_img(img, sigma=5, labels_img=None, thresh=None):
    """Generate a Laplacian of Gaussian (LoG) image.
    
    Args:
        img: Image as Numpy array from which the LoG will be generated.
        sigma: Sigma for Gaussian filters; defaults to 5.
        labels_img: Labels image as Numpy array in same shape as ``img``, 
            to use to assist with thresholding out background. Defaults 
            to None to skip background removal.
        thresh: Threshold of atlas image to find background solely from 
            atlas, without using labels; ``labels_img`` will be 
            ignored if ``thresh`` is given. Defaults to None.
             
    """
    # apply Laplacian of Gaussian filter (sequentially)
    img_log = filters.gaussian(img, sigma)
    img_log = filters.laplace(img_log)
    vmin, vmax = np.percentile(img_log, (2, 98))
    img_log = np.clip(img_log, vmin, vmax)
    
    # remove background signal to improve zero-detection at outside borders
    mask = None
    if thresh is not None:
        # simple threshold of atlas to find background
        mask = img > thresh
    elif labels_img is not None:
        # remove signal below threshold while keeping any signal in labels; 
        # TODO: consider using atlas rather than LoG image
        mask = segmenter.mask_atlas(img_log, labels_img)
    if mask is not None:
        img_log[~mask] = np.amin(img_log)
    
    return img_log

def zero_crossing(img, filter_size):
    """Apply a zero-crossing detector to an image, such as an image 
    produced by a Laplacian of Gaussian.
    
    Args:
        img: Image as a Numpy array.
        filter_size: Size of structuring element, where larger sizes 
            give thicker edges.
    
    Returns:
        Array of same size as ``img`` as a mask of edges.
    """
    selem = morphology.ball(filter_size)
    eroded = morphology.erosion(img, selem)
    dilated = morphology.dilation(img, selem)
    # find pixels of border transition, where eroded or dilated pixels 
    # switch signs compared with original image
    crossed = np.logical_or(
        np.logical_and(img > 0, eroded < 0), 
        np.logical_and(img < 0, dilated > 0))
    return crossed

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

def plot_3d_surface(roi, scene_mlab, channel, segment=False, flipud=False):
    """Plots areas with greater intensity as 3D surfaces.
    
    Args:
        roi: Region of interest.
        scene_mlab: ``MayaviScene.mlab`` attribute to draw the contour. Any 
            current image will be cleared first.
        segment: True to denoise and segment ``roi`` before displaying, 
            which may remove artifacts that might otherwise lead to 
            spurious surfaces. Defaults to False.
        flipud: True to invert blobs along z-axis to match handedness 
            of Matplotlib with z progressing upward; defaults to False.
    """
    # Plot in Mayavi
    #mlab.figure()
    print("viewing 3D surface")
    pipeline = scene_mlab.pipeline
    scene_mlab.clf()
    settings = config.process_settings
    if flipud:
        # invert along z-axis to match handedness of Matplotlib with z up
        roi = np.flipud(roi)
    
    # saturate to remove noise and normalize values
    roi = saturate_roi(roi, settings["clip_vmax"], channel=channel)
    #roi = np.clip(roi, 0.2, 0.8)
    #roi = restoration.denoise_tv_chambolle(roi, weight=0.1)
    
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
                walker = segmenter.segment_rw(roi_show, i, vmin=0.6, vmax=0.7)
                roi_show *= np.subtract(walker[0], 1)
            else:
                print("deferring segmentation as {} px is above threshold"
                      .format(num_pixels))
        
        # ROI is in (z, y, x) order, so need to transpose or swap x,z axes
        roi_show = np.transpose(roi_show)
        surface = pipeline.scalar_field(roi_show)
        
        # Contour -> Surface pipeline
        
        # create the surface
        surface = pipeline.contour(surface)
        # remove many more extraneous points
        surface = pipeline.user_defined(surface, filter="SmoothPolyDataFilter")
        surface.filter.number_of_iterations = 400
        surface.filter.relaxation_factor = 0.015
        # distinguishing pos vs neg curvatures?
        surface = pipeline.user_defined(surface, filter="Curvatures")
        surface = scene_mlab.pipeline.surface(surface)
        module_manager = surface.module_manager
        module_manager.scalar_lut_manager.data_range = np.array([-2, 0])
        module_manager.scalar_lut_manager.lut_mode = "gray"
        
        '''
        # Surface pipleline with contours enabled (similar to above?)
        surface = pipeline.contour_surface(
            surface, color=(0.7, 1, 0.7), line_width=6.0)
        surface.actor.property.representation = 'wireframe'
        #surface.actor.property.line_width = 6.0
        surface.actor.mapper.scalar_visibility = False
        '''
        
        '''
        # IsoSurface pipeline
        
        # uses unique IsoSurface module but appears to have 
        # similar output to contour_surface
        surface = pipeline.iso_surface(surface)
        
        # limit contours for simpler surfaces including smaller file sizes; 
        # TODO: consider making settable as arg or through profile
        surface.contour.number_of_contours = 1
        try:
            # increase min to further reduce complexity
            surface.contour.minimum_contour = 0.5
            surface.contour.maximum_contour = 0.8
        except Exception as e:
            print(e)
            print("ignoring min/max contour for now")
        '''
        
        isotropic = settings["isotropic_vis"]
        if isotropic is not None:
            # adjust for anisotropy
            surface.actor.actor.scale = isotropic[::-1]
    
    print("time to render 3D surface: {}".format(time() - time_start))
    
def plot_3d_points(roi, scene_mlab, channel, flipud=False):
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
        flipud: True to invert blobs along z-axis to match handedness 
            of Matplotlib with z progressing upward; defaults to False.
    
    Returns:
        True if points were rendered, False if no points to render.
    """
    print("plotting as 3D points")
    scene_mlab.clf()
    settings = config.process_settings
    
    # streamline the image
    if roi is None or roi.size < 1: return False
    roi = saturate_roi(roi, 98.5, channel)
    roi = np.clip(roi, 0.2, 0.8)
    roi = restoration.denoise_tv_chambolle(roi, weight=0.1)
    
    # separate parallel arrays for each dimension of all coordinates for
    # Mayavi input format, with the ROI itself given as a 1D scalar array 
    time_start = time()
    shape = roi.shape
    z = np.ones((shape[0], shape[1] * shape[2]))
    if flipud:
        # invert along z-axis to match handedness of Matplotlib with z up
        z *= -1
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
            try:
                thresh = filters.threshold_otsu(roi_show, 64)
            except ValueError as e:
                thresh = np.median(roi_show)
                print("could not determine Otsu threshold, taking median "
                      "({}) instead".format(thresh))
            thresh *= settings["points_3d_thresh"]
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
            roi_show_1d, mode="sphere", 
            scale_mode="scalar", mask_points=mask, line_width=1.0, vmax=1.0, 
            vmin=0.0, transparent=True)
        cmap = colormaps.get_cmap(config.cmaps, i)
        if cmap is not None:
            pts.module_manager.scalar_lut_manager.lut.table = cmap(
                range(0, 256)) * 255
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

def show_blobs(segments, mlab, segs_in_mask, show_shadows=False, flipud=False):
    """Shows 3D blob segments.
    
    Args:
        segments: Labels from 3D blob detection method.
        mlab: Mayavi object.
        segs_in_mask: Boolean mask for segments within the ROI; all other 
            segments are assumed to be from padding and border regions 
            surrounding the ROI.
        show_shadows: True if shadows of blobs should be depicted on planes 
            behind the blobs; defaults to False.
        flipud: True to invert blobs along z-axis to match handedness 
            of Matplotlib with z progressing upward; defaults to False.
    
    Returns:
        A 3-element tuple containing ``pts_in``, the 3D points within the 
        ROI; ``cmap'', the random colormap generated with a color for each 
        blob, and ``scale``, the current size of the points.
    """
    if segments.shape[0] <= 0:
        return None, None, 0
    settings = config.process_settings
    segs = np.copy(segments)
    if flipud:
        # invert along z-axis to match handedness of Matplotlib with z up
        segs[:, 0] *= -1
    isotropic = settings["isotropic_vis"]
    if isotropic is not None:
        # adjust position based on isotropic factor
        segs[:, :3] = np.multiply(segs[:, :3], isotropic)
    
    radii = segs[:, 3]
    scale = 5 if radii is None else np.mean(np.mean(radii) + np.amax(radii))
    print("blob point scaling: {}".format(scale))
    # colormap has to be at least 2 colors
    segs_in = segs[segs_in_mask]
    num_colors = segs_in.shape[0] if segs_in.shape[0] >= 2 else 2
    cmap = colormaps.discrete_colormap(num_colors, 170, True, config.seed)
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

def build_ground_truth(img3d, blobs, ellipsoid=False, labels=None, 
                       spacing=None):
    """Build ground truth volumetric image from blobs.
    
    Attributes:
        img3d: Image as 3D Numpy array in which to store results
        blobs: Numpy array of segments to display, given as an 
            (n, 4) dimension array, where each segment is in (z, y, x, radius).
        ellipsoid: True to draw blobs as ellipsoids; defaults to False.
        labels: Array of labels the same length as ``blobs`` to assign 
            as the values for each ground truth; defaults to None to 
            assign a default value of 1 instead.
        spacing: Spacing by which to multiply blobs` radii; defaults to None, 
            in which case each blob's radius will be used for all dimensions.
    
    Returns:
        ``img3d`` with ground drawn as circles or ellipsoids.
    """
    if ellipsoid:
        # draw blobs as ellipses
        for i, blob in enumerate(blobs):
            if spacing is None:
                centroid = np.repeat(blob[3], 3)
            else:
                # multiply spacing directly rather than using in ellipsoid 
                # function since the fn does not appear to place the 
                # ellipsoide in the center of the array
                centroid = np.multiply(blob[3], spacing)
            ellip = draw.ellipsoid(*centroid)
            label = True if labels is None else labels[i]
            replace_vol(img3d, ellip, blob[:3], vol_as_mask=label)
    else:
        # draw blobs as circles only in given z-planes
        if labels is None: labels = np.ones(len(blobs), dtype=int)
        for i in range(img3d.shape[0]):
            mask = blobs[:, 0] == i
            blobs_in = blobs[mask]
            labels_in = labels[mask]
            for blob, label in zip(blobs_in, labels_in):
                rr, cc = draw.circle(*blob[1:4], img3d[i].shape)
                #print("drawing circle of {} x {}".format(rr, cc))
                img3d[i, rr, cc] = label
    return img3d


def _test_interpolate_between_planes():
    '''
    img = np.zeros((2, 4, 4), dtype=int)
    img[0, :2, :1] = 1
    img[1, :2, :] = 1
    '''
    img = np.zeros((5, 4, 4), dtype=int)
    img[0, :2, :1] = 1
    img[4, :2, :] = 1
    img[4, 1, 3] = 0
    print(img)
    #img = interp_shape(img[0], img[-1], np.linspace(0, 1, 5))
    interpolate_label_between_planes(img, 1, 0, [0, 4])
    print(img)

if __name__ == "__main__":
    _test_interpolate_between_planes()
