# Computer vision library functions for n-dimensions.
# Author: David Young, 2018, 2019
"""Computer vision library functions for n-dimensions.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import interpolate
from scipy import ndimage
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import transform

from magmap.settings import config
from magmap.cv import segmenter
from magmap.io import libmag

_logger = config.logger.getChild(__name__)


def in_paint(roi, to_fill):
    """In-paint to interpolate values into pixels to fill from nearest 
    neighbors.
    
    Args:
        roi (:class:`numpy.ndarray`): ROI in which to fill pixels.
        to_fill (:class:`numpy.ndarray`): Boolean array of same shape as
            ``roi`` where True values designate the pixels to fill.
    
    Returns:
        :class:`numpy.ndarray`: Copy of ROI with pixels corresponding to
        ``to_fill`` filled with nearest neighbors.
    
    """
    indices = ndimage.distance_transform_edt(
        to_fill, return_distances=False, return_indices=True)
    filled = roi[tuple(indices)]
    return filled


def carve(roi, thresh=None, holes_area=None, return_unfilled=False):
    """Carve image by thresholding and filling in small holes.
    
    Args:
        roi (:class:`numpy.ndarray`): Image as Numpy array.
        thresh (float): Value by which to threshold. Defaults to None, in which 
            case a mean threshold will be applied.
        holes_area (int): Maximum area of holes to fill; defaults to None
            to leave unfilled.
        return_unfilled (bool): True to return the carved image without
            any filling; defaults to False.
    
    Returns:
        :class:`numpy.ndarray`, :class:`numpy.ndarray`: The carved image and
        the mask used to carve this image. If ``return_unfilled`` is True,
        also returns the carved ``roi`` without ``holes_area`` applied. All
        arrays are of the same shape as that of ``roi``.
    
    """
    roi_carved = np.copy(roi)
    if thresh is None:
        thresh = filters.threshold_mean(roi_carved)
    mask = roi_carved > thresh
    roi_unfilled = roi_carved
    if holes_area:
        if return_unfilled:
            roi_unfilled = np.copy(roi_carved)
            roi_unfilled[~mask] = 0
        pxs_orig = np.sum(mask)
        mask = morphology.remove_small_holes(mask, holes_area)
        print("{} pxs in holes recovered".format(np.sum(mask) - pxs_orig))
    roi_carved[~mask] = 0
    
    if return_unfilled:
        return roi_carved, mask, roi_unfilled
    return roi_carved, mask


def rotate_nd(
        img_np: np.ndarray, angle: float, axis: int = 0, order: int = 1,
        resize: bool = False) -> np.ndarray:
    """Rotate an image of arbitrary dimensions along a given axis.
    
    This function is essentially a wrapper of
    :func:`skimage.transform.rotate`, applied to each 2D plane along a
    given axis for volumetric rotation.
    
    Args:
        img_np: 2D or higher dimensional NumPy array.
        angle: Angle by which to rotate.
        axis: Axis along which to rotate, given as an int in standard
            Numpy axis convention; defaults to 0
        order: Spline interpolation order; defaults to 1.
        resize: True to resize the output image to avoid any
            image cropping; defaults to False.
    
    Returns:
        The rotated image.
    
    """
    is_2d = img_np.ndim == 2
    if is_2d:
        # wrap in another dim to make 3D
        img_np = img_np[None]
    
    slices = [slice(None)] * img_np.ndim
    imgs = []
    for i in range(img_np.shape[axis]):
        # rotate each 2D image in the stack along the given axis
        slices[axis] = i
        img2d = img_np[tuple(slices)]
        imgs.append(transform.rotate(
            img2d, angle, order=order, mode="constant", preserve_range=True,
            resize=resize))
    
    if resize:
        # find output shape based on max plane size, allowing rotated images
        # to be of different sizes such as for progressive rotation, although
        # the current implementation of `rotate` outputs planes of same shape
        shapes = [img.shape for img in imgs]
        max_2d = np.amax(shapes, axis=0)
        rot_shape = list(max_2d)
        rot_shape.insert(axis, len(imgs))
        rotated = np.zeros(rot_shape, dtype=img_np.dtype)
        for i, img in enumerate(imgs):
            # center rotated image in the output plane
            offset = np.subtract(max_2d, img.shape) // 2
            slices = [slice(offset[0], offset[0]+img.shape[0]),
                      slice(offset[1], offset[1]+img.shape[1])]
            slices.insert(axis, i)
            rotated[tuple(slices)] = img
    else:
        # output with same shape as that of original image
        rotated = np.copy(img_np)
        for i, img in enumerate(imgs):
            slices[axis] = i
            rotated[tuple(slices)] = img
    
    if is_2d:
        # convert back to 2D
        rotated = rotated[0]
    return rotated


def rotate90(roi: np.ndarray, rotate: int,
             axes: Optional[Sequence[int]] = None,
             multichannel: bool = False) -> np.ndarray:
    """Rotate an image by increments of 90 degrees.
    
    Serves as a wrapper for :meth:`numpy.rot90` with default rotation in
    the xy plane.
    
    Args:
        roi: Image as a 3D+/-channel array. Can be None to return as-is.
        rotate: Number of times to rotate 90 degrees.
        axes: Sequence of two axes defining the plane to rotate; defaults to
            None to use ``[-2, -1]``, the 2nd to last and last axes.
        multichannel: True if the image is multichannel; defaults to False.
            Only used if ``axes`` contains negative axis indices.

    Returns:
        Rotated image.

    """
    if rotate is None:
        return roi
    if axes is None:
        # default to using the last 2 axes (xy plane)
        ax = [-2, -1]
    else:
        ax = list(axes)
    for i, a in enumerate(ax):
        if a < 0:
            # wrap neg axes to the end of the axes
            ax[i] += roi.ndim
            if multichannel:
                # skip the channel axis
                ax[i] -= 1
    roi = np.rot90(roi, libmag.get_if_within(rotate, 0), ax)
    return roi


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
        axis_attach: Axis along which the sheared region will remain 
            attached to the original image to provide a smooth transition 
            in the case of selective affines. Another affine will be 
            performed to along this axis, starting with 0 shift at the 
            point of attachment to the full shift for the given plane at 
            the opposite side. The direction of shearing is based on 
            the corresponding ``bounds`` for this axis. Defaults to None, 
            in which case this affine will be ignored.
        attach_far: True to attach from the opposite, farther side along 
            ``axis_attach``, from higher to lower indices; defaults to False.
    
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


def perimeter_nd(
        img_np: np.ndarray, largest_only: bool = False,
        footprint: Optional[Sequence[int]] = None) -> np.ndarray:
    """Get perimeter of image subtracting eroded image from given image.
    
    Args:
        img_np: Numpy array of arbitrary dimensions.
        largest_only : True to retain only the largest connected
            component, typically the outer border; defaults to False.
        footprint: Structure element for eroding the interior, which sets
            the border thickness; defaults to None.
    
    Returns:
        The perimeter as a boolean array where True
        represents the border that would have been eroded.
    
    """
    # get interior by eroding the image, where footprint determines the border
    # thickness, and subtracting the interior from the full image
    interior = morphology.binary_erosion(img_np, footprint)
    img_border = np.logical_xor(img_np, interior)
    
    if largest_only:
        # retain only the largest perimeter based on pixel count
        labeled = measure.label(img_border)
        labels, counts = np.unique(labeled[labeled != 0], return_counts=True)
        labels = labels[np.argsort(counts)]
        img_border[labeled != labels[-1]] = False
    
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


def surface_area_3d(
        img_np: np.ndarray, level: float = 0.0,
        spacing: Optional[Sequence[float]] = None) -> float:
    """Measure the surface area for a 3D volume.
    
    Wrapper for :func:`measure.marching_cubes_lewiner` and 
    :func:`measure.mesh_surface_area`.
    
    Args:
        img_np: 3D image array, which can be a mask.
        level: Contour value for :func:`measure.marching_cubes_lewiner`;
            defaults to 0.0.
        spacing: Sequence of voxel spacing in same order as for ``img_np``;
            defaults to None, which will use a value of ``np.ones(3)``.

    Returns:
        Surface area in the coordinate units squared.

    """
    if spacing is None:
        spacing = np.ones(3)
    try:
        if hasattr(measure, "marching_cubes_lewiner"):
            # skimage 0.14 removed `marching_cubes`
            fn_marching = measure.marching_cubes_lewiner
        else:
            # skimage 0.19 removed `marching_cubes_lewiner` and went back to
            # `marching_cubes`
            fn_marching = measure.marching_cubes
        verts, faces, normals, vals = fn_marching(
            img_np, level=level, spacing=spacing)
        return measure.mesh_surface_area(verts, faces)
    except ValueError as ve:
        # marching cubes gives this error if img_np is all the same value,
        # such as a mask that is completely True
        _logger.exception(ve)
        # if np.amin(img_np) == np.amax(img_np):
        if len(np.unique(img_np)) == 1:
            raise ValueError(
                "All values in array are the same value, please check "
                "threshold for array")
        raise ve
    except RuntimeError as e:
        _logger.error(e)
    return np.nan


def compactness_count(mask_borders, mask_object):
    """Compute compactness based on simple boundary and size counts.
    
    Args:
        mask_borders (:obj:`np.ndarray`): Mask of the borders to find the 
            perimeter (2D) or surface area (3D) by simple boundary pixel 
            count.
        mask_object (:obj:`np.ndarray`): Mask of the object to find the area 
            (2D) or volume (3D).
    
    Returns:
        Tuple of compactness metric value from :func:`calc_compactness`, 
        borders measurement, and object size.
    """
    # TODO: consider deprecating since currently unused
    # simple boundary pixel count, which generally underestimates the 
    # true border measurement
    borders_meas = np.sum(mask_borders)
    size_object = np.sum(mask_object)
    # convert vol to native Python scalar since np int may not be large enough
    compactness = calc_compactness(
        mask_object.ndim, borders_meas.item(), size_object.item())
    return compactness, borders_meas, size_object


def compactness_3d(img_np, spacing=None):
    """Compute compactness with 3D area based on :func:`surface_area_3d` 
    and support for anisotropy.

    Args:
        img_np (:obj:`np.ndarray`): Mask of the object.
        spacing (List[float]): Sequence of voxel spacing in same order 
            as for ``img_np``; defaults to None.

    Returns:
        Tuple of compactness metric value from :func:`calc_compactness`, 
        surface area, and volume.
    """
    area = surface_area_3d(img_np, spacing=spacing)
    vol = np.sum(img_np)
    if spacing is not None:
        vol *= np.prod(spacing)
    # convert vol to native Python scalar since np int may not be large enough
    compactness = calc_compactness(img_np.ndim, area, vol.item())
    return compactness, area, vol


def calc_compactness(ndim, size_borders, size_object):
    """Compute the classical compactness, currently supported for 2D or 3D.

    For 2D, the equation is given by: perimeter^2 / area. 
    For 3D: area^3 / vol^2.

    Args:
        ndim (int): Number of dimensions, used to determine whether to use 
            the 2D or 3D compactness formula.
        size_borders (float): Perimeter (2D) or surface area (3D) .
        size_object (float): Area (2D) or volume (3D).

    Returns:
        Compactness metric value. If ``size_object`` is 0, return NaN instead.
    """
    compact = np.nan
    if size_object > 0:
        compact = size_borders ** ndim / size_object ** (ndim - 1)
    return compact


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
    """Get slices from a bounding box from which to extract the ROI 
    from the original image as a view and with an optional padding.
    
    Args:
        bbox: Bounding box as given by :func:``get_label_bbox``.
        padding: Int of padding in pixels outside of the box to include; 
            defaults to 0.
        img_shape: Sequence of maximum output coordinates, where any 
            end coordinate will be truncated to the corresponding 
            value. Defaults to None.
    
    Returns:
        Tuple of the ROI shape and list of ROI slices containing the 
        start and end indices of the ROI along each axis.
    """
    dims = len(bbox) // 2  # bbox has min vals for each dim, then maxes
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
    props = measure.regionprops(label_mask.astype(int))
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
    return props[0].bbox if len(props) >= 1 else None


def extract_region(
        labels_img: np.ndarray, label_id: int
) -> Tuple[Optional[np.ndarray], Optional[List[slice]]]:
    """Wrapper for extracting a labeled region.
    
    Args:
        labels_img: Labels image as an integer array.
        label_id: ID of label to extract.

    Returns:
        Tuple of the bounding box of the region containing ``label_id`` and
        the list of slices in ``labels_img`` defining the extracted indices.
        If no bounding box was found, each value is returned as None. If
        multiple separate regions are found, only the first is returned.

    """
    bbox = get_label_bbox(labels_img, label_id)
    if bbox is None:
        return None, None
    _, slices = get_bbox_region(bbox)
    return labels_img[tuple(slices)], slices


def meas_region(mask, res):
    """Measure the dimensions of a masked region.

    Args:
        mask (:obj:`np.ndarray`): Binary array as a region mask. Assumes
            that this mask defines a single foreground region.
        res (List[float]): Sequence of resolutions/spacing by dimension in
            ``mask``.

    Returns:
        List[float], float, List: Dimensions of the bounding box of the
        first region defined by ``mask`` in the physical units of ``res``;
        total volume of the mask in physical units; the region properties
        of ``mask`` as given by :meth:`measure.regionprops`.

    """
    props = measure.regionprops(mask.astype(int))
    shape = get_bbox_region(props[0].bbox)[0]
    meas = np.multiply(shape, res)
    vol = np.prod(res) * np.sum(mask)
    return meas, vol, props


def get_thresholded_regionprops(img_np, threshold=10, sort_reverse=False):
    """Get the region properties for a thresholded image.
    
    Args:
        img_np (:obj:`np.ndarray`): Image array.
        threshold (int, float): Threshold level; defaults to 10. If None, 
            assume ``img_np`` is already binary.
        sort_reverse (bool): Sort properties from largest to smallest area; 
            defaults to False, in which case sorting is from smallest to 
            largest.
    
    Returns:
        List of ``(prop, area)``, sorted by area from smallest to largest 
        unless reversed by ``sort_reverse``.
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


def crop_to_labels(img_labels, img_ref, mask=None, dil_size=2, padding=5):
    """Crop images to match the extent of a labels volume.
    
    Both labels and reference images will be cropped to match the extent of 
    labels with a small padding region. Reference image pixels outside 
    of a small dilation of the labels mask will be turned to zero.
    
    Args:
        img_labels (:obj:`np.ndarray`): Labels image.
        img_ref (:obj:`np.ndarray`): Reference image of same shape as that 
            of ``img_labels``.
        mask (:obj:`np.ndarray`, optional): Binary array of same shape as 
            that of ``img_labels`` to use in place of it for determining 
            the extent of cropping. Defaults to None. Will not be used
            to mask out signal within the cropped in volume.
        dil_size (int, optional): Size of structuring element for dilating 
            the crop region; defaults to 2.
        padding (int, optional): Size of padding around the mask 
            bounding box; defaults to 5.
    
    Returns:
        Tuple of ``extracted_labels``, the cropped labels image; 
        ``extracted_ref``, the cropped reference image, extracting only 
        pixels corresponding to the labels; and ``slices``, a list of 
        the bounding box slices by which the images were cropped.
    """
    if mask is None:
        # default to get bounding box of all labels, assuming 0 is background
        mask = img_labels != 0
    props = measure.regionprops(mask.astype(int))
    if not props or len(props) < 1: return
    shape, slices = get_bbox_region(
        props[0].bbox, padding=padding, img_shape=img_labels.shape)
    
    # crop images to bbox and erase reference pixels just outside of 
    # corresponding labels; dilate to include immediate neighborhood
    extracted_labels = img_labels[tuple(slices)]
    extracted_ref = img_ref[tuple(slices)]
    extracted_mask = mask[tuple(slices)]
    remove_bg_from_dil_fg(
        extracted_ref, extracted_mask, morphology.ball(dil_size))
    return extracted_labels, extracted_ref, slices


def remove_bg_from_dil_fg(img, mask, selem):
    """Remove background by converting the area outside of a dilated 
    foreground mask to background.
    
    Args:
        img (:obj:`np.ndarray`): Image array whose pixels outside of the 
            dilated ``mask`` will be zeroed in-place.
        mask (:obj:`np.ndarray`): Foreground as a binary array of 
            same size as ``img`` that will be dilated. 
        selem (:obj:`np.ndarray`): Array of the same dimensions as ``img`` 
            to serve as the structuring element for dilation.

    """
    mask_dil = morphology.binary_dilation(mask, selem)
    img[~mask_dil] = 0


def interpolate_contours(bottom, top, fracs):
    """Interpolate contours between two planes.
    
    Args:
        bottom (:obj:`np.ndarray`): Bottom plane as an binary mask.
        top (:obj:`np.ndarray`): Top plane as an binary mask.
        fracs (List[float]): List of fractions from 0 to 1, inclusive, at
            which to interpolate contours. 0 corresponds to the bottom plane,
            and 1 is the top plane.
    
    Returns:
        :obj:`np.ndarray`: Array with each plane corresponding to the
        interpolated plane at the fractions corresponding to ``fracs``.
    """
    # convert planes to contour distance maps, where pos distances are 
    # inside the original image
    bottom = -1 * signed_distance_transform(None, bottom.astype(bool))
    top = -1 * signed_distance_transform(None, top.astype(bool))
    r, c = top.shape

    # merge dist maps into an array with shape (2, r, c) and prep 
    # meshgrid and output array
    stack = np.stack((bottom, top))
    points = (np.r_[0, 1], np.arange(r), np.arange(c))
    grid = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r * c, 2))
    interpolated = np.zeros((len(fracs), r, c), dtype=bool)
    
    for i, frac in enumerate(fracs):
        # interpolate plane at given fraction between bottom and top planes
        xi = np.c_[np.full(r * c, frac), grid]
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
    
    # interpolate contours in each plane between bounds (exclusive)
    region_planes = region[tuple(slices_planes)]
    region_planes_mask = np.zeros_like(region_planes)
    region_planes_mask[region_planes == label_id] = 1
    slices_plane = [slice(None)] * 3
    slices_plane[axis] = 0
    start = region_planes_mask[tuple(slices_plane)]
    slices_plane[axis] = 1
    end = region_planes_mask[tuple(slices_plane)]
    # interpolate evenly spaced fractions from 0-1, excluding the ends
    interpolated = interpolate_contours(
        start, end,
        np.linspace(0, 1, bounds_sorted[1] - bounds_sorted[0] + 1)[1:-1])
    # interpolate_contours puts the bounded planes at the ends of a z-stack, 
    # so need to transform back to the original orientation
    if axis == 1:
        interpolated = np.swapaxes(interpolated, 0, 1)
    elif axis == 2:
        interpolated = np.moveaxis(interpolated, 0, -1)
    
    # fill interpolated areas with label and replace corresponding sub-images; 
    # could consider in-painting if labels were removed, but could slightly 
    # decrease other areas
    slices_planes[axis] = slice(bounds_sorted[0] + 1, bounds_sorted[1])
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
        :obj:`np.ndaarry`: An image of shape ``shape`` with values
        corresponding to the number of point occurrences at each pixel.
    """
    if coords is not None and len(coords) > 0:
        # get counts of points at the same coordinate as a measure of density
        coords_unique, coords_count = np.unique(
            coords, return_counts=True, axis=0)
        coordsi = libmag.coords_for_indexing(coords_unique)
        dtype = libmag.dtype_within_range(0, np.amax(coords_count), True, False)
        heat_map = np.zeros(shape, dtype=dtype)
        heat_map[tuple(coordsi)] = coords_count
    else:
        # generate an array with small int type if no coords are available
        heat_map = np.zeros(shape, dtype=np.uint8)
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
    selem = get_selem(img.ndim)(filter_size)
    eroded = morphology.erosion(img, selem)
    dilated = morphology.dilation(img, selem)
    # find pixels of border transition, where eroded or dilated pixels 
    # switch signs compared with original image
    crossed = np.logical_or(
        np.logical_and(img > 0, eroded < 0), 
        np.logical_and(img < 0, dilated > 0))
    return crossed


def filter_adaptive_size(mask, fn_filter, filter_size, min_filter_size=1,
                         use_min_filter=False, min_size_ratio=None, name=""):
    """Filter morphologically with adaptive kernel sizes.
    
    Args:
        mask (:class:`numpy.ndarray`): Numpy array as a mask.
        fn_filter (func): Morphological filter function to apply.
        filter_size (int): Starting filter kernel size.
        min_filter_size (int): Minimum kernel size.
        use_min_filter (bool): True to use the filtered result if the kernel
            size is below ``min_filter_size`` even if filter criteria are
            not met; defaults to False.
        min_size_ratio (float): Minimum size ratio of filtered to original mask;
            defaults to None to use a ratio of 0.2.
        name (str): Name to prepend to output message; defaults to "".

    Returns:
        :class:`numpy.ndarray`, int: Filtered ``mask`` and kernel size at which
        ``mask`` was filtered, or ``np.nan`` if not filtered.

    """
    if min_size_ratio is None:
        min_size_ratio = 0.2
    region_size = np.sum(mask)
    fn_selem = get_selem(mask.ndim)
    
    # filter the label, starting with the given filter size and decreasing
    # if the resulting label size falls below a given min size ratio
    chosen_selem_size = np.nan
    filtered = mask
    for selem_size in range(filter_size, -1, -1):
        if selem_size < min_filter_size:
            if not use_min_filter:
                filtered = mask
                chosen_selem_size = np.nan
            break
        # erode check size ratio
        filtered = fn_filter(mask, fn_selem(selem_size))
        region_size_filtered = np.sum(filtered)
        size_ratio = region_size_filtered / region_size
        chosen_selem_size = selem_size
        if region_size_filtered != region_size and size_ratio > min_size_ratio:
            # stop filtering if filter made some change but stayed above
            # threshold size; skimage erosion treats border outside image
            # as True, so images filled by foreground are not eroded and size
            # should not be selected until lowest filter size is taken (eg NaN)
            break
    region_size_filt = np.sum(filtered)
    print(f"{name}: changed pixels from {region_size} to {region_size_filt} "
          f"(size ratio {region_size_filt / region_size}), initial filter size "
          f"of {filter_size}, chosen size of {chosen_selem_size}")
    return filtered, chosen_selem_size


def calc_isotropic_factor(
        scale: Union[float, Sequence[float]] = 1,
        res: Optional[Sequence[float]] = None) -> np.ndarray:
    """Calculate the isotropic factor based on the current image resolutions.
    
    The resolutions are divided by their minimum value and multiplied by
    the given scaling factor. The resulting value can be used to rescale
    images or coordinates to be isotropic.
    
    Args:
        scale: Float scalar or sequence of scaling factors in ``z, y, x`` by
            which to multiply the currently loaded image's resolutions.
            Defaults to 1.
        res: Resolutions in the same order as for ``scale``. Default to None,
            in which case :attr:`magmap.settings.config.resolutions` will be
            used instead.

    Returns:
        Isotropic factor.

    """
    if res is None:
        res = config.resolutions[0]
    resize_factor = np.divide(res, np.amin(res))
    resize_factor *= scale
    #print("isotropic resize factor: {}".format(resize_factor))
    return resize_factor


def make_isotropic(
        roi: np.ndarray, scale: Union[float, Sequence[float]] = 1,
        res: Optional[Sequence[float]] = None, **kwargs
) -> np.ndarray:
    """Make an array isotropic.
    
    Args:
        roi: Region of interest array in ``z, y, x`` format.
        scale: Float scalar or sequence of scaling factors in ``z, y, x`` by
            which to multiply the currently loaded image's resolutions.
            Defaults to 1.
        res: Resolutions in the same order as for ``scale``. Default to None,
            in which case :attr:`magmap.settings.config.resolutions` will be
            used instead.
        kwargs: Additional arguments to :meth:`rescale_resize`.

    Returns:
        Isotropic version of ``roi``.

    """
    resize_factor = calc_isotropic_factor(scale, res)
    isotropic_shape = np.array(roi.shape)
    isotropic_shape[:3] = (isotropic_shape[:3] * resize_factor).astype(int)
    libmag.printv("original ROI shape: {}, isotropic: {}"
                  .format(roi.shape, isotropic_shape))
    
    mode = "reflect"
    if np.any(np.array(roi.shape) == 1):
        # may crash with floating point exception if 1px thick (see
        # https://github.com/scikit-image/scikit-image/issues/3001, which
        # causes multiprocessing Pool to hang since the exception isn't
        # raised), so need to change mode in this case
        mode = "edge"
    
    # additional args override defaults
    args = dict(preserve_range=True, mode=mode)
    args.update(kwargs)
    return rescale_resize(roi, isotropic_shape, **args)


def rescale_resize(
        roi: np.ndarray,
        target_size: Optional[Union[float, Sequence[int]]] = None,
        multichannel: bool = False, preserve_range: bool = False, **kwargs
) -> np.ndarray:
    """Rescale or resize an array.
    
    Args:
        roi: Array to rescale or resize, in ``z, y, x, [c]` order.
        target_size: Target rescaling size for the given sub-ROI in
           ``z, y, x``, or a single number by which to rescale.
        multichannel: True if the final dimension is for channels; defaults
            to False.
        preserve_range: True to preserve the range of ``roi``; defaults to
            False.
        kwargs: Additional arguments passed to :meth:`transform.rescale`
            or :meth:`transform.resize`, such as ``order`` for label images.

    Returns:
        Rescaled array.

    """
    dtype = roi.dtype
    args = {
        "image": roi,
        "mode": "reflect",
        "preserve_range": preserve_range,
    }
    args.update(kwargs)
    
    if "order" in args and args["order"] == 0 and "anti_aliasing" not in args:
        # default to turn off anti-aliasing when order is 0 to preserve the
        # exact original values
        args["anti_aliasing"] = False
    
    if libmag.is_seq(target_size):
        # resize the image to a custom shape
        args["output_shape"] = target_size
        rescaled = transform.resize(**args)
    
    else:
        # rescale the image by a given factor
        args["scale"] = target_size
        if multichannel:
            # rescale multichannel image
            try:
                # Scikit-image >= v0.19
                rescaled = transform.rescale(**args, channel_axis=roi.ndim - 1)
            except TypeError:
                # Scikit-image < v0.19
                rescaled = transform.rescale(**args, multichannel=multichannel)
        else:
            # rescale single channel image
            rescaled = transform.rescale(**args)
    
    if preserve_range:
        rescaled = rescaled.astype(dtype)
    
    return rescaled


def angle_indices(
        shape: Sequence[int], offset: Sequence[int], size: Sequence[int],
        nsteps: Optional[int] = None):
    """Generate indices for an angled plane or other shape.
    
    Can be used to construct a polygon mask with angled faces. Indices
    co-vary so that as ``offset_z`` -> ``offset_z + size_z``,
    ``offset_y`` -> ``offset_y + size_y``, etc.
    
    Args:
        shape: Shape of object containing the desired plane.
        offset: Offset within the object in the orderr ``z, y, x``.
        size: Size within the object corresponding to elements in ``offset``.
        nsteps: Number of steps to interpolate. Defaults to None, where
            the max of ``shape`` x 10 is used to reduce chance of gaps.

    Returns:
        Indices in the same order as for ``shape``.
    
    Examples:
        See :meth:`magmap.tests.test_cv_nd.test_angle_indices`.

    """
    if nsteps is None:
        # default to more steps than max dimension to minimize chance of gaps
        nsteps = max(shape) * 10
    
    inds = [np.s_[:]] * len(shape)
    for i, (off, siz) in enumerate(zip(offset, size)):
        # make indices for given dimension with consistent step count
        inds[i] = np.linspace(off, siz, nsteps, False).astype(int)
    
    return inds


def get_selem(ndim):
    """Get structuring element appropriate for the number of dimensions.
    
    Currently only supports disk or ball structuring elements.
    
    Args:
        ndim (int): Number of dimensions.

    Returns:
        :func: Structuring element function. A :func:`morphology.ball`
        is returned for 3 or more dimensions, otherwise a
        :func:`morphology.disk`.

    """
    return morphology.ball if ndim >= 3 else morphology.disk


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
