# 3D plots from stacks of imaging data
# Author: David Young, 2017, 2020
"""3D image stack interaction.

Prepare volumetric image stacks for plotting.
"""

import numpy as np
from skimage import draw
from skimage import exposure
from skimage import filters
from skimage import morphology
from skimage import restoration

from magmap.settings import config
from magmap.io import libmag
from magmap.cv import cv_nd
from magmap.cv import segmenter


def setup_channels(roi, channel, dim_channel):
    """Setup channels array for the given ROI dimensions.
    
    Args:
        roi (:obj:`np.ndarray`): Region of interest, which is either a 3D
            or 4D array in the formate ``[[z, y, x, (c)], ...]``.
        channel (List[int]): Channels to select, which can be None to indicate
            all channels.
        dim_channel (int): Index of the channel dimension.
    
    Returns:
        bool, List[int]: A boolean value where True indicates that
        the ROI is multichannel (ie 4D) and an array of the channel indices
        of ``roi`` to include, which is the same as ``channel`` for
        multichannel ROIs or only the first element if ``roi`` is single
        channel.
    """
    multichannel = roi.ndim > dim_channel
    channels = channel
    if multichannel:
        if channel is None:
            # None indicates all channels
            channels = range(roi.shape[dim_channel])
    else:
        # only use the first given channel if ROI is single channel
        channels = [0]
    return multichannel, channels


def saturate_roi(roi, clip_vmin=-1, clip_vmax=-1, max_thresh_factor=-1,
                 channel=None):
    """Saturates an image, clipping extreme values and stretching remaining
    values to fit the full range.
    
    Args:
        roi (:obj:`np.ndarray`): Region of interest.
        clip_vmin (float): Percent for lower clipping. Defaults to -1
            to use the profile setting.
        clip_vmax (float): Percent for upper clipping. Defaults to -1
            to use the profile setting.
        max_thresh_factor (float): Multiplier of :attr:`config.near_max`
            for ROI's scaled maximum value. If the max data range value
            adjusted through``clip_vmax``is below this product, this max
            value will be set to this product. Defaults to -1 to use the
            profile setting.
        channel (List[int]): Sequence of channel indices in ``roi`` to
            saturate. Defaults to None to use all channels.
    
    Returns:
        Saturated region of interest.
    """
    multichannel, channels = setup_channels(roi, channel, 3)
    roi_out = None
    for chl in channels:
        roi_show = roi[..., chl] if multichannel else roi
        settings = config.get_roi_profile(chl)
        if clip_vmin == -1:
            clip_vmin = settings["clip_vmin"]
        if clip_vmax == -1:
            clip_vmax = settings["clip_vmax"]
        if max_thresh_factor == -1:
            max_thresh_factor = settings["max_thresh_factor"]
        # enhance contrast and normalize to 0-1 scale
        vmin, vmax = np.percentile(roi_show, (clip_vmin, clip_vmax))
        libmag.printv(
            "vmin:", vmin, "vmax:", vmax, "near max:", config.near_max[chl])
        # adjust the near max value derived globally from image5d for the chl
        max_thresh = config.near_max[chl] * max_thresh_factor
        if vmax < max_thresh:
            vmax = max_thresh
            libmag.printv("adjusted vmax to {}".format(vmax))
        saturated = np.clip(roi_show, vmin, vmax)
        saturated = (saturated - vmin) / (vmax - vmin)
        if multichannel:
            if roi_out is None:
                roi_out = np.zeros(roi.shape, dtype=saturated.dtype)
            roi_out[..., chl] = saturated
        else:
            roi_out = saturated
    return roi_out


def denoise_roi(roi, channel=None):
    """Apply further saturation, denoising, unsharp filtering, and erosion
    as image preprocessing for blob detection.

    Each step can be configured including turned off by
    :attr:`config.process_settings`.
    
    Args:
        roi: Region of interest as a 3D (z, y, x) array. Note that 4D arrays 
            with channels are not allowed as the Scikit-Image gaussian filter 
            only accepts specifically 3 channels, presumably for RGB.
        channel (List[int]): Sequence of channel indices in ``roi`` to
            saturate. Defaults to None to use all channels.
    
    Returns:
        Denoised region of interest.
    """
    multichannel, channels = setup_channels(roi, channel, 3)
    roi_out = None
    for chl in channels:
        roi_show = roi[..., chl] if multichannel else roi
        settings = config.get_roi_profile(chl)
        # find gross density
        saturated_mean = np.mean(roi_show)
        
        # further saturation
        denoised = np.clip(roi_show, settings["clip_min"], settings["clip_max"])

        tot_var_denoise = settings["tot_var_denoise"]
        if tot_var_denoise:
            # total variation denoising
            denoised = restoration.denoise_tv_chambolle(
                denoised, weight=tot_var_denoise)
        
        # sharpening
        unsharp_strength = settings["unsharp_strength"]
        if unsharp_strength:
            blur_size = 8
            # turn off multichannel since assume operation on single channel at
            # a time and to avoid treating as multichannel if 3D ROI happens to
            # have x size of 3
            blurred = filters.gaussian(denoised, blur_size, multichannel=False)
            high_pass = denoised - unsharp_strength * blurred
            denoised = denoised + high_pass
        
        # further erode denser regions to decrease overlap among blobs
        thresh_eros = settings["erosion_threshold"]
        if thresh_eros and saturated_mean > thresh_eros:
            #print("denoising for saturated mean of {}".format(saturated_mean))
            denoised = morphology.erosion(denoised, morphology.octahedron(1))
        if multichannel:
            if roi_out is None:
                roi_out = np.zeros(roi.shape, dtype=denoised.dtype)
            roi_out[..., chl] = denoised
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
    settings = config.roi_profile
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
        roi_thresh = libmag.normalize(roi_thresh, -1.0, 1.0)
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


def remap_intensity(roi, channel=None):
    """Remap intensities, currently using adaptive histogram equalization
    but potentially plugging in alternative methods in the future.

    Args:
        roi (:obj:`np.ndarray`): Region of interest as a 3D or 3D+channel array.
        channel (int): Channel index of ``roi`` to saturate. Defaults to None
            to use all channels. If a specific channel is given, all other
            channels remain unchanged.

    Returns:
        :obj:`np.ndarray`: Remapped region of interest as a new array.

    """
    multichannel, channels = setup_channels(roi, channel, 3)
    roi_out = np.copy(roi)
    for chl in channels:
        roi_show = roi[..., chl] if multichannel else roi
        settings = config.get_roi_profile(chl)
        lim = settings["adapt_hist_lim"]
        print("Performing adaptive histogram equalization on channel {}, "
              "clip limit {}".format(chl, lim))
        equalized = []
        for plane in roi_show:
            # workaround for lack of current nD support in scikit-image CLAHE
            # implementation (but this PR looks promising:
            # https://github.com/scikit-image/scikit-image/pull/2761 )
            equalized.append(
                exposure.equalize_adapthist(plane, clip_limit=lim))
        equalized = np.stack(equalized)
        if multichannel:
            roi_out[..., chl] = equalized
        else:
            roi_out = equalized
    return roi_out


def get_isotropic_vis(settings):
    """Get the isotropic factor scaled by the profile setting for visualization.
    
    Visualization may require the isotropic factor to be further scaled,
    such as for images whose z resolution is less precise that x/y resolution.
    
    Args:
        settings (:class:`magmap.settings.profiles.SettingsDict`): Settings
            dictionary from which to retrieve the isotropic visualization value.

    Returns:
        :class:`numpy.ndarray`: Isotropic factor based on the profile setting.

    """
    isotropic = settings["isotropic_vis"]
    if isotropic is not None:
        isotropic = cv_nd.calc_isotropic_factor(isotropic)
    return isotropic


def prepare_subimg(image5d, offset, size, ndim_base=5):
    """Extracts a subimage from a larger image.
    
    Args:
        image5d (:obj:`np.ndarray`): 5D image array in the order,
            ``t,z,y,x[,c]``, where the final dimension is optional as with
            many one channel images.
        offset (List[int]): Tuple of offset given as ``z,y,x`` for the region
            of interest. Defaults to ``(0, 0, 0)``.
        size (List[int]): Size of the region of interest as ``z,y,x``.
        ndim_base (int): Number of dimensions on which ``image5d`` is based.
            Typically 3 or 5, defaulting to 5 as ``t,z,y,x[,c]``.
            If 3, the ``t`` dimension is removed.
    
    Returns:
        :obj:`np.ndarray`: The sub-imge without separate time dimension as
        a 3D (or 4-D array if channel dimension exists) array.
    
    """
    cube_slices = [slice(o, o + s) for o, s in zip(offset, size)]
    libmag.printv("preparing sub-image at offset: {}, size: {}, slices: {}"
                  .format(offset, size, cube_slices))
    
    # cube with corner at offset and shape given by size
    img = image5d
    if ndim_base >= 5:
        # remove time axis
        img = image5d[0]
    return img[cube_slices[0], cube_slices[1], cube_slices[2]]


def prepare_roi(image5d, roi_offset, roi_size, ndim_base=5):
    """Extracts a region of interest (ROI).

    Calls :meth:`prepare_subimage` but expects size and offset variables to
    be in x,y,z order following this software's legacy convention.

    Args:
        image5d (:obj:`np.ndarray`): 5D image array in the order,
            ``t,z,y,x[,c]``, where the final dimension is optional as with
            many one channel images.
        roi_offset (List[int]): Tuple of offset given as ``x,y,z`` for the region
            of interest. Defaults to ``(0, 0, 0)``.
        roi_size (List[int]): Size of the region of interest as ``x,y,z``.
        ndim_base (int): Number of dimensions on which ``image5d`` is based.
            Typically 3 or 5, defaulting to 5 as ``t,z,y,x[,c]``.
            If 3, the ``t`` dimension is removed.

    Returns:
        :obj:`np.ndarray`: The region of interest without separate time
        dimension as a 3D (or 4-D array if channel dimension exists) array.
    
    """
    libmag.printv("preparing ROI at x,y,z:")
    return prepare_subimg(image5d, roi_offset[::-1], roi_size[::-1], ndim_base)


def roi_center_to_offset(offset, shape, reverse=False):
    """Convert an ROI offset given as the center of the ROI to the
    coordinates of the upper left hand corner of the ROI.
    
    Args:
        offset (list[int]): Offset taken as the center of the ROI in any
            order, typically either ``x,y,z`` or ``z,y,x``.
        shape (list[int]): ROI shape in the same order as that of ``offset``.
        reverse (bool): True to treat ``offset`` as the upper left hand
            corner of the ROI and to obtain the center coordinates of
            this ROI; defaults to False.

    Returns:
        list[int]: Coordinates of the upper left corner of the ROI, or the
        center of the ROI if ``reverse`` is True.

    """
    fn = np.add if reverse else np.subtract
    return fn(offset, np.floor_divide(shape, 2))


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


def replace_vol(img, vol, center=None, offset=None, vol_as_mask=None):
    """Replace a volume within an image, centering on the given coordinates 
    and cropping the input volume to fit.

    Args:
        img (:class:`numpy.ndarray`): Image as a Numpy array into which
            ``vol`` will be placed. Updated in-place.
        vol (:class:`numpy.ndarray`): Volume to place in ``img``.
        center (tuple[int, int, int]): Coordinates of the center of volume,
            given as a sequence of ``z,y,x``. Either ``center`` or ``offset``
            must be given. Takes precedence over ``offset``.
        offset (tuple[int, int, int]): Coordinates of offset within ``img``
            to place ``vol``, given as a sequence of ``z,y,x``.
        vol_as_mask (:class:`numpy.ndarray`): If ``vol`` should be taken as
            a mask, where only its True values will replace the corresponding
            pixels in ``img``, assign this value to the mask locations.
            Defaults to None, in which case the entire ``vol`` will be assigned.

    Returns:
        :class:`numpy.ndarray`: ``img`` modified in-place.
    
    Raises:
        ValueError: if ``center`` and ``offset`` are both None.
    
    """
    if center is None and offset is None:
        raise ValueError("`center` or `offset` must be given in `replace_vol`")
    dims = vol.ndim
    slices_img = []
    slices_vol = []
    for i in range(dims):
        start_vol = 0
        stop_vol = int(vol.shape[i])
        if center is not None:
            # center volumes with odd-numbered length, and skew slightly
            # toward lower values for even-numbered length
            start = int(center[i] - vol.shape[i] // 2)
        else:
            start = offset[i]
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


def pad_img(img, offset, shape):
    """Pad image surroundings with zeros.
    
    Args:
        img (:class:`numpy.ndarray`): Image array.
        offset (tuple[int, int, int]): Offset within padded image at which
            to place ``img``, given as a sequence of ``z,y,x``.
        shape (tuple[int, int, int]): Shape of resulting image, given as
            a sequence of ``z,y,x``. Values can be None or sequence can stop
            early to use the corresponding original shape values from ``img``.

    Returns:
        :class:`numpy.ndarray`: Padded image.

    """
    shape_padded = list(img.shape)
    for axis, n in enumerate(shape):
        if shape[axis] is not None:
            shape_padded[axis] = shape[axis]
    img_padded = np.zeros(shape_padded, dtype=img.dtype)
    return replace_vol(img_padded, img, offset=offset)


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
