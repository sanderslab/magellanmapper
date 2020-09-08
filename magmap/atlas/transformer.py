# Transform images with multiprocessing
# Author: David Young, 2019, 2020
"""Transform large images with multiprocessing, including up/downsampling 
and image transposition.
"""

from time import time

import numpy as np
from skimage import transform

from magmap.cv import chunking, cv_nd
from magmap.settings import config
from magmap.settings import profiles
from magmap.io import importer
from magmap.io import libmag
from magmap.plot import plot_3d


class Downsampler(object):
    """Downsample (or theoretically upsample) a large image in a way 
    that allows multiprocessing without global variables.
    
    Attributes:
        img (:obj:`np.ndarray`): Full image array.
    """
    img = None
    
    @classmethod
    def set_data(cls, img):
        """Set the class attributes to be shared during multiprocessing.

        Args:
            img (:obj:`np.ndarray`): See attributes.

        """
        cls.img = img
    
    @classmethod
    def rescale_sub_roi(cls, coord, slices, rescale, target_size, multichannel,
                        sub_roi=None):
        """Rescale or resize a sub-ROI.
        
        Args:
            coord: Coordinates as a tuple of (z, y, x) of the sub-ROI within the 
                chunked ROI.
            slices (Tuple[slice]): Sequence of slices within
                :attr:``img`` defining the sub-ROI.
            rescale: Rescaling factor. Can be None, in which case 
                ``target_size`` will be used instead.
            target_size: Target rescaling size for the given sub-ROI in 
               (z, y, x). If ``rescale`` is not None, ``target_size`` 
               will be ignored.
            multichannel: True if the final dimension is for channels.
            sub_roi (:obj:`np.ndarray`): Array chunk to rescale/resize;
                defaults to None to extract from :attr:`img` if available.
        
        Return:
            Tuple of ``coord`` and the rescaled sub-ROI, where 
            ``coord`` is the same as the given parameter to identify 
            where the sub-ROI is located during multiprocessing tasks.
        """
        if sub_roi is None and cls.img is not None:
            sub_roi = cls.img[slices]
        rescaled = None
        if rescale is not None:
            rescaled = transform.rescale(
                sub_roi, rescale, mode="reflect", multichannel=multichannel)
        elif target_size is not None:
            rescaled = transform.resize(
                sub_roi, target_size, mode="reflect", anti_aliasing=True)
        return coord, rescaled


def make_modifier_plane(plane):
    """Make a string designating a plane orthogonal transformation.
    
    Args:
        plane: Plane to which the image was transposed.
    
    Returns:
        String designating the orthogonal plane transformation.
    """
    return "plane{}".format(plane.upper())


def make_modifier_scale(scale):
    """Make a string designating a scaling transformation, typically for
    filenames of rescaled images.
    
    Args:
        scale (float): Scale to which the image was rescaled. Any decimal
            point will be replaced with "pt" to avoid confusion with
            path extensions.
    
    Returns:
        str: String designating the scaling transformation.
    """
    mod = "scale{}".format(scale)
    return mod.replace(".", "pt")


def make_modifier_resized(target_size):
    """Make a string designating a resize transformation.
    
    Note that the final image size may differ slightly from this size as 
    it only reflects the size targeted.
    
    Args:
        target_size: Target size of rescaling in x,y,z.
    
    Returns:
        String designating the resize transformation.
    """
    return "resized({},{},{})".format(*target_size)


def get_transposed_image_path(img_path, scale=None, target_size=None):
    """Get path, modified for any transposition by :func:``transpose_npy`` 
    naming conventions.
    
    Args:
        img_path: Unmodified image path.
        scale: Scaling factor; defaults to None, which ignores scaling.
        target_size: Target size, typically given by a register profile; 
            defaults to None, which ignores target size.
    
    Returns:
        Modified path for the given transposition, or ``img_path`` unmodified 
        if all transposition factors are None.
    """
    img_path_modified = img_path
    if scale is not None or target_size is not None:
        # use scaled image for pixel comparison, retrieving 
        # saved scaling as of v.0.6.0
        modifier = None
        if scale is not None:
            # scale takes priority as command-line argument
            modifier = make_modifier_scale(scale)
            print("loading scaled file with {} modifier".format(modifier))
        else:
            # otherwise assume set target size
            modifier = make_modifier_resized(target_size)
            print("loading resized file with {} modifier".format(modifier))
        img_path_modified = libmag.insert_before_ext(
            img_path, "_" + modifier)
    return img_path_modified


def transpose_img(filename, series, plane=None, rescale=None, target_size=None):
    """Transpose Numpy NPY saved arrays into new planar orientations and 
    rescaling or resizing.
    
    Rescaling/resizing take place in multiprocessing. Files are saved
    through memmap-based arrays to minimize RAM usage. Output filenames
    are based on the ``make_modifer_[task]`` functions. Currently transposes
    all channels, ignoring :attr:``config.channel`` parameter.
    
    Args:
        filename: Full file path in :attribute:cli:`filename` format.
        series: Series within multi-series file.
        plane: Planar orientation (see :attribute:plot_2d:`PLANES`). Defaults 
            to None, in which case no planar transformation will occur.
        rescale: Rescaling factor; defaults to None. Takes precedence over
            ``target_size``.
        target_size (List[int]): Target shape in x,y,z; defaults to None,
            in which case the target size will be extracted from the register
            profile if available if available.

    """
    if target_size is None:
        target_size = config.atlas_profile["target_size"]
    if plane is None and rescale is None and target_size is None:
        print("No transposition to perform, skipping")
        return
    
    time_start = time()
    # even if loaded already, reread to get image metadata
    # TODO: consider saving metadata in config and retrieving from there
    img5d, info = importer.read_file(filename, series, return_info=True)
    image5d = img5d.img
    sizes = info["sizes"]
    
    # make filenames based on transpositions
    modifier = ""
    if plane is not None:
        modifier = make_modifier_plane(plane)
    # either rescaling or resizing
    if rescale is not None:
        modifier += make_modifier_scale(rescale)
    elif target_size:
        # target size may differ from final output size but allows a known 
        # size to be used for finding the file later
        modifier += make_modifier_resized(target_size)
    filename_image5d_npz, filename_info_npz = importer.make_filenames(
        filename, series, modifier=modifier)
    
    # TODO: image5d should assume 4/5 dimensions
    offset = 0 if image5d.ndim <= 3 else 1
    multichannel = image5d.ndim >= 5
    image5d_swapped = image5d
    
    if plane is not None and plane != config.PLANE[0]:
        # swap z-y to get (y, z, x) order for xz orientation
        image5d_swapped = np.swapaxes(image5d_swapped, offset, offset + 1)
        config.resolutions[0] = libmag.swap_elements(
            config.resolutions[0], 0, 1)
        if plane == config.PLANE[2]:
            # swap new y-x to get (x, z, y) order for yz orientation
            image5d_swapped = np.swapaxes(image5d_swapped, offset, offset + 2)
            config.resolutions[0] = libmag.swap_elements(
                config.resolutions[0], 0, 2)
    
    scaling = None
    if rescale is not None or target_size is not None:
        # rescale based on scaling factor or target specific size
        rescaled = image5d_swapped
        # TODO: generalize for more than 1 preceding dimension?
        if offset > 0:
            rescaled = rescaled[0]
        max_pixels = [100, 500, 500]
        sub_roi_size = None
        if target_size:
            # to avoid artifacts from thin chunks, fit image into even
            # number of pixels per chunk by rounding up number of chunks
            # and resizing each chunk by ratio of total size to chunk num
            target_size = target_size[::-1]  # change to z,y,x
            shape = rescaled.shape[:3]
            num_chunks = np.ceil(np.divide(shape, max_pixels))
            max_pixels = np.ceil(
                np.divide(shape, num_chunks)).astype(np.int)
            sub_roi_size = np.floor(
                np.divide(target_size, num_chunks)).astype(np.int)
            print("Resizing image of shape {} to target_size: {}, using "
                  "num_chunks: {}, max_pixels: {}, sub_roi_size: {}"
                  .format(rescaled.shape, target_size, num_chunks, max_pixels,
                          sub_roi_size))
        else:
            print("Rescaling image of shape {} by factor of {}"
                  .format(rescaled.shape, rescale))
        
        # rescale in chunks with multiprocessing
        sub_roi_slices, _ = chunking.stack_splitter(rescaled.shape, max_pixels)
        is_fork = chunking.is_fork()
        if is_fork:
            Downsampler.set_data(rescaled)
        sub_rois = np.zeros_like(sub_roi_slices)
        pool = chunking.get_mp_pool()
        pool_results = []
        for z in range(sub_roi_slices.shape[0]):
            for y in range(sub_roi_slices.shape[1]):
                for x in range(sub_roi_slices.shape[2]):
                    coord = (z, y, x)
                    slices = sub_roi_slices[coord]
                    args = [coord, slices, rescale, sub_roi_size,
                            multichannel]
                    if not is_fork:
                        # pickle chunk if img not directly available
                        args.append(rescaled[slices])
                    pool_results.append(pool.apply_async(
                        Downsampler.rescale_sub_roi, args=args))
        for result in pool_results:
            coord, sub_roi = result.get()
            print("replacing sub_roi at {} of {}"
                  .format(coord, np.add(sub_roi_slices.shape, -1)))
            sub_rois[coord] = sub_roi
        
        pool.close()
        pool.join()
        rescaled_shape = chunking.get_split_stack_total_shape(sub_rois)
        if offset > 0:
            rescaled_shape = np.concatenate(([1], rescaled_shape))
        print("rescaled_shape: {}".format(rescaled_shape))
        # rescale chunks directly into memmap-backed array to minimize RAM usage
        image5d_transposed = np.lib.format.open_memmap(
            filename_image5d_npz, mode="w+", dtype=sub_rois[0, 0, 0].dtype,
            shape=tuple(rescaled_shape))
        chunking.merge_split_stack2(sub_rois, None, offset, image5d_transposed)
        
        if rescale is not None:
            # scale resolutions based on single rescaling factor
            config.resolutions = np.multiply(
                config.resolutions, 1 / rescale)
        else:
            # scale resolutions based on size ratio for each dimension
            config.resolutions = np.multiply(
                config.resolutions, 
                (image5d_swapped.shape / rescaled_shape)[1:4])
        sizes[0] = rescaled_shape
        scaling = importer.calc_scaling(image5d_swapped, image5d_transposed)
    else:
        # transfer directly to memmap-backed array
        image5d_transposed = np.lib.format.open_memmap(
            filename_image5d_npz, mode="w+", dtype=image5d_swapped.dtype, 
            shape=image5d_swapped.shape)
        if plane == config.PLANE[1] or plane == config.PLANE[2]:
            # flip upside-down if re-orienting planes
            if offset:
                image5d_transposed[0, :] = np.fliplr(image5d_swapped[0, :])
            else:
                image5d_transposed[:] = np.fliplr(image5d_swapped[:])
        else:
            image5d_transposed[:] = image5d_swapped[:]
        sizes[0] = image5d_swapped.shape
    
    # save image metadata
    print("detector.resolutions: {}".format(config.resolutions))
    print("sizes: {}".format(sizes))
    image5d.flush()
    importer.save_image_info(
        filename_info_npz, info["names"], sizes, config.resolutions, 
        info["magnification"], info["zoom"], 
        *importer.calc_intensity_bounds(image5d_transposed), scaling, plane)
    print("saved transposed file to {} with shape {}".format(
        filename_image5d_npz, image5d_transposed.shape))
    print("time elapsed (s): {}".format(time() - time_start))


def rotate_img(roi, rotate=None, order=None):
    """Rotate an ROI based on atlas profile settings.

    Args:
        roi (:obj:`np.ndarray`): Region of interst array (z,y,x[,c]).
        rotate (dict): Dictionary of rotation settings in
            :class:`magmap.settings.atlas_profile`. Defaults to None
            to take the value from :attr:`config.register_settings`.
        order (int): Spline interpolation order; defalts to None to use
            the value from within ``rotate``. Should be 0 for labels.

    Returns:
        :obj:`np.ndarray`: The rotated image array.

    """
    if rotate is None:
        rotate = config.atlas_profile["rotate"]
    if order is None:
        order = rotate["order"]
    roi = np.copy(roi)
    for rot in rotate["rotation"]:
        print("rotating by", rot)
        roi = cv_nd.rotate_nd(
            roi, rot[0], rot[1], order=order, resize=rotate["resize"])
    return roi


def preprocess_img(image5d, preprocs, channel, out_path):
    """Pre-process an image in 3D.

    Args:
        image5d (:obj:`np.ndarray`): 5D array in t,z,y,x[,c].
        preprocs (List[:obj:`profiles.PreProcessKeys`): Sequence of
            pre-processing tasks to perform in the order given.
        channel (int): Channel to preprocess, or None for all channels.
        out_path (str): Output base path.

    Returns:
        :obj:`np.ndarray`: The pre-processed image array.

    """
    if preprocs is None:
        print("No preprocessing tasks to perform, skipping")
        return

    roi = image5d[0]
    for preproc in preprocs:
        # perform global pre-processing task
        print("Pre-processing task:", preproc)
        if preproc is profiles.PreProcessKeys.SATURATE:
            roi = plot_3d.saturate_roi(roi, channel=channel)
        elif preproc is profiles.PreProcessKeys.DENOISE:
            roi = plot_3d.denoise_roi(roi, channel)
        elif preproc is profiles.PreProcessKeys.REMAP:
            roi = plot_3d.remap_intensity(roi, channel)
        elif preproc is profiles.PreProcessKeys.ROTATE:
            roi = rotate_img(roi)

    # save to new file
    image5d = importer.roi_to_image5d(roi)
    importer.save_np_image(image5d, out_path)
    return image5d
