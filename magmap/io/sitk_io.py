# Input/Output for SimpleITK/SimpleElastix objects
# Author: David Young, 2019
"""Input/Output for SimpleITK/SimpleElastix objects.

Manage import and export of :class:`simpleitk.Image` objects.
"""
import os
import shutil
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import itk
except ImportError:
    itk = None
import numpy as np
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
from skimage import transform

from magmap.settings import config
from magmap.io import importer, np_io
from magmap.io import libmag

#: Extensions of 3D formats supported through SimpleITK.
EXTS_3D: Sequence[str] = (".mhd", ".mha", ".nii.gz", ".nii", ".nhdr", ".nrrd")
# TODO: include all formats supported by SimpleITK

_logger = config.logger.getChild(__name__)


def reg_out_path(file_path, reg_name, match_ext=False):
    """Generate a path for a file registered to another file.
    
    Args:
        file_path: Full path of file registered to. :attr:`config.series`
            will be appended unless ``file_path`` is a directory.
        reg_name: Suffix for type of registration, eg :const:``IMG_LABELS``.
        match_ext: True to change the extension of ``reg_name`` to match 
            that of ``file_path``.
    
    Returns:
        str: Full path with the registered filename including appropriate
        extension at the end.
    """
    if match_ext:
        reg_name = libmag.match_ext(file_path, reg_name)
    if os.path.isdir(file_path):
        return os.path.join(file_path, reg_name)
    else:
        file_path_base = importer.filename_to_base(
            file_path, config.series)
        return file_path_base + "_" + reg_name


def convert_img(
        img: Union["sitk.Image", "itk.Image", np.ndarray],
        multichannel: Optional[bool] = None, to_sitk: Optional[bool] = None
) -> Union["sitk.Image", "itk.Image", np.ndarray]:
    """Convert SimpleITK, ITK, and NumPy images.
    
    Args:
        img: SimpleITK Image, ITK Image, or NumPy array. Non-NumPy arrays
            will be converted to a NumPy array.
        multichannel: True if the resulting SimpleITK or ITK Image should be
            multichannel. Defaults to None, which will attempt to auto-detect
            multichannel images. Only used if ``img`` is a NumPy array.
        to_sitk: True to convert a NumPy ``img`` to a SimpleITK Image. If
            False, an ITK Image is output instead. If None, the parameter will
            be True if the SimpleITK library is found. Ignored if ``img`` is
            not a NumPy array.

    Returns:
        The converted image.

    """
    if to_sitk is None:
        # default to convert to SimpleITK if the library is present
        to_sitk = sitk is not None
    
    conv = img
    if sitk and isinstance(img, sitk.Image):
        # convert an sitk Image to an np array
        conv = sitk.GetArrayFromImage(img)
    elif itk and isinstance(img, (itk.Image, itk.VectorImage)):
        # convert an ITK image to an np array
        conv = itk.GetArrayFromImage(img)
    elif sitk and to_sitk:
        # convert an np array to sitk Image
        conv = sitk.GetImageFromArray(img, multichannel)
    elif itk:
        # convert an np array to ITK Image
        if multichannel is None:
            # default to treat 4D+ images as multichannel
            multichannel = img.ndim > 3
        conv = itk.GetImageFromArray(img, multichannel)
    return conv


def replace_sitk_with_numpy(
        img_sitk: Union["sitk.Image", "itk.Image"], img_np: np.ndarray,
        multichannel: bool = None
) -> Union["sitk.Image", "itk.Image"]:
    """Generate a :class:``sitk.Image`` object based on another Image object,
    but replace its array with a new Numpy array.
    
    Args:
        img_sitk: Image object to use as template.
        img_np: Numpy array to swap in.
        multichannel: True for multichannel images. Defaults to None, where
            the multichannel setting from ``img_sitk`` will be used.
    
    Returns:
        New SimpleITK or ITK ``sitk.Image`` object with same spacing, origin,
        and direction as that of ``img_sitk`` and array replaced by ``img_np``.
    
    """
    # get original settings
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    direction = img_sitk.GetDirection()
    
    # treat as vector (multichannel) image if source is multichannel
    if multichannel is None:
        multichannel = (
            True if img_sitk.GetNumberOfComponentsPerPixel() > 1 else None)
    img_sitk_back = convert_img(
        img_np, multichannel, sitk and isinstance(img_sitk, sitk.Image))
    
    # transfer original settings to new Image, matching length for ITK
    spacing = libmag.pad_seq(tuple(spacing), len(img_sitk_back.GetSpacing()), 1)
    origin = libmag.pad_seq(tuple(origin), len(img_sitk_back.GetOrigin()), 1)
    img_sitk_back.SetSpacing(tuple(spacing))
    img_sitk_back.SetOrigin(tuple(origin))
    
    # convert directions to 2D arrays if necessary
    dir_np = np.array(direction)
    dir_back = np.array(img_sitk_back.GetDirection())
    is_dir_1d = dir_np.ndim == 1
    if is_dir_1d:
        dir_np = np.reshape(np.array(dir_np), [len(img_sitk.GetSpacing())] * 2)
        dir_back = np.reshape(
            np.array(dir_back), [len(img_sitk_back.GetSpacing())] * 2)
    
    # fill target direction with source, truncating it if necessary for ITK
    shape = np.minimum(dir_np.shape, dir_back.shape)
    dir_back[:shape[0], :shape[1]] = dir_np[:shape[0], :shape[1]]
    if is_dir_1d:
        dir_back = np.ravel(dir_back)
    
    try:
        img_sitk_back.SetDirection(dir_back)
    except RuntimeError:
        # direction format and length may not be directly transferable, such
        # as direction from groupwise reg image
        _logger.warn(
            "Could not replace image direction with: %s\n"
            "Leaving default direction: %s",
            direction, img_sitk_back.GetDirection())
    
    return img_sitk_back


def match_world_info(
        source: Union["sitk.Image", "itk.Image"],
        target: Union["sitk.Image", "itk.Image"],
        spacing: Union[bool, Tuple[int], List[int]] = True,
        origin: Union[bool, Tuple[int], List[int]] = True,
        direction: Union[bool, Tuple[int], List[int]] = True):
    """Transfer world information from one image object to another.

    Supported world information is spacing, origin, direction. Matching
    this information is sometimes necessary for slight differences in
    metadata perhaps from founding that may prevent ITK filters from executing.

    Args:
        source: Source object whose relevant metadata
            will be copied into ``target``.
        target: Target object whose corresponding
            metadata will be overwritten by that of ``source``.
        spacing: True to copy the spacing from ``source`` to ``target``, or
            the spacing to set in ``target``; defaults to True.
        origin: True to copy the origin from ``source`` to ``target``, or
            the origin to set in ``target``; defaults to True.
        direction: True to copy the direction from ``source`` to ``target``, or
            the direction to set in ``target``; defaults to True.

    """
    # get the world info from the source if not already set
    if spacing is True:
        spacing = source.GetSpacing()
    if origin is True:
        origin = source.GetOrigin()
    if direction is True:
        direction = source.GetDirection()
    
    # set the values in the target
    _logger.debug(
        "Adjusting spacing from %s to %s, origin from %s to %s, "
        "direction from %s to %s", target.GetSpacing(), spacing,
        target.GetOrigin(), origin, target.GetDirection(), direction)
    if spacing:
        target.SetSpacing(spacing)
    if origin:
        target.SetOrigin(origin)
    if direction:
        target.SetDirection(direction)


def read_sitk(
        path: str, dryrun: bool = False
) -> Tuple[Optional["sitk.Image"], Optional[str]]:
    """Read an image file into SimpleITK ``sitk.Image`` format
    
    Checks for alternative supported extensions if necessary.
    
    Args:
        path: Path, including prioritized extension to check first.
        dryrun: True to find an existing path if available, without
            loading the image; defaults to False.
    
    Returns:
        Tuple of:
        - ``img_sitk``: Image object located at ``path`` with the found
          extension, or None if unable to load
        - ``path_loaded``: the loaded path, or None if no matching, existing
          path is found. If a file at ``path`` cannot be found, its extension
          is replaced successively with remaining extensions in
          :const:``EXTS_3D`` until a file is found.
    
    """
    # prioritize given extension
    path_split = libmag.splitext(path)
    exts = list(EXTS_3D)
    if path_split[1] in exts: exts.remove(path_split[1])
    exts.insert(0, path_split[1])
    
    # attempt to load using each extension until found
    img_sitk = None
    path_loaded = None
    for ext in exts:
        img_path = path_split[0] + ext
        if os.path.exists(img_path):
            if not dryrun:
                _logger.debug("Loading image with SimpleITK: %s", img_path)
                img_sitk = sitk.ReadImage(img_path)
            path_loaded = img_path
            break
    if not dryrun and img_sitk is None:
        _logger.warn(
            "could not find image from %s and extensions %s", path_split[0],
            exts)
    return img_sitk, path_loaded


def _load_reg_img_to_combine(path, reg_name, img_nps):
    # load registered image in sitk format to combine with other images
    # by resizing to the shape of the first image
    img_np_base = None
    if img_nps:
        # use first image in list as basis for shape
        img_np_base = img_nps[0]
    img_sitk, loaded_path = load_registered_img(
        path, reg_name, get_sitk=True, return_path=True)
    img_np = sitk.GetArrayFromImage(img_sitk)
    if img_np_base is not None:
        if img_np_base.shape != img_np.shape:
            # resize to first image
            img_np = transform.resize(
                img_np, img_np_base.shape, preserve_range=True, 
                anti_aliasing=True, mode="reflect")
        # normalize to max of first image to make comparable when combining
        img_np = libmag.normalize(
            img_np * 1.0, 0, np.amax(img_np_base))
    img_nps.append(img_np)
    return img_sitk, loaded_path


def _make_3d(img_sitk: "sitk.Image", spacing_z=1) -> "sitk.Image":
    """Make a 2D image into a 3D image with a single plane.
    
    Args:
        img_sitk: Image in SimpleITK format.
        spacing_z: Z-axis spacing; defaults to 1.

    Returns:
        ``img_sitk`` converted to a 3D image if previously 2D.

    """
    spacing = img_sitk.GetSpacing()
    if len(spacing) == 2:
        # prepend an additional axis for 2D images to make them 3D
        img_np = sitk.GetArrayFromImage(img_sitk)[None]
        spacing = list(spacing) + [spacing_z]
        img_sitk = sitk.GetImageFromArray(img_np)
        img_sitk.SetSpacing(spacing)
    return img_sitk


def read_sitk_files(
        filename_sitk: str,
        reg_names: Optional[Union[str, Sequence[str]]] = None,
        return_sitk: bool = False, make_3d: bool = False
) -> Union["np_io.Image5d", Tuple["np_io.Image5d", "sitk.Image"]]:
    """Read image files through SimpleITK.
    
    Supports identifying files based on registered suffixes and combining
    multiple registered image files into a single image.
    
    Also sets up spacing from the first loaded image in
    :attr:`magmap.settings.config.resolutions` if not already set.

    Args:
        filename_sitk: Path to file in a format that can be read by
            SimpleITK.
        reg_names: Path or sequence of paths of
            registered names. Can be a registered suffix or a full path.
            Defaults to None to open ``filename_sitk`` as-is through
            :meth:`read_sitk`.
        return_sitk: True to return the loaded SimpleITK Image object.
        make_3d: True to convert 2D images to 3D; defaults to False.

    Returns:
        ``img5d``: Image5d instance with the loaded image in Numpy 5D format
        (or 4D if not multi-channel, and 3D if originally 2D and ``make_3d``
        is False). Associated metadata is loaded into
        :module:`magmap.settings.config` attributes.
        
        If ``return_sitk`` is True, also returns the first loaded image
        in SimpleITK format.

    Raises:
        ``FileNotFoundError`` if ``filename_sitk`` cannot be found, after 
        attempting to load metadata from ``filename_np``.
    
    """
    # load image via SimpleITK
    img_sitk = None
    loaded_path = filename_sitk
    if reg_names:
        img_nps = []
        if not libmag.is_seq(reg_names):
            reg_names = [reg_names]
        for reg_name in reg_names:
            # load each registered suffix into list of images with same shape,
            # keeping first image in sitk format
            img, path = _load_reg_img_to_combine(
                filename_sitk, reg_name, img_nps)
            if img_sitk is None:
                img_sitk = img
                loaded_path = path
        
        if len(img_nps) > 1:
            # merge images into separate channels
            img_np = np.stack(img_nps, axis=img_nps[0].ndim)
        else:
            img_np = img_nps[0]
        
        if img_sitk:
            # update sitk image with np array
            img_sitk = replace_sitk_with_numpy(img_sitk, img_np)
    
    else:
        # load filename_sitk directly
        if not os.path.exists(filename_sitk):
            raise FileNotFoundError(
                "could not find file {}".format(filename_sitk))
        img_sitk, _ = read_sitk(filename_sitk)
        img_np = sitk.GetArrayFromImage(img_sitk)
    
    if make_3d:
        # convert 2D images to 3D
        # TODO: consider converting img_np to 3D regardless so array in
        #   Image5d is at least 4D
        img_sitk = _make_3d(img_sitk)
        img_np = sitk.GetArrayFromImage(img_sitk)
    
    if config.resolutions is None:
        # fallback to determining metadata directly from sitk file
        _logger.warn(
            "MagellanMapper image metadata file not loaded; will fallback to "
            "%s for metadata", loaded_path)
        config.resolutions = np.array([img_sitk.GetSpacing()[::-1]])
        _logger.debug("set resolutions to %s", config.resolutions)
    
    # add time axis and insert into Image5d with original name
    img5d = np_io.Image5d(
        img_np[None], filename_sitk, img_io=config.LoadIO.SITK)
    if return_sitk:
        return img5d, img_sitk
    return img5d


def load_registered_imgs(
        img_path: str, reg_names: Sequence[str], *args, **kwargs
) -> Dict[str, Union[Union[np.ndarray, "sitk.Image"],
          Tuple[Union[np.ndarray, "sitk.Image"], str]]]:
    """Load atlas-based images registered to another image.
    
    Args:
        img_path: Base image path passed to :meth:`load_registered_img`.
        reg_names: Atlas image suffixes, passed to above function.
        args: Arguments passed to above function.
        kwargs: Arguments passed to above function.
    
    Returns:
        A dictionary of :meth:`load_registered_img` output with ``reg_names``
        values as keys and any unloaded file ignored.
    
    """
    # recursively load sequences of images
    out = {}
    for reg in reg_names:
        try:
            out[reg] = load_registered_img(img_path, reg, *args, **kwargs)
        except FileNotFoundError:
            _logger.warn(
                "%s registered to %s not found, skipping", reg, img_path)
    return out
    

def load_registered_img(
        img_path: str, reg_name: str, get_sitk: bool = False,
        return_path: bool = False
) -> Union[Union[np.ndarray, "sitk.Image"],
           Tuple[Union[np.ndarray, "sitk.Image"], str]]:
    """Load atlas-based image that has been registered to another image.
    
    Args:
        img_path: Path as had been given to generate the registered
            images, with the parent path of the registered images and base name
            of the original image.
        reg_name: Atlas image suffix to open. Can be an absolute path,
            which will be used directly, ignoring ``img_path``.
        get_sitk: True if the image should be returned as a SimpleITK
            image; defaults to False, in which case the corresponding Numpy
            array will be extracted instead.
        return_path: True to return the path from which the image
            was loaded; defaults to False.
    
    Returns:
        - ``reg_img``: the registered image as a Numpy array, or SimpleITK
          Image if ``get_sitk`` is True
        - If ``return_path`` is True: the loaded path
    
    Raises:
        ``FileNotFoundError`` if the path cannot be found.
    
    """
    reg_img_path = reg_name
    if not os.path.isabs(reg_name):
        # use suffix given as an absolute path directly; otherwise, get
        # registered image extension matched to that of main image
        reg_img_path = reg_out_path(img_path, reg_name, True)
    reg_img, reg_img_path = read_sitk(reg_img_path)
    
    if reg_img is None:
        # fallback to loading barren reg_name from img_path's dir
        reg_img_path = os.path.join(
            os.path.dirname(img_path),
            libmag.match_ext(img_path, reg_name))
        reg_img, reg_img_path = read_sitk(reg_img_path)
        if reg_img is None:
            raise FileNotFoundError(
                "could not find registered image from {} and {}"
                .format(img_path, os.path.splitext(reg_name)[0]))
    
    if not get_sitk:
        # convert to ndarray
        reg_img = sitk.GetArrayFromImage(reg_img)
    
    # return image, including path if flagged
    return (reg_img, reg_img_path) if return_path else reg_img


def find_atlas_labels(
        labels_ref_path: str, drawn_labels_only: bool, labels_ref_lookup: Dict
) -> List[int]:
    """Find atlas label IDs from the labels directory.
    
    Args:
        labels_ref_path: Path to labels reference from which to load
            labels from the original labels if ``drawn_labels_only`` is True.
        drawn_labels_only: True to load the atlas in the ``load_labels``
            folder to collect only labels drawn in this atlas; False to use
            all labels in ``labels_ref_lookup``.
        labels_ref_lookup: Labels reverse lookup dictionary of
            label IDs to labels.
    
    Returns:
        List of label IDs.
    
    """
    if drawn_labels_only:
        # need all labels from a reference as registered image may have lost
        # labels; use all drawn labels in original labels image
        orig_labels_sitk, orig_labels_path = read_sitk(reg_out_path(
            os.path.dirname(labels_ref_path),
            config.RegNames.IMG_LABELS.value))
        if orig_labels_sitk is not None:
            config.labels_img_orig = sitk.GetArrayFromImage(orig_labels_sitk)
            return np.unique(config.labels_img_orig).tolist()
    # fall back to using all labels in ontology reference, including any
    # hierarchical labels
    return list(labels_ref_lookup.keys())


def write_registered_image(
        img_np: np.ndarray, img_path: str, reg_name: str,
        img_sitk: Optional["sitk.Image"] = None,
        load_reg_names: Optional[Sequence[str]] = None,
        overwrite: bool = False) -> "sitk.Image":
    """Write a Numpy array as a registered 3D image file through SimpleITK.
    
    To find metadata for the output image, another SimpleITK image must
    be given or discovered as a registered image.
    
    Args:
        img_np: Image array to write.
        img_path: Base path from which to construct output path.
        reg_name: Registered image suffix, which will also specify the
            output file format.
        img_sitk: SimpleITK Image object to use as
            a template for image metadata; defaults to None, in which case
            a registered image will be loaded through ``load_reg_names``.
        load_reg_names: Sequence of registered image suffixes
            from which to load a template image for metdata. Names are
            checked until the first image loads successfully. Defaults to
            None to use the atlas volume followed by the experimental image.
            If ``img_sitk`` is None and no registered image can be found,
            writing will be aborted.
        overwrite: True to overwrite any existing image at the output
            path; defaults to False.

    Returns:
        The saved image as a SimpleITK Image object.
    
    Raises:
        FileExistsError: if ``overwrite`` is false and existing file is at
            the output path.
        FileNotFoundError: if template image cannot be found.

    """
    reg_img_path = reg_out_path(img_path, reg_name)
    if not overwrite and os.path.exists(reg_img_path):
        # avoid overwriting existing file
        raise FileExistsError(
            "{} already exists, will not overwrite".format(reg_img_path))
    
    if img_sitk is None:
        # TODO: consider constructing SimpleITK object from existing metadata
        if load_reg_names is None:
            # default to using basic intensity images
            load_reg_names = (config.RegNames.IMG_ATLAS,
                              config.RegNames.IMG_EXP)
        for name in load_reg_names:
            try:
                # attempt to load another registered image to use as a
                # template for metadata
                img_sitk = load_registered_img(
                    img_path, name.value, get_sitk=True)
                break
            except FileNotFoundError:
                pass
    
    if img_sitk:
        # copy the template and replace its array with the given array to save
        reg_img = replace_sitk_with_numpy(img_sitk, img_np)
        sitk.WriteImage(reg_img, reg_img_path, False)
        print("wrote {} with current registered image".format(reg_img_path))
        return reg_img
    else:
        raise FileNotFoundError(
            f"Unable to find a template file for saving a registered image to "
            f"'{reg_img_path}'")


def write_reg_images(
        imgs_write: Dict[str, "sitk.Image"], prefix: str,
        copy_to_suffix: bool = False, ext: Optional[str] = None,
        prefix_is_dir: bool = False):
    """Write registered images to file.
    
    Args:
        imgs_write: Dictionary of ``{suffix: image}``, where ``suffix``
            is a registered images suffix, such as :const:``IMAGE_LABELS``,
            and ``image`` is a SimpleITK image object. If the image does
            not exist, the file will not be written.
        prefix: Base path from which to construct registered file paths.
            Parent directories will be created if necessary.
        copy_to_suffix: If True, copy the output path to a file in the
            same directory with ``suffix`` as the filename, which may
            be useful when setting the registered images as the
            main images in the directory.
        ext: Replace extension with this value if given; defaults to None.
        prefix_is_dir: True to treat ``prefix`` as a directory.
            If False, will use the directory name of ``prefix``.
    
    """
    # get parent directories and create them if necessary
    target_dir = prefix if prefix_is_dir else os.path.dirname(prefix)
    if len(target_dir) > 0 and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for suffix in imgs_write.keys():
        # write a registered image file for each entry
        img = imgs_write[suffix]
        if img is None: continue
        if ext: suffix = libmag.match_ext(ext, suffix)
        out_path = reg_out_path(prefix, suffix)
        sitk.WriteImage(img, out_path, False)
        print("wrote registered image to", out_path)
        if copy_to_suffix:
            # copy metadata file to allow opening images from bare suffix name,
            # such as when this atlas becomes the new atlas for registration
            out_path_copy = os.path.join(target_dir, suffix)
            shutil.copy(out_path, out_path_copy)
            print("also copied to", out_path_copy)


def write_pts(path: str, pts: Sequence[Sequence[int]], pt_type: str = "index"):
    """Write file for corresponding points in Elastix.
    
    See format described in the Elastix manual, section 4.2 (as of
    Elastix 5.0.1).
    
    Args:
        path: Output path.
        pts: Points as a nested sequence of ints, in `x, y, [z]` order.
        pt_type: Point type, either "index" (default) for points as image
            indices, or "points" as physical coordinates.

    """
    with open(path, mode="w") as f:
        f.write(f"{pt_type}\n")  # point type
        f.write(f"{len(pts)}\n")  # number of points
        for pt in pts:
            # write space-delimited points
            f.write(f"{' '.join([str(p) for p in pt])}\n")


def merge_images(
        img_paths: Sequence[str], reg_name: str, prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        fn_combine: Callable[[Sequence[np.ndarray]], np.ndarray] = np.sum
) -> "sitk.Image":
    """Merge images from multiple paths.
    
    Assumes that the images are relatively similar in size, but will resize 
    them to the size of the first image to combine the images.
    
    Args:
        img_paths: Paths from which registered paths will be found.
        reg_name: Registration suffix to load for the given paths
            in ``img_paths``.
        prefix: Start of output path. If None, the first path in ``img_paths``
            will be used instead.
        suffix: Portion of path to be combined with each path
            in ``img_paths`` and output path.
        fn_combine: Function to apply to combine images with ``axis=0``.
            If None, each image will be inserted as a separate channel.
    
    Returns:
        The combined image in SimpleITK format.
    
    """
    if len(img_paths) < 1: return None
    
    img_sitk = None
    img_nps = []
    for img_path in img_paths:
        mod_path = img_path
        if suffix is not None:
            # adjust image path with suffix
            mod_path = libmag.insert_before_ext(mod_path, suffix)
        print("loading", mod_path)
        # load and resize images to shape of first loaded image
        img, _ = _load_reg_img_to_combine(mod_path, reg_name, img_nps)
        if img_sitk is None: img_sitk = img
    
    # combine images and write single combo image
    if fn_combine is None:
        # combine raw images into separate channels
        img_combo = np.stack(img_nps, axis=img_nps[0].ndim)
    else:
        # merge by custom function
        img_combo = fn_combine(img_nps, axis=0)
    combined_sitk = replace_sitk_with_numpy(img_sitk, img_combo)
    # fallback to using first image's name as base
    output_base = img_paths[0] if prefix is None else prefix
    if suffix is not None:
        output_base = libmag.insert_before_ext(output_base, suffix)
    output_reg = libmag.combine_paths(
        reg_name, config.RegNames.COMBINED.value)
    write_reg_images({output_reg: combined_sitk}, output_base)
    return combined_sitk


def load_numpy_to_sitk(
        img5d: Union[str, "np_io.Image5d"], rotate: bool = False,
        channel: Optional[Union[int, Sequence[int]]] = None) -> "sitk.Image":
    """Load Numpy image array to SimpleITK Image object.

    Use ``channel`` to extract a single channel before generating a
    :obj:`sitk.Image` object for many SimpleITK filters that require
    single-channel ("scalar" rather than "vector") images.
    
    Args:
        img5d: Path to Numpy archive file or ``Image5d`` instance.
        rotate: True if the image should be rotated 180 deg; defaults to
            False.
        channel: Integer or sequence of integers specifying
            channels to keep.
    
    Returns:
        The image in SimpleITK format.
    
    """
    if isinstance(img5d, str):
        # load path
        img5d = importer.read_file(img5d, config.series)
    
    # get np array without time dimension
    image5d = img5d.img
    roi = image5d[0]
    
    if channel is not None and len(roi.shape) >= 4:
        roi = roi[..., channel]
        _logger.debug("Extracted channel(s) for SimpleITK image: %s", channel)
    
    if rotate:
        # rotate 180 deg
        # TODO: consider deprecating, deferring to atlas_refier.transpose_img
        roi = np.rot90(roi, 2, (1, 2))
    
    # get sitk image and transfer metadata to it
    sitk_img = sitk.GetImageFromArray(roi)
    spacing = config.resolutions[0]
    sitk_img.SetSpacing(spacing[::-1])
    # TODO: consider setting z-origin to 0 since image generally as
    #   tightly bound to subject as possible
    #sitk_img.SetOrigin([0, 0, 0])
    sitk_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    #sitk_img.SetOrigin([0, 0, -roi.shape[0]])
    return sitk_img


def sitk_to_itk_img(sitk_img: "sitk.Image") -> "itk.Image":
    """Convert a SimpleITK image to an ITK Image.
    
    Args:
        sitk_img: SimpleITK Image. If None, will simply be returned.

    Returns:
        ITK Image.

    """
    if sitk_img is None: return sitk_img
    
    # construct an ITK Image through the ndarray extracted from the sitk Image
    itk_img = itk.GetImageFromArray(
        sitk.GetArrayFromImage(sitk_img),
        is_vector=sitk_img.GetNumberOfComponentsPerPixel() > 1)
    
    # transfer metadata
    itk_img.SetOrigin(sitk_img.GetOrigin())
    itk_img.SetSpacing(sitk_img.GetSpacing())
    
    # sitk Image directions are flattened arrays, but ITK expects an
    # (ndim x ndim) matrix
    itk_img.SetDirection(itk.GetMatrixFromArray(np.reshape(
        np.array(sitk_img.GetDirection()), [sitk_img.GetDimension()] * 2)))
    
    return itk_img


def itk_to_sitk_img(itk_img: "itk.Image") -> "sitk.Image":
    """Convert an ITK image to a SimpleITK Image.
    
    Args:
        itk_img: ITK Image. If None, will simply be returned.

    Returns:
        SimpleITK Image.

    """
    if itk_img is None: return itk_img
    
    # construct a sitk Image through the ndarray extracted from the ITK Image
    sitk_img = sitk.GetImageFromArray(
        itk.GetArrayFromImage(itk_img),
        isVector=itk_img.GetNumberOfComponentsPerPixel() > 1)
    
    # transfer metadata
    sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
    
    # ITK Image directions are 2D arrays, but ITK expects a 1D array
    sitk_img.SetDirection(np.ravel(np.array(itk_img.GetDirection())))
    
    return sitk_img
