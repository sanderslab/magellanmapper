# Input/Output for SimpleITK/SimpleElastix objects
# Author: David Young, 2019
"""Input/Output for SimpleITK/SimpleElastix objects.

Manage import and export of :class:`simpleitk.Image` objects.
"""
import os
import shutil

import numpy as np
import SimpleITK as sitk
from skimage import transform

from magmap.settings import config
from magmap.io import importer
from magmap.io import libmag

EXTS_3D = (".mhd", ".mha", ".nii.gz", ".nii", ".nhdr", ".nrrd")


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


def replace_sitk_with_numpy(img_sitk, img_np):
    """Generate a :class:``sitk.Image`` object based on another Image object,
    but replace its array with a new Numpy array.
    
    Args:
        img_sitk: Image object to use as template.
        img_np: Numpy array to swap in.
    
    Returns:
        New :class:``sitk.Image`` object with same spacing, origin, and 
        direction as that of ``img_sitk`` and array replaced with ``img_np``.
    """
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    direction = img_sitk.GetDirection()
    img_sitk_back = sitk.GetImageFromArray(img_np)
    img_sitk_back.SetSpacing(spacing)
    img_sitk_back.SetOrigin(origin)
    img_sitk_back.SetDirection(direction)
    return img_sitk_back


def match_world_info(source, target):
    """Copy world information (eg spacing, origin, direction) from one
    image object to another.

    This matching is sometimes necessary for slight differences in
    metadata perhaps from founding that may prevent ITK filters from executing.

    Args:
        source (:obj:`sitk.Image`): Source object whose relevant metadata
            will be copied into ``target``.
        target (:obj:`sitk.Image`): Target object whose corresponding
            metadata will be overwritten by that of ``source``.

    """
    spacing = source.GetSpacing()
    origin = source.GetOrigin()
    direction = source.GetDirection()
    print("Adjusting spacing from {} to {}, origin from {} to {}, "
          "direction from {} to {}"
          .format(target.GetSpacing(), spacing, target.GetOrigin(), origin,
                  target.GetDirection(), direction))
    target.SetSpacing(spacing)
    target.SetOrigin(origin)
    target.SetDirection(direction)


def read_sitk(path, dryrun=False):
    """Read an image file into :class:``sitk.Image`` format, checking for 
    alternative supported extensions if necessary.
    
    Args:
        path (str): Path, including prioritized extension to check first.
        dryrun (bool): True to load the image; defaults to False. Use False
            to test whether an path to load is found.
    
    Returns:
        :obj:`sitk.Image`, str: Image object located at ``path`` with
        the found extension, or None if unable to load; and the loaded path,
        or None if no matching, existing path is found. If a file at
        ``path`` cannot be found, its extension is replaced successively
        with remaining extensions in :const:``EXTS_3D`` until a file is found.
    
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
                print("Loading image with SimpleITK:", img_path)
                img_sitk = sitk.ReadImage(img_path)
            path_loaded = img_path
            break
    if not dryrun and img_sitk is None:
        print("could not find image from {} and extensions {}"
              .format(path_split[0], exts))
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


def read_sitk_files(filename_sitk, reg_names=None, return_sitk=False):
    """Read files through SimpleITK and export to Numpy array format, 
    loading associated metadata.

    Args:
        filename_sitk: Path to file in a format that can be read by SimpleITK.
        reg_names: Path or sequence of paths of registered names. Can 
            be a registered suffix or a full path. Defaults to None.
        return_sitk (bool): True to return the loaded SimpleITK Image object.

    Returns:
        :class:`numpy.ndarray`: Image array in Numpy 3D format (or 4D if
        multi-channel). Associated metadata is loaded into :module:`config`
        attributes.
        
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
    else:
        # load filename_sitk directly
        if not os.path.exists(filename_sitk):
            raise FileNotFoundError(
                "could not find file {}".format(filename_sitk))
        img_sitk, _ = read_sitk(filename_sitk)
        img_np = sitk.GetArrayFromImage(img_sitk)
    
    if config.resolutions is None:
        # fallback to determining metadata directly from sitk file
        libmag.warn(
            "MagellanMapper image metadata file not loaded; will fallback to "
            "{} for metadata".format(loaded_path))
        config.resolutions = np.array([img_sitk.GetSpacing()[::-1]])
        print("set resolutions to {}".format(config.resolutions))
    
    if return_sitk:
        return img_np, img_sitk
    return img_np


def load_registered_img(img_path, reg_name, get_sitk=False, replace=None,
                        return_path=False):
    """Load atlas-based image that has been registered to another image.
    
    Args:
        img_path: Path as had been given to generate the registered images, 
            with the parent path of the registered images and base name of 
            the original image.
        reg_name: Atlas image suffix to open.
        get_sitk: True if the image should be returned as a SimpleITK image; 
            defaults to False, in which case the corresponding Numpy array will 
            be extracted instead.
        replace: Numpy image with which to replace and overwrite the loaded 
            image. Defaults to None, in which case no replacement will take 
            place.
        return_path (bool): True to return the path from which the image
            was loaded; defaults to False.
    
    Returns:
        :obj:`np.ndarray`: The atlas-based image as a Numpy array, or a
        :obj:`sitk.Image` object if ``get_sitk`` is True. Also returns the
        loaded path if ``return_path`` is True.
    
    Raises:
        ``FileNotFoundError`` if the path cannot be found.
    """
    # prioritize registered image extension matched to that of main image
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
    if replace is not None:
        reg_img = replace_sitk_with_numpy(reg_img, replace)
        sitk.WriteImage(reg_img, reg_img_path, False)
        print("replaced {} with current registered image".format(reg_img_path))
    if not get_sitk:
        reg_img = sitk.GetArrayFromImage(reg_img)
    return (reg_img, reg_img_path) if return_path else reg_img


def find_atlas_labels(load_labels, max_level, labels_ref_lookup):
    """Find atlas label IDs from the labels directory.
    
    Args:
        load_labels: Path to labels directory.
        max_level: Labels level, where None indicates that only the 
            drawn labels should be found, whereas an int level will 
            cause the full set of labels from ``labels_ref_lookup`` 
            to be taken.
        labels_ref_lookup: Labels reverse lookup dictionary of 
            label IDs to labels.
    
    Returns:
        Sequence of label IDs.
    """
    orig_atlas_dir = os.path.dirname(load_labels)
    orig_labels_path = os.path.join(
        orig_atlas_dir, config.RegNames.IMG_LABELS.value)
    # need all labels from a reference as registered image may have lost labels
    if max_level is None and os.path.exists(orig_labels_path):
        # use all drawn labels in original labels image
        config.labels_img_orig = load_registered_img(
            config.load_labels, config.RegNames.IMG_LABELS.value)
        orig_labels_sitk = sitk.ReadImage(orig_labels_path)
        orig_labels_np = sitk.GetArrayFromImage(orig_labels_sitk)
        label_ids = np.unique(orig_labels_np).tolist()
    else:
        # use all labels in ontology reference to include hierarchical 
        # labels or if original labels image isn't present
        label_ids = list(labels_ref_lookup.keys())
    return label_ids


def write_reg_images(imgs_write, prefix, copy_to_suffix=False, ext=None,
                     prefix_is_dir=False):
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
            main images in the directory. Defaults to False.
        ext: Replace extension with this value if given; defaults to None.
        prefix_is_dir (bool): True to treat ``prefix`` as a directory;
            defaults to False to get the directory name of ``prefix``.
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


def merge_images(img_paths, reg_name, prefix=None, suffix=None, 
                 fn_combine=np.sum):
    """Merge images from multiple paths.
    
    Assumes that the images are relatively similar in size, but will resize 
    them to the size of the first image to combine the images.
    
    Args:
        img_paths: Paths from which registered paths will be found.
        reg_name: Registration suffix to load for the given paths 
            in ``img_paths``.
        prefix: Start of output path; defaults to None to use the first 
           path in ``img_paths`` instead.
        suffix: Portion of path to be combined with each path 
            in ``img_paths`` and output path; defaults to None.
        fn_combine: Function to apply to combine images with ``axis=0``. 
            Defaults to :func:``np.sum``. If None, each image will be 
            inserted as a separate channel.
    
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
