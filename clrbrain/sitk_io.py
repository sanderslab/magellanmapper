#!/bin/bash
# Input/Output for SimpleITK/SimpleElastix objects
# Author: David Young, 2019
"""Input/Output for SimpleITK/SimpleElastix objects.

Manage import and export of :class:`simpleitk.Image` objects.
"""
import os
import warnings

import numpy as np

from clrbrain import config
from clrbrain import importer
from clrbrain import lib_clrbrain

try:
    import SimpleITK as sitk
except ImportError as e:
    sitk = None
    warnings.warn(config.WARN_IMPORT_SITK, ImportWarning)

EXTS_3D = (".mhd", ".mha", ".nii.gz", ".nii", ".nhdr", ".nrrd")


def reg_out_path(file_path, reg_name, match_ext=False):
    """Generate a path for a file registered to another file.
    
    Args:
        file_path: Full path of file registered to.
        reg_name: Suffix for type of registration, eg :const:``IMG_LABELS``.
        match_ext: True to change the extension of ``reg_name`` to match 
            that of ``file_path``.
    
    Returns:
        Full path with the registered filename including appropriate 
        extension at the end.
    """
    file_path_base = importer.filename_to_base(
        file_path, config.series)
    if match_ext:
        reg_name = lib_clrbrain.match_ext(file_path, reg_name)
    return file_path_base + "_" + reg_name


def replace_sitk_with_numpy(img_sitk, img_np):
    """Replace Numpy array in :class:``sitk.Image`` object with a new array.
    
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


def read_sitk(path):
    """Read an image into :class:``sitk.Image`` format, checking for 
    alternative supported extensions if necessary.
    
    Args:
        path: Path, including prioritized extension to check first.
    
    Returns:
        Tuple of the :class:``sitk.Image`` object located at ``path`` 
        and the found extension. If a file at ``path`` cannot be found, 
        its extension is replaced successively with remaining extensions 
        in :const:``EXTS_3D`` until a file is found.
    """
    # prioritize given extension
    path_split = lib_clrbrain.splitext(path)
    exts = list(EXTS_3D)
    if path_split[1] in exts: exts.remove(path_split[1])
    exts.insert(0, path_split[1])
    
    # attempt to load using each extension until found
    img_sitk = None
    path_loaded = None
    for ext in exts:
        img_path = path_split[0] + ext
        if os.path.exists(img_path):
            print("loading image from {}".format(img_path))
            img_sitk = sitk.ReadImage(img_path)
            path_loaded = img_path
            break
    if img_sitk is None:
        print("could not find image from {} and extensions {}"
              .format(path_split[0], exts))
    return img_sitk, path_loaded


def load_registered_img(img_path, reg_name, get_sitk=False, replace=None):
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
    
    Returns:
        The atlas-based image, either as a SimpleITK image or its 
        corresponding Numpy array.
    
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
            lib_clrbrain.match_ext(img_path, reg_name))
        reg_img, reg_img_path = read_sitk(reg_img_path)
        if reg_img is None:
            raise FileNotFoundError(
                "could not find registered image from {} and {}"
                .format(img_path, os.path.splitext(reg_name)[0]))
    if replace is not None:
        reg_img = replace_sitk_with_numpy(reg_img, replace)
        sitk.WriteImage(reg_img, reg_img_path, False)
        print("replaced {} with current registered image".format(reg_img_path))
    if get_sitk:
        return reg_img
    return sitk.GetArrayFromImage(reg_img)


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
