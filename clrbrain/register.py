#!/bin/bash
# Image registration
# Author: David Young, 2017, 2019
"""Register images to one another.

The registration type can be given on the command-line (see :mod:``cli``) as 
outlined here. Each type can be coupled with additional arguments in ``cli``.
    * ``single``: Register a single image to a reference atlas. Filenames 
        should be given in :attr:``config.filenames``, where the first 
        filename is the fixed image (eg the experiment), and the 
        second image is the moving image (eg the atlas). The atlas rather 
        than the experimental image is typically moved here since the 
        experimental image is usually much larger and more difficult to move. 
        For faster registration, a downsampled image can be given as this 
        experiment image, and the third argument to ``filenames`` can 
        optionally be given as a name prefix corresponding to the original 
        experiment file. ``flip`` can also be given to flip the 
    * ``group``: Register multiple experimental images to one another via 
        groupwise registration. All the images given in 
        :attr:``config.filenames`` will be registered to one another, except 
        the last filename, which will be used as the output name prefix.
    * ``overlays``: Overlay the moving image on top of the fixed image to 
        visualize alignment. ``flip`` can be given to flip the moving image.
    * ``volumes``: Calculate volumes for each region in the experimental 
        image based on the corresponding registered labels image. ``labels`` 
        should be given to specify the labels file and level. ``rescale`` is 
        used to find the rescaled, typically downsampled images to use as a 
        much faster way to calculate volumes. ``no_show`` suppresses 
        graph display.
    *   ``densities``: Similar to ``volumes`` but include nuclei densities 
        per region by loading the processed stack file.
    *   ``export_vols``: Export regional volumes and related measures to CSV 
        file.
    *   ``export_regions``: Export atlas annotation region information such 
        as region ID to name CSV file and region network SIF file.
    *   ``new_atlas``: Based on "single" registration, outputting registered 
        files in the naming format of a new atlas and bypassing label 
        truncation.
"""

import os
import copy
import csv
from enum import Enum
import multiprocessing as mp
from collections import OrderedDict
import shutil
from time import time
import warnings
import pandas as pd
try:
    import SimpleITK as sitk
except ImportError as e:
    print(e)
    print("WARNING: SimpleElastix could not be found, so there will be error "
          "when attempting to register images or load registered images")
import numpy as np
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import transform

from clrbrain import config
from clrbrain import detector
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import ontology
from clrbrain import plot_2d
from clrbrain import plot_3d
from clrbrain import plot_support
from clrbrain import segmenter
from clrbrain import stats
from clrbrain import transformer
from clrbrain import vols

# registered image suffixes
IMG_ATLAS = "atlasVolume.mhd"
IMG_LABELS = "annotation.mhd"
IMG_EXP = "exp.mhd"
IMG_GROUPED = "grouped.mhd"
IMG_BORDERS = "borders.mhd"
IMG_HEAT_MAP = "heat.mhd"
IMG_ATLAS_EDGE = "atlasEdge.mhd"
IMG_ATLAS_LOG = "atlasLoG.mhd"
IMG_LABELS_TRUNC = "annotationTrunc.mhd"
IMG_LABELS_EDGE = "annotationEdge.mhd"
IMG_LABELS_DIST = "annotationDist.mhd"
IMG_LABELS_MARKERS = "annotationMarkers.mhd"
IMG_LABELS_INTERIOR = "annotationInterior.mhd"
IMG_LABELS_SUBSEG = "annotationSubseg.mhd"
IMG_LABELS_DIFF = "annotationDiff.mhd"
IMG_LABELS_LEVEL = "annotationLevel{}.mhd"
IMG_LABELS_EDGE_LEVEL = "annotationEdgeLevel{}.mhd"

SAMPLE_VOLS = "vols_by_sample"
SAMPLE_VOLS_LEVELS = SAMPLE_VOLS + "_levels"
SAMPLE_VOLS_SUMMARY = SAMPLE_VOLS + "_summary"

COMBINED_SUFFIX = "combined"
REREG_SUFFIX = "rereg"

SMOOTHING_METRIC_MODES = (
    "vol", "area_edt", "area_radial", "area_displvol", "compactness")
# 3D format extensions to check when finding registered files
EXTS_3D = (".mhd", ".mha", ".nii.gz", ".nii", ".nhdr", ".nrrd")
_SIGNAL_THRESHOLD = 0.01

class AtlasMetrics(Enum):
    """General atlas metric enumerations."""
    SAMPLE = "Sample"
    REGION = "Region"
    CONDITION = "Condition"
    DSC_ATLAS_LABELS = "DSC_atlas_labels"
    DSC_ATLAS_SAMPLE = "DSC_atlas_sample"
    DSC_ATLAS_SAMPLE_CUR = "DSC_atlas_sample_curated"

class SmoothingMetrics(Enum):
    """Smoothing metric enumerations."""
    COMPACTED = "Compacted"
    DISPLACED = "Displaced"
    SM_QUALITY = "Smoothing_quality"
    COMPACTNESS = "Compactness"
    DISPLACEMENT = "Displacement"
    ROUGHNESS = "Roughness"
    ROUGHNESS_SM = "Roughness_sm"
    SA_VOL_ABS = "SA_to_vol_abs"
    SA_VOL = "SA_to_vol"
    LABEL_LOSS = "Label_loss"
    FILTER_SIZE = "Filter_size"

def _reg_out_path(file_path, reg_name, match_ext=False):
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

def _translation_adjust(orig, transformed, translation, flip=False):
    """Adjust translation based on differences in scaling between original 
    and transformed images to allow the translation to be applied to the 
    original image.
    
    Assumes (x, y, z) order for consistency with SimpleITK since this method 
    operates on SimpleITK format images.
    
    Args:
        orig: Original image in SimpleITK format.
        transformed: Transformed image in SimpleITK format.
        translation: Translation in (x, y, z) order, taken from transform 
            parameters and scaled to the transformed images's spacing.
    
    Returns:
        The adjusted translation in (x, y, z) order.
    """
    if translation is None:
        return translation
    # TODO: need to check which space the TransformParameter is referring to 
    # and how to scale it since the adjusted translation does not appear to 
    # be working yet
    orig_origin = orig.GetOrigin()
    transformed_origin = transformed.GetOrigin()
    origin_diff = np.subtract(transformed_origin, orig_origin)
    print("orig_origin: {}, transformed_origin: {}, origin_diff: {}"
          .format(orig_origin, transformed_origin, origin_diff))
    orig_size = orig.GetSize()
    transformed_size = transformed.GetSize()
    size_ratio = np.divide(orig_size, transformed_size)
    print("orig_size: {}, transformed_size: {}, size_ratio: {}"
          .format(orig_size, transformed_size, size_ratio))
    translation_adj = np.multiply(translation, size_ratio)
    #translation_adj = np.add(translation_adj, origin_diff)
    print("translation_adj: {}".format(translation_adj))
    if flip:
        translation_adj = translation_adj[::-1]
    return translation_adj

def _show_overlays(imgs, translation, fixed_file, plane):
    """Shows overlays via :func:plot_2d:`plot_overlays_reg`.
    
    Args:
        imgs: List of images in Numpy format
        translation: Translation in (z, y, x) format for Numpy consistency.
        fixed_file: Path to fixed file to get title.
        plane: Planar transposition.
    """
    cmaps = ["Blues", "Oranges", "prism"]
    #plot_2d.plot_overlays(imgs, z, cmaps, os.path.basename(fixed_file), aspect)
    show = not config.no_show
    plot_2d.plot_overlays_reg(
        *imgs, *cmaps, translation, os.path.basename(fixed_file), plane, show)

def _handle_transform_file(fixed_file, transform_param_map=None):
    base_name = _reg_out_path(fixed_file, "")
    filename = base_name + "transform.txt"
    param_map = None
    if transform_param_map is None:
        param_map = sitk.ReadParameterFile(filename)
    else:
        sitk.WriteParameterFile(transform_param_map[0], filename)
        param_map = transform_param_map[0]
    return param_map, None # TODO: not using translation parameters
    transform = np.array(param_map["TransformParameters"]).astype(np.float)
    spacing = np.array(param_map["Spacing"]).astype(np.float)
    len_spacing = len(spacing)
    #spacing = [16, 16, 20]
    translation = None
    # TODO: should parse the transforms into multiple dimensions
    if len(transform) == len_spacing:
        translation = np.divide(transform[0:len_spacing], spacing)
        print("transform: {}, spacing: {}, translation: {}"
              .format(transform, spacing, translation))
    else:
        print("Transform parameters do not match scaling dimensions")
    return param_map, translation

def _get_bbox(img_np, threshold=10):
    """Get the bounding box for the largest object within an image.
    
    Args:
        img_np: Image as a Numpy array.
        threshold: Threshold level; defaults to 10. If None, assume 
            ``img_np`` is already binary.
    
    Returns:
        Bounding box of the largest object in the image.
    """
    props_sizes = plot_3d.get_thresholded_regionprops(
        img_np, threshold=threshold, sort_reverse=True)
    if props_sizes is None: return None
    labels_bbox = props_sizes[0][0].bbox
    #print("bbox: {}".format(labels_bbox))
    return labels_bbox

def _truncate_labels(img_np, x_frac=None, y_frac=None, z_frac=None):
    """Truncate image by zero-ing out pixels outside of given bounds.
    
    Args:
        img_np: 3D image array in Numpy format.
        x_frac: 2D tuple of (x_start, x_end), given as fractions of the image. 
            Defaults to None, in which case the full image in that direction 
            will be preserved.
        y_frac: Same as ``x_frac`` but for y-axis.
        z_frac: Same as ``x_frac`` but for z-axis.
    
    Returns:
        The truncated image.
    """
    shape = img_np.shape
    bounds = (z_frac, y_frac, x_frac)
    axis = 0
    bounds_len = len(bounds)
    for bound in bounds:
        if bound is not None:
            # convert boundary fractions to absolute coordinates for the 
            # given axis, filling the other axes with full slices
            bound_abs = np.multiply(bound, shape[axis]).astype(np.int)
            slices = [slice(None)] * bounds_len
            slices[axis] = slice(0, bound_abs[0])
            img_np[tuple(slices)] = 0
            slices[axis] = slice(bound_abs[1], None)
            img_np[tuple(slices)] = 0
            print("truncated axis {} outside of bounds {}"
                  .format(axis, bound_abs))
        axis += 1
    return img_np

def _mirror_planes(img_np, start, mirror_mult=1, resize=True, start_dup=None, 
                   rand_dup=None, check_equality=False):
    """Mirror image across its sagittal midline.
    
    Args:
        img: Labels image in Numpy format, which will be edited directly 
            unless ``resize`` is True.
        start: Starting index at which to begin mirroring, inclusive.
        mirror_mult: Multiplier for mirrored portion of image, such as -1 
            when creating a labels map with one side distinct from the other; 
            defaults to 1.
        resize: True if the image should be resized to be symmetric in size 
            across ``start``; defaults to True.
        start_dup: Fraction at which to start duplicating planes before 
            mirroring, which may be useful for atlases where entired labeled 
            regions are past midline, where duplication of a few atlas 
            planes may simply expand rather than duplicating structures; 
            defaults to None.
        rand_dup: Multiplier for randomizer to choose duplicate planes from 
            a random selection within this given range prior to ``start_dup``. 
            Defaults to None, in which case the plane just prior to 
            the duplication starting plane will simply be duplicated 
            throughout the region.
        check_equality: True to check equality from one half to the other 
            along axis 0; defaults to False.
    
    Returns:
        The mirrored image in Numpy format.
    """
    if resize:
        # made mirror index midline by adjusting image shape
        shape = img_np.shape
        shape_resized = np.copy(shape)
        shape_resized[0] = start * 2
        print("original shape: {}, new shape: {}".format(shape, shape_resized))
        if shape_resized[0] > shape[0]:
            # shift image into larger array, adding extra z padding
            img_resized = np.zeros(shape_resized, dtype=img_np.dtype)
            img_resized[:shape[0]] = img_np
            img_np = img_resized
        else:
            # truncate original image to fit into smaller shape
            img_np = img_np[:shape_resized[0]]
    tot_planes = len(img_np)
    if start_dup is not None:
        # duplicate planes starting at this fraction of total planes, 
        # typically corresponding to the true midline
        n = int(start_dup * tot_planes)
        num_planes = start - n
        if rand_dup is not None:
            # randomly pick planes from 1 to rand_dup behind the starting plane
            np.random.seed(num_planes)
            dup = n - np.ceil(np.random.rand(num_planes) * rand_dup)
            dup = dup.astype(int)
            dup[dup < 0] = 0
        else:
            # duplicate the prior plane through the entire region
            dup = np.repeat(n - 1, num_planes)
        for i in range(num_planes):
            plane_i = n + i
            if plane_i > 0 and plane_i < tot_planes:
                print("duplicating plane {} from {}".format(plane_i, dup[i]))
                img_np[plane_i] = img_np[dup[i]]
    if start <= tot_planes and start >= 0:
        # if empty planes at end, fill the empty space with the preceding 
        # planes in mirrored fashion
        remaining_planes = tot_planes - start
        end = start - remaining_planes - 1
        if end < 0:
            end = None # to slice to beginning with neg increment
            remaining_planes = start
        print("start: {}, end: {}, remaining_planes: {}, tot_planes: {}"
              .format(start, end, remaining_planes, tot_planes))
        img_np[start:start+remaining_planes] = np.multiply(
            img_np[start-1:end:-1], mirror_mult)
    else:
        # skip mirroring if no planes are empty or only first plane is empty
        print("nothing to mirror")
    if check_equality:
        print("Checking labels symmetry after mirroring:")
        check_mirrorred(img_np, mirror_mult=mirror_mult)
    return img_np

def check_mirrorred(img_np, mirror_mult=1):
    """Check whether a given image, typically a labels image, is symmetric.
    
    Args:
        img_np: Numpy array of image, typically a labels image, where 
            symmetry will be detected based on equality of two halves 
            split by the image's last dimension.
        mirror_mult: Number by which to divide the 2nd half before 
            checking for symmetry; defaults to 1. Typically a number 
            used to generate the 2nd half when mirroring.
    
    Returns:
        Tuple of ``(boolean, boolean)``, where the first value is True if 
        the sides are equal, and the 2nd is True if the sides have equal 
        sets of unique values.
    """
    half_len = len(img_np) // 2
    half_before = img_np[:half_len]
    half_after = img_np[:half_len-1:-1] / mirror_mult
    equality_vals = np.array_equal(half_before, half_after)
    print("halves equal?", equality_vals)
    equality_lbls = np.array_equal(
        np.unique(half_before), np.unique(half_after))
    print("same labels in each half?", equality_lbls)
    return equality_vals, equality_lbls

def _curate_labels(img, img_ref, mirror=None, edge=None, expand=None, 
                   rotate=None, smooth=None, affine=None, resize=True):
    """Curate labels through extension, mirroring, and smoothing.
    
    Extension fills in missing edge labels by extrapolating edge planes 
    to the volume of the underlying reference image. Mirroring copies labels 
    from one half of the image to the other for symmetry. Smoothing 
    removes jagged edges from slightly misaligned labels.
    
    Assume that the image is in sagittal sections, typically consisting of 
    only one hemisphere, empty from the far z planes toward the middle but not 
    necessarily the exact middle of the image. The egde to extend are low z 
    planes. Mirroring takes place across a z-plane, ideally at the true 
    midline.
    
    Args:
        img: Labels image in SimpleITK format.
        img_ref: Reference atlas image in SimpleITK format.
        mirror: Fraction of planes at which to start mirroring; 
            defaults to None, in which case mirroring will be skipped. 
            -1 will cause the mirror plane to be found automatically 
            based on the first plane completely without labels, starting 
            from the highest plane and working downward.
        edge: Fraction of planes at which to start extending the near 
            (assumed to be lateral) edge; defaults to None, in which 
            case edge extension will be skipped. -1 will cause the 
            edge plane to be found automatically based on the first 
            plane with any labels, starting from the lowest plane.
        expand: Tuple of 
            ((x_pixels_start, end), (y, ...), ...), (next_region, ...)) 
            specifying slice boundaries for regions to expand the labels to 
            the size of the atlas. Defaults to None.
        rotate: Tuple of ((angle0, axis0), ...) by which to rotate the 
            labels. Defaults to None to not rotate.
        smooth: Filter size for smoothing, or sequence of sizes to test 
            various smoothing strengths via multiprocessing. Defaults to 
            None to not smooth..
        affine: Dictionary for selective affine transformation, passed 
            to :func:`plot_3d.affine_nd`. Defaults to None to not 
            affine transform.
        resize: True to resize the image during mirroring; defaults to True.
    
    Returns:
        Tuple of the mirrored image in Numpy format; a tuple of the 
        indices at which the edge was extended and theimage was 
        mirrored, respectively; a borders image of the same size as the 
        mirrored image, or None if any smoothing did not give a 
        borders image; and a data frame of smoothing stats, or None 
        if smoothing was not performed.
    """
    # cast to signed int that takes the full range of the labels image
    img_np = sitk.GetArrayFromImage(img)
    try:
        dtype = lib_clrbrain.dtype_within_range(0, np.amax(img_np), True, True)
        if dtype != img_np.dtype:
            print("casting labels image to type", dtype)
            img_np = img_np.astype(dtype)
    except TypeError as e:
        # fallback to large signed data type
        print(e)
        img_np.astype(np.int32)
    img_ref_np = sitk.GetArrayFromImage(img_ref)
    tot_planes = len(img_np)
    
    # extend near edges: lateral edges of atlas labels (ie low z) are missing 
    # in many ABA developing mouse atlases, requiring extension
    edgei = 0
    edgei_first = None
    if edge is not None:
        if edge == -1:
            # find the first non-zero plane
            for plane in img_np:
                if not np.allclose(plane, 0):
                    if edgei_first is None:
                        edgei_first = edgei
                    elif edgei - edgei_first >= 1:
                        # require at least 2 contiguous planes with signal
                        edgei = edgei_first
                        print("found start of contiguous non-zero planes at {}"
                              .format(edgei))
                        break
                else:
                    edgei_first = None
                edgei += 1
        else:
            # based on profile settings
            edgei = int(edge * tot_planes)
            print("will extend near edge from plane {}".format(edgei))
        
        # find the bounds of the reference image in the given plane and resize 
        # the corresponding section of the labels image to the bounds of the 
        # reference image in the next plane closer to the edge, recursively 
        # extending the nearest plane with labels based on the underlying 
        # atlas; assume that each nearer plane is the same size or smaller 
        # than the next farther plane, such as a tapering specimen
        plot_3d.extend_edge(
            img_np, img_ref_np, config.register_settings["atlas_threshold"], 
            None, edgei)
    
    if expand:
        # expand selected regions
        for expand_limits in expand:
            # get region from slices specified by tuple of (start, end) pixels
            expand_slices = tuple(
                slice(*limits) for limits in expand_limits[::-1])
            region = img_np[expand_slices]
            region_ref = img_ref_np[expand_slices]
            print("expanding planes in slices", expand_slices)
            for expandi in range(len(region_ref)):
                # find bounding boxes for labels and atlas within region
                bbox = _get_bbox(region[expandi], 0) # assume pos labels region
                shape, slices = plot_3d.get_bbox_region(bbox)
                plane_region = region[expandi, slices[0], slices[1]]
                bbox_ref = _get_bbox(region_ref[expandi])
                shape_ref, slices_ref = plot_3d.get_bbox_region(bbox_ref)
                # expand bounding box region of labels to that of atlas
                plane_region = transform.resize(
                    plane_region, shape_ref, preserve_range=True, order=0, 
                    anti_aliasing=False, mode="reflect")
                region[expandi, slices_ref[0], slices_ref[1]] = plane_region
    
    # find approximate midline by locating the last zero plane from far edge 
    # at which to start mirroring across midline
    mirrori = tot_planes
    for plane in img_np[::-1]:
        if not np.allclose(plane, 0):
            print("found last zero plane from far border at {}".format(mirrori))
            break
        mirrori -= 1
    
    if rotate:
        if mirror is not None:
            # mirroring labels with original values in case rotation will cause 
            # some labels to be cut off, then rotate for each specified axis
            for i in range(mirrori, tot_planes):
                img_np[i] = img_np[mirrori - 1]
        for rot in rotate:
            print("rotating by", rot)
            img_np = plot_3d.rotate_nd(img_np, rot[0], rot[1], order=0)
    
    if affine:
        for aff in affine:
            print("performing affine of", aff)
            img_np = plot_3d.affine_nd(img_np, **aff)
    
    if mirror is not None and mirror != -1:
        # reset mirror based on fractional profile setting
        mirrori = int(mirror * tot_planes)
        print("will mirror starting at plane index {}".format(mirrori))
    
    borders_img_np = None
    df_smoothing = None
    if smooth is not None:
        # minimize jaggedness in labels, often seen outside of the original 
        # orthogonal direction, using pre-mirrored slices only since rest will 
        # be overwritten
        img_smoothed = img_np[:mirrori]
        img_smoothed_orig = np.copy(img_smoothed)
        borders = None
        if lib_clrbrain.is_seq(smooth):
            # test sequence of filter sizes via multiprocessing, in which 
            # case the original array will be left unchanged
            df_smoothing = _smoothing_mp(
                img_smoothed, img_smoothed_orig, smooth)
        else:
            # single filter size
            _, df_smoothing, borders = _smoothing(
                img_smoothed, img_smoothed_orig, smooth)
        if borders is not None:
            # mirror borders image
            shape = list(borders.shape)
            shape[0] = img_np.shape[0]
            borders_img_np = np.zeros(shape, dtype=np.int32)
            borders_img_np[:mirrori] = borders
            borders_img_np = _mirror_planes(
                borders_img_np, mirrori, -1, resize=resize)
    
    # check that labels will fit in integer type
    lib_clrbrain.printv(
        "type: {}, max: {}, max avail: {}".format(
            img_np.dtype, np.max(img_np), np.iinfo(img_np.dtype).max))
    
    if mirror is None:
        print("Checking baseline labels symmetry without mirroring:")
        check_mirrorred(img_np)
    else:
        # mirror, check beforehand for labels that will be loss
        labels_lost(np.unique(img_np), np.unique(img_np[:mirrori]))
        img_np = _mirror_planes(
            img_np, mirrori, mirror_mult=-1, check_equality=True, resize=resize)
        print("total final labels: {}".format(np.unique(img_np).size))
    return img_np, (edgei, mirrori), borders_img_np, df_smoothing

def _crop_to_orig(labels_img_np_orig, labels_img_np, crop):
    # crop new labels to extent of original labels unless crop is False
    print("cropping to original labels' extent with filter size of", crop)
    if crop is False: return
    mask = labels_img_np_orig == 0
    if crop > 0:
        # smooth mask
        mask = morphology.binary_opening(mask, morphology.ball(crop))
    labels_img_np[mask] = 0

def _smoothing(img_np, img_np_orig, filter_size, save_borders=False):
    """Smooth image and calculate smoothing metric for use individually or 
    in multiprocessing.
    
    Args:
        img_np: Image as Numpy array, which will be directly updated.
        img_np_orig: Original image as Numpy array for comparison with 
            smoothed image in metric.
        filter_size: Structuring element size for smoothing.
        save_borders: True to save borders; defaults to False
    
    Returns:
        Tuple of ``filter_size``; a data frame of smoothing metrices; 
        and the borders image.
    """
    smoothing_mode = config.register_settings["smoothing_mode"]
    smooth_labels(img_np, filter_size, smoothing_mode)
    borders, df_metrics, df_raw = label_smoothing_metric(
        img_np_orig, img_np, save_borders=save_borders)
    df_metrics[SmoothingMetrics.FILTER_SIZE.value] = [filter_size]
    
    # curate back to lightly smoothed foreground of original labels
    crop = config.register_settings["crop_to_orig"]
    _crop_to_orig(img_np_orig, img_np, crop)
    
    print("\nMeasuring foreground overlap of labels after smoothing:")
    measure_overlap_labels(
        make_labels_fg(sitk.GetImageFromArray(img_np)), 
        make_labels_fg(sitk.GetImageFromArray(img_np_orig)))
    
    return filter_size, df_metrics, borders

def _smoothing_mp(img_np, img_np_orig, filter_sizes):
    """Smooth image and calculate smoothing metric for a list of smoothing 
    strengths.
    
    Args:
        img_np: Image as Numpy array, which will be directly updated.
        img_np_orig: Original image as Numpy array for comparison with 
            smoothed image in metric.
        filter_size: Tuple or list of structuring element sizes.
    
    Returns:
        Data frame of combined metrics from smoothing for each filter size.
    """
    pool = mp.Pool()
    pool_results = []
    for n in filter_sizes:
        pool_results.append(
            pool.apply_async(_smoothing, args=(img_np, img_np_orig, n)))
    dfs = []
    for result in pool_results:
        filter_size, df_metrics, _ = result.get()
        dfs.append(df_metrics)
        print("finished smoothing with filter size {}".format(filter_size))
    pool.close()
    pool.join()
    df = pd.concat(dfs)
    return df

def labels_lost(label_ids_orig, label_ids, label_img_np_orig=None):
    print("Measuring label loss:")
    print("number of labels changed from {} to {}"
          .format(label_ids_orig.size, label_ids.size))
    labels_lost = label_ids_orig[np.isin(
        label_ids_orig, label_ids, invert=True)]
    print("IDs of labels lost: {}".format(labels_lost))
    if label_img_np_orig is not None:
        for lost in labels_lost:
            region_lost = label_img_np_orig[label_img_np_orig == lost]
            print("size of lost label {}: {}".format(lost, region_lost.size))
    return labels_lost

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

def smooth_labels(labels_img_np, filter_size=3, mode=None):
    """Smooth each label within labels annotation image.
    
    Labels images created in one orthogonal direction may have ragged, 
    high-frequency edges when viewing in the other orthogonal directions. 
    Smooth these edges by applying a filter to each label.
    
    Args:
        labels_img_np: Labels image as a Numpy array.
        filter_size: Structuring element or kernel size; defaults to 3.
        mode: One of :attr:``config.SmoothingModes``, where ``opening`` applies 
            a morphological opening filter unless the size is severely 
            reduced, in which case a closing filter is applied instead;  
            ``gaussian`` applies a Gaussian blur; and ``closing`` applies 
            a closing filter only.
    """
    if mode is None: mode = config.SmoothingModes.opening
    print("Smoothing labels with filter size of {}, mode {}"
          .format(filter_size, mode))
    if filter_size == 0:
        print("filter size of 0, skipping")
        return
    
    # copy original for comparison
    labels_img_np_orig = np.copy(labels_img_np)
    
    # sort labels by size, starting from largest to smallest
    label_ids = np.unique(labels_img_np)
    label_sizes = {}
    for label_id in label_ids:
        label_sizes[label_id] = len(labels_img_np[labels_img_np == label_id])
    label_sizes_ordered = OrderedDict(
        sorted(label_sizes.items(), key=lambda x: x[1], reverse=True))
    label_ids_ordered = label_sizes_ordered.keys()
    
    for label_id in label_ids_ordered:
        # smooth by label
        print("smoothing label ID {}".format(label_id))
        
        # get bounding box for label region
        bbox = plot_3d.get_label_bbox(labels_img_np, label_id)
        if bbox is None: continue
        _, slices = plot_3d.get_bbox_region(
            bbox, np.ceil(2 * filter_size).astype(int), labels_img_np.shape)
        
        # get region, skipping if no region left
        region = labels_img_np[tuple(slices)]
        label_mask_region = region == label_id
        region_size = np.sum(label_mask_region)
        if region_size == 0:
            print("no pixels to smooth, skipping")
            continue
        
        # smoothing based on mode
        region_size_smoothed = 0
        if mode is config.SmoothingModes.opening:
            # smooth region with opening filter, reducing filter size for 
            # small volumes and changing to closing filter 
            # if region would be lost or severely reduced
            selem_size = filter_size
            if region_size < 5000:
                selem_size = selem_size // 2
                print("using a smaller filter size of {} for a small region "
                      "of {} pixels".format(selem_size, region_size))
            selem = morphology.ball(selem_size)
            smoothed = morphology.binary_opening(label_mask_region, selem)
            region_size_smoothed = np.sum(smoothed)
            size_ratio = region_size_smoothed / region_size
            if size_ratio < 0.01:
                print("region would be lost or too small "
                      "(ratio {}), will use closing filter instead"
                      .format(size_ratio))
                smoothed = morphology.binary_closing(label_mask_region, selem)
                region_size_smoothed = np.sum(smoothed)
            
            # fill empty spaces with closest surrounding labels
            region = plot_3d.in_paint(region, label_mask_region)
            
        elif mode is config.SmoothingModes.gaussian:
            # smoothing with gaussian blur
            smoothed = filters.gaussian(
                label_mask_region, filter_size, mode="nearest", 
                multichannel=False).astype(bool)
            region_size_smoothed = np.sum(smoothed)
            
        elif mode is config.SmoothingModes.closing:
            # smooth region with closing filter
            smoothed = morphology.binary_closing(
                label_mask_region, morphology.ball(filter_size))
            region_size_smoothed = np.sum(smoothed)
            
            # fill empty spaces with closest surrounding labels
            region = plot_3d.in_paint(region, label_mask_region)
        
        # replace smoothed volume within in-painted region
        region[smoothed] = label_id
        labels_img_np[tuple(slices)] = region
        print("changed num of pixels from {} to {}"
              .format(region_size, region_size_smoothed))
    
    # show label loss metric
    print()
    label_ids_smoothed = np.unique(labels_img_np)
    labels_lost(label_ids, label_ids_smoothed, 
        label_img_np_orig=labels_img_np_orig)
    
    # show DSC for labels
    print("\nMeasuring overlap of labels:")
    measure_overlap_labels(
        sitk.GetImageFromArray(labels_img_np_orig), 
        sitk.GetImageFromArray(labels_img_np))
    
    # weighted pixel ratio metric of volume change
    weighted_size_ratio = 0
    tot_pxs = 0
    for label_id in label_ids_ordered:
        # skip backgroud since not a "region"
        if label_id == 0: continue
        size_orig = np.sum(labels_img_np_orig == label_id)
        size_smoothed = np.sum(labels_img_np == label_id)
        weighted_size_ratio += size_smoothed
        tot_pxs += size_orig
    weighted_size_ratio /= tot_pxs
    print("\nVolume ratio (smoothed:orig) weighted by orig size: {}\n"
          .format(weighted_size_ratio))

def label_smoothing_metric(orig_img_np, smoothed_img_np, filter_size=None, 
                           penalty_wt=None, mode=SMOOTHING_METRIC_MODES[4], 
                           save_borders=False):
    """Measure degree of appropriate smoothing, defined as smoothing that 
    retains the general shape and placement of the region.
    
    Several methods are available as metrices:
    
    ``vol``: Compare the difference in size of a broad, smooth volume 
    encompassing ragged edges before and after the smoothing algorithm, giving 
    a measure of reduction in size and thus increase in compactness by 
    smoothing. To penalize inappropriate smoothing, the smoothed volume 
    lying outside of the original broad volume is subtracted from this 
    size reduction, accounting for unwanted deformation that brings the 
    region outside of its original bounds.
    
    ``area``: Compare surface areas of each label before and after 
    smoothing. Compaction is taken as a reduction in overall surface area, 
    while the displacement penalty is measured by the distance transform 
    of the smoothed border from the original border.
    
    ``compactness``: Similar to ``area``, but use the formal compactness 
    measurement.
    
    Args:
        original_img_np: Unsmoothed labels image as Numpy array.
        smoothing_img_np: Smoothed labels image as Numpy array, which 
            should be of the same shape as ``original_img_np``.
        filter_size: Structuring element size for determining the filled, 
            broad volume of each label. Defaults to None. Larger sizes 
            favor greater smoothing in the final labels.
        penalty_wt: Weighting factor for the penalty term. For ``vol`` 
            mode, larger  values favor labels that remain within their 
            original bounds. For ``area`` mode, this value is used as a 
            denominator for pixel perimeter displacement, where larger values 
            tolerate more displacement. Defaults to None.
        mode: One of :const:``SMOOTHING_METRIC_MODES`` (see above for 
            description of the modes). Defaults to 
            :const:``SMOOTHING_METRIC_MODES[4]``
        save_borders: True to save borders of original and smoothed images 
            in a separate, multichannel image; defaults to False.
    
    Returns:
        Tuple of a borders image, a Numpy array of the same same as 
        ``original_img_np`` except with an additional channel dimension at 
        the end, where channel 0 contains the broad borders of the 
        original image's labels, and channel 1 is that of the smoothed image, 
        or None if ``save_borders`` is False; a data frame of the 
        smoothing metrics; and another data frame of the raw metric 
        components.
    """
    start_time = time()
    
    # check parameters by mode
    if mode == SMOOTHING_METRIC_MODES[0]:
        if filter_size is None:
            raise TypeError(
                "filter size must be an integer, not {}, for mode {}"
                .format(filter_size, mode))
        if penalty_wt is None:
            penalty_wt = 1.0
            print("defaulting to penalty weight of {} for mode {}"
                  .format(penalty_wt, mode))
    elif not mode in SMOOTHING_METRIC_MODES:
        raise TypeError("no metric of mode {}".format(mode))
    print("Calculating smoothing metrics, mode {} with filter size of {}, "
          "penalty weighting factor of {}"
          .format(mode, filter_size, penalty_wt))
    
    # prepare roughness images to track global overlap
    shape = list(orig_img_np.shape)
    roughs = [np.zeros(shape, dtype=np.int8)]
    roughs.append(np.copy(roughs[0]))
    
    # prepare borders image with channel for each set of borders
    shape.append(2)
    borders_img_np = None
    if save_borders:
        borders_img_np = np.zeros(shape, dtype=np.int32)
    
    # pepare labels and default selem used to find "broad volume"
    label_ids = np.unique(orig_img_np)
    
    def update_borders_img(borders, slices, label_id, channel):
        nonlocal borders_img_np
        if borders_img_np is None: return
        borders_region = borders_img_np[tuple(slices)]
        borders_region[borders, channel] = label_id
    
    def broad_borders(img_np, slices, label_id, channel, rough_img_np):
        # use closing filter to approximate volume encompassing rough edges
        # get region, skipping if no region left
        region = img_np[tuple(slices)]
        label_mask_region = region == label_id
        filtered = morphology.binary_closing(label_mask_region, selem)
        rough_img_np[tuple(slices)] = np.add(
            rough_img_np[tuple(slices)], filtered.astype(np.int8))
        filtered_border = plot_3d.perimeter_nd(filtered)
        update_borders_img(filtered_border, slices, label_id, channel)
        return label_mask_region, filtered
    
    def surface_area(img_np, slices, label_id, rough_img_np):
        # use closing filter to approximate volume encompassing rough edges
        # get region, skipping if no region left
        region = img_np[tuple(slices)]
        label_mask_region = region == label_id
        borders = plot_3d.perimeter_nd(label_mask_region)
        rough_img_np[tuple(slices)] = np.add(
            rough_img_np[tuple(slices)], borders.astype(np.int8))
        return label_mask_region, borders
    
    padding = 2 if filter_size is None else 2 * filter_size
    pxs = {}
    cols = ("label_id", "pxs_reduced", "pxs_expanded", "size_orig")
    for label_id in label_ids:
        # calculate metric for each label
        if label_id == 0: continue
        print("finding border for {}".format(label_id))
        
        # use bounding box that fits around label in both original and 
        # smoothed image to improve efficiency over filtering whole image
        label_mask = np.logical_or(
            orig_img_np == label_id, smoothed_img_np == label_id)
        props = measure.regionprops(label_mask.astype(np.int))
        if len(props) < 1 or props[0].bbox is None: continue
        _, slices = plot_3d.get_bbox_region(
            props[0].bbox, padding, orig_img_np.shape)
        
        if mode == SMOOTHING_METRIC_MODES[0]:
            # "vol": measure wrapping volume by closing filter
            selem = morphology.ball(filter_size)
            mask_orig, broad_orig = broad_borders(
                orig_img_np, slices, label_id, 0, roughs[0])
            _, broad_smoothed = broad_borders(
                smoothed_img_np, slices, label_id, 1, roughs[1])
            # reduction in broad volumes (compaction)
            pxs_reduced = np.sum(broad_orig) - np.sum(broad_smoothed)
            # expansion past original broad volume (displacement penalty)
            pxs_expanded = (
                np.sum(np.logical_and(broad_smoothed, ~broad_orig)) 
                * penalty_wt)
            # normalize to total foreground
            size_orig = np.sum(mask_orig)
            if label_id != 0: tot_size += size_orig
        elif mode in SMOOTHING_METRIC_MODES[1:]:
            
            # measure surface area for SA:vol and to get vol mask
            mask_orig, borders_orig = surface_area(
                orig_img_np, slices, label_id, roughs[0])
            update_borders_img(borders_orig, slices, label_id, 0)
            mask_smoothed, borders_smoothed = surface_area(
                smoothed_img_np, slices, label_id, roughs[1])
            
            # compaction
            if mode in SMOOTHING_METRIC_MODES[1:4]:
                # "area"-based: reduction in surface area (compaction), 
                # normalized to orig SA
                pxs_reduced = np.sum(borders_orig) - np.sum(borders_smoothed)
                size_orig = np.sum(borders_orig)
            elif mode == SMOOTHING_METRIC_MODES[4]:
                # "compactness": reduction in compactness, multiplied by 
                # orig vol for wt avg
                size_orig = np.sum(mask_orig)
                compactness_orig = plot_3d.compactness(borders_orig, mask_orig)
                compactness_smoothed = plot_3d.compactness(
                    borders_smoothed, mask_smoothed)
                pxs.setdefault("compactness_orig", []).append(compactness_orig)
                pxs.setdefault("compactness_smoothed", []).append(
                    compactness_smoothed)
                pxs_reduced = ((compactness_orig - compactness_smoothed) 
                               / compactness_orig)
            
            # displacement
            if mode in SMOOTHING_METRIC_MODES[3:]:
                # measure displacement by simple vol expansion
                displ = np.sum(np.logical_and(mask_smoothed, ~mask_orig))
            else:
                # signed distance between borders; option to use filter for 
                # buffer around irregular borders to offset distances there
                dist_to_orig, indices, borders_orig_filled = (
                    plot_3d.borders_distance(
                        borders_orig, borders_smoothed, mask_orig=mask_orig, 
                        filter_size=filter_size, gaus_sigma=penalty_wt))
                if mode == SMOOTHING_METRIC_MODES[1]:
                    # "area_edt": direct distances between borders
                    if filter_size is not None:
                        update_borders_img(
                            borders_orig_filled, slices, label_id, 1)
                    dist_to_orig[dist_to_orig < 0] = 0 # only expansion
                elif mode == SMOOTHING_METRIC_MODES[2]:
                    # "area_radial": radial distances from center to get 
                    # signed distances (DEPRECATED: use signed dist in area_edt)
                    region = orig_img_np[tuple(slices)]
                    props = measure.regionprops(
                        (region == label_id).astype(np.int))
                    centroid = props[0].centroid
                    radial_dist_orig = plot_3d.radial_dist(
                        borders_orig, centroid)
                    radial_dist_smoothed = plot_3d.radial_dist(
                        borders_smoothed, centroid)
                    radial_diff = plot_3d.radial_dist_diff(
                        radial_dist_orig, radial_dist_smoothed, indices)
                    dist_to_orig = np.abs(radial_diff)
                # SA weighted by distance is essentially the integral of the 
                # SA, so this sum can be treated as a vol
                displ = np.sum(dist_to_orig)
            
            pxs.setdefault("displacement", []).append(displ)
            if mode == SMOOTHING_METRIC_MODES[1:4]:
                # normalize weighted displacement by tot vol for a unitless 
                # fraction; will later multiply by orig SA to bring back 
                # to compaction units
                pxs_expanded = displ / np.sum(mask_orig)
            elif mode == SMOOTHING_METRIC_MODES[4]:
                # fraction of displaced volume
                pxs_expanded = displ / size_orig
            
            # SA:vol metrics, including ratio of SA:vol ratios
            sa_to_vol_orig = np.sum(borders_orig) / np.sum(mask_orig)
            vol_smoothed = np.sum(mask_smoothed)
            sa_to_vol_smoothed = 0
            if vol_smoothed > 0:
                sa_to_vol_smoothed = np.sum(borders_smoothed) / vol_smoothed
            sa_to_vol_ratio = sa_to_vol_smoothed / sa_to_vol_orig
            pxs.setdefault("SA_to_vol_orig", []).append(sa_to_vol_orig)
            pxs.setdefault("SA_to_vol_smoothed", []).append(sa_to_vol_smoothed)
            pxs.setdefault("SA_to_vol_ratio", []).append(sa_to_vol_ratio)
        
        vals = (label_id, pxs_reduced, pxs_expanded, size_orig)
        for col, val in zip(cols, vals):
            pxs.setdefault(col, []).append(val)
        print("pxs_reduced: {}, pxs_expanded: {}, smoothing quality: {}"
              .format(pxs_reduced, pxs_expanded, pxs_reduced - pxs_expanded))
    
    totals = {}
    for key in pxs.keys():
        if key == "label_id": continue
        vals = pxs[key]
        if key != "size_orig": vals = np.multiply(vals, pxs["size_orig"])
        totals[key] = np.nansum(vals)
    
    metrics = dict.fromkeys(SmoothingMetrics, np.nan)
    tot_size = totals["size_orig"]
    if tot_size > 0:
        frac_reduced =  totals["pxs_reduced"] / tot_size
        frac_expanded =  totals["pxs_expanded"] / tot_size
        metrics[SmoothingMetrics.COMPACTED] = [frac_reduced]
        metrics[SmoothingMetrics.DISPLACED] = [frac_expanded]
        metrics[SmoothingMetrics.SM_QUALITY] = [frac_reduced - frac_expanded]
        metrics[SmoothingMetrics.COMPACTNESS] = [
            totals["compactness_smoothed"] / tot_size]
        metrics[SmoothingMetrics.DISPLACEMENT] = [
            totals["displacement"] / tot_size]
        if mode == SMOOTHING_METRIC_MODES[0]:
            # find only amount of overlap, subtracting label count itself
            roughs = [rough - 1 for rough in roughs]
        roughs_metric = [np.sum(rough) / tot_size for rough in roughs]
        metrics[SmoothingMetrics.SA_VOL_ABS] = [
            totals["SA_to_vol_smoothed"] / tot_size]
        metrics[SmoothingMetrics.SA_VOL] = [
            totals["SA_to_vol_ratio"] / tot_size]
        metrics[SmoothingMetrics.ROUGHNESS] = [roughs_metric[0]]
        metrics[SmoothingMetrics.ROUGHNESS_SM] = [roughs_metric[1]]
        num_labels_orig = len(label_ids)
        metrics[SmoothingMetrics.LABEL_LOSS] = [
            (num_labels_orig - len(np.unique(smoothed_img_np))) 
            / num_labels_orig]
    # raw stats
    print()
    df_pxs = pd.DataFrame(pxs)
    print(df_pxs.to_csv(sep="\t", index=False))
    print("\nTotal foreground pxs: {}".format(tot_size))
    
    # data frame just for aligned printing but return metrics dict for 
    # concatenating multiple runs
    df_metrics = stats.dict_to_data_frame(metrics, show="\t")
    print("\ntime elapsed for smoothing metric (s): {}"
          .format(time() - start_time))
    return borders_img_np, df_metrics, df_pxs

def smoothing_peak(df, thresh_label_loss=None, filter_size=None):
    """Extract the row of peak smoothing quality from the given data 
    frame matching the given criteria.
    
    Args:
        df: Data frame from which to extract.
        thresh_label_loss: Only check rows below or equal to this 
            fraction of label loss; defaults to None to ignore.
        filter_size: Only rows with the given filter size; defaults 
            to None to ignore.
    
    Returns:
        New data frame with the row having the peak smoothing quality 
        meeting criteria.
    """
    if thresh_label_loss is not None:
        df = df.loc[df[SmoothingMetrics.LABEL_LOSS.value] <= thresh_label_loss]
    if filter_size is not None:
        df = df.loc[df[SmoothingMetrics.FILTER_SIZE.value] == filter_size]
    sm_qual = df[SmoothingMetrics.SM_QUALITY.value]
    df_peak = df.loc[sm_qual == sm_qual.max()]
    return df_peak

def transpose_img(img_sitk, plane, rotate=None, target_size=None, 
                  flipud=False):
    """Transpose a SimpleITK format image via Numpy and re-export to SimpleITK.
    
    Args:
        img_sitk: Image in SimpleITK format.
        plane: One of :attr:``config.PLANES`` elements, specifying the 
            planar orientation in which to transpose the image. The current 
            orientation is taken to be "xy".
        rotate: Number of times to rotate by 90 degrees; defaults to None.
        target_size: Size of target image, typically one to which ``img_sitk`` 
            will be registered, in (x,y,z, SimpleITK standard) ordering.
        flipud: Flip along the final z-axis; defaults to False.
    
    Returns:
        Transposed image in SimpleITK format.
    """
    img = sitk.GetArrayFromImage(img_sitk)
    img_dtype = img.dtype
    spacing = img_sitk.GetSpacing()[::-1]
    origin = img_sitk.GetOrigin()[::-1]
    transposed = img
    if plane is not None and plane != config.PLANE[0]:
        # transpose planes and metadata
        arrs_3d, arrs_1d = plot_support.transpose_images(
            plane, [transposed], [spacing, origin])
        transposed = arrs_3d[0]
        spacing, origin = arrs_1d
        if flipud: transposed = np.flipud(transposed)
    if rotate:
        # rotate the final output image by 90 deg
        # TODO: need to change origin? make axes accessible (eg (0, 2) for 
        # horizontal rotation)
        transposed = np.rot90(transposed, rotate, (1, 2))
        if rotate % 2 != 0:
            spacing = lib_clrbrain.swap_elements(spacing, 1, 2)
            origin = lib_clrbrain.swap_elements(origin, 1, 2)
    resize_factor = config.register_settings["resize_factor"]
    if target_size is not None and resize_factor:
        # rescale based on xy dimensions of given and target image so that
        # they are not so far off from one another that scaling does not occur; 
        # assume that size discrepancies in z don't affect registration and 
        # for some reason may even prevent registration
        size_diff = np.divide(target_size[::-1][1:3], transposed.shape[1:3])
        rescale = np.mean(size_diff) * resize_factor
        if rescale > 0.5:
            print("rescaling image by {}x after applying resize factor of {}"
                  .format(rescale, resize_factor))
            transposed = transform.rescale(
                transposed, rescale, mode="constant", preserve_range=True, 
                multichannel=False, anti_aliasing=False, 
                order=0).astype(img_dtype)
            spacing = np.divide(spacing, rescale)
        # casted back since transpose changes data type even when 
        # preserving range
        print(transposed.dtype, np.min(transposed), np.max(transposed))
    transposed = sitk.GetImageFromArray(transposed)
    transposed.SetSpacing(spacing[::-1])
    transposed.SetOrigin(origin[::-1])
    return transposed

def _load_numpy_to_sitk(numpy_file, rotate=False):
    """Load Numpy image array to SimpleITK Image object.
    
    Args:
        numpy_file: Path to Numpy archive file.
        rotate: True if the image should be rotated 180 deg; defaults to False.
    
    Returns:
        The image in SimpleITK format.
    """
    image5d = importer.read_file(numpy_file, config.series)
    roi = image5d[0, ...] # not using time dimension
    if rotate:
        roi = np.rot90(roi, 2, (1, 2))
    sitk_img = sitk.GetImageFromArray(roi)
    spacing = detector.resolutions[0]
    sitk_img.SetSpacing(spacing[::-1])
    # TODO: consider setting z-origin to 0 since image generally as 
    # tightly bound to subject as possible
    #sitk_img.SetOrigin([0, 0, 0])
    sitk_img.SetOrigin([0, 0, -roi.shape[0] // 2])
    #sitk_img.SetOrigin([0, 0, -roi.shape[0]])
    return sitk_img

def _curate_img(fixed_img, labels_img, imgs=None, inpaint=True, carve=True, 
                holes_area=None):
    """Curate images by in-painting where corresponding pixels are present in 
    fixed image but not labels or other images and removing pixels 
    present in those images but not the fixed image.
    
    Args:
        fixed_img: Image in SimpleITK format by which to curate other images.
        labels_img: Labels image in SimpleITK format, used to determine 
            missing pixels and measure overlap.
        imgs: Array of additional images to curate corresponding pixels 
            as those curated in ``labels_img``. Defaults to None.
        inpaint: True to in-paint; defaults to True.
        carve: True to carve based on ``fixed_img``; defaults to True. If 
            ``inpaint`` is True, ``carve`` should typically be True as well.
        holes_area: Maximum area of holes to fill when carving.
    
    Returns:
        A list of images in SimpleITK format that have been curated.
    """
    fixed_img_np = sitk.GetArrayFromImage(fixed_img)
    labels_img_np = sitk.GetArrayFromImage(labels_img)
    result_img = labels_img
    result_img_np = labels_img_np
    # ensure that labels image is first
    if imgs:
        imgs.insert(0, labels_img)
    else:
        imgs = [labels_img]
    
    # mask image showing where result is 0 but fixed image is above thresh 
    # to fill in with nearest neighbors
    thresh = filters.threshold_mean(fixed_img_np)
    print("thresh: {}".format(thresh))
    # fill only empty regions corresponding to filled pixels, but fills 
    # some with 0 from dist transform pointing to appropriately empty pixels
    #to_fill = np.logical_and(labels_img_np == 0, fixed_img_np > thresh)
    to_fill = labels_img_np == 0
    
    result_imgs = []
    for i in range(len(imgs)):
        # in-paint and remove pixels from result image where fixed image is 
        # below threshold
        img = imgs[i]
        result_img_np = sitk.GetArrayFromImage(img)
        if inpaint:
            result_img_np = plot_3d.in_paint(result_img_np, to_fill)
        if carve:
            _, mask = plot_3d.carve(fixed_img_np, thresh, holes_area)
            result_img_np[~mask] = 0
        result_img = replace_sitk_with_numpy(img, result_img_np)
        result_imgs.append(result_img)
        if i == 0:
            # check overlap based on labels images; should be 1.0 by def
            result_img_np[result_img_np != 0] = 2
            result_img_for_overlap = replace_sitk_with_numpy(
                img, result_img_np)
            measure_overlap(
                fixed_img, result_img_for_overlap, transformed_thresh=1)
    return result_imgs

def _transform_labels(transformix_img_filter, labels_img, truncation=None, 
                      flip=False):
    if truncation is not None:
        # truncate ventral and posterior portions since variable 
        # amount of tissue or quality of imaging in these regions
        labels_img_np = sitk.GetArrayFromImage(labels_img)
        truncation = list(truncation)
        if flip:
            # assume labels were rotated 180deg around the z-axis, so 
            # need to flip y-axis fracs
            truncation[1] = np.subtract(1, truncation[1])[::-1]
        _truncate_labels(labels_img_np, *truncation)
        labels_img = replace_sitk_with_numpy(labels_img, labels_img_np)
    
    # apply atlas transformation to labels image
    labels_pixel_id = labels_img.GetPixelID() # now as signed int
    print("labels_pixel type: {}".format(labels_img.GetPixelIDTypeAsString()))
    transformix_img_filter.SetMovingImage(labels_img)
    transformix_img_filter.Execute()
    transformed_labels_img = transformix_img_filter.GetResultImage()
    transformed_labels_img = sitk.Cast(transformed_labels_img, labels_pixel_id)
    '''
    # remove bottom planes after registration; make sure to turn off 
    # bottom plane removal in mirror step
    img_np = sitk.GetArrayFromImage(labels_img)
    _truncate_labels(img_np, z_frac=(0.2, 1.0))
    labels_img = replace_sitk_with_numpy(labels_img, img_np)
    '''
    print(transformed_labels_img)
    '''
    LabelStatistics = sitk.LabelStatisticsImageFilter()
    LabelStatistics.Execute(fixed_img, labels_img)
    count = LabelStatistics.GetCount(1)
    mean = LabelStatistics.GetMean(1)
    variance = LabelStatistics.GetVariance(1)
    print("count: {}, mean: {}, variance: {}".format(count, mean, variance))
    '''
    return transformed_labels_img

def _config_reg_resolutions(grid_spacing_schedule, param_map, ndim):
    if grid_spacing_schedule:
        # assume spacing given as single val for all dimensions
        param_map["GridSpacingSchedule"] = grid_spacing_schedule
        num_res = len(grid_spacing_schedule)
        res_set = grid_spacing_schedule[:ndim]
        if len(np.unique(res_set)) != ndim:
            # any repeated resolutions must mean that each value refers to 
            # a dimension rather than to all dimensions; note that schedules 
            # with completely different values per dimension won't be 
            # identified correctly
            num_res /= ndim
        param_map["NumberOfResolutions"] = [str(num_res)]

def match_atlas_labels(img_atlas, img_labels, flip=False):
    """Apply register profile settings to labels and match atlas image 
    accordingly.
    
    Args:
        img_labels: Labels image as SimpleITK image.
        img_ref: Reference image as SimpleITK image.
        flip: True to rotate images 180deg around the final z axis; 
            defaults to False.
    
    Returns:
        Tuple of ``img_atlas``, the updated atlas; ``img_labels``, the 
        updated labels; ``img_borders``, a new SimpleITK image of the 
        same shape as the prior images except an extra channels dimension 
        as given by :func:``_curate_labels``; and ``df_smoothing``, a 
        data frame of smoothing stats, or None if smoothing was not performed.
    """
    pre_plane = config.register_settings["pre_plane"]
    extend_atlas = config.register_settings["extend_atlas"]
    mirror = config.register_settings["labels_mirror"]
    edge = config.register_settings["labels_edge"]
    expand = config.register_settings["expand_labels"]
    rotate = config.register_settings["rotate"]
    smooth = config.register_settings["smooth"]
    crop = config.register_settings["crop_to_labels"]
    affine = config.register_settings["affine"]
    
    if pre_plane:
        # images in the correct desired orientation may need to be 
        # transposed prior to label curation since mirroring assumes 
        # a sagittal orientation
        img_atlas_np = sitk.GetArrayFromImage(img_atlas)
        img_labels_np = sitk.GetArrayFromImage(img_labels)
        arrs_3d, _ = plot_support.transpose_images(
            pre_plane, [img_atlas_np, img_labels_np])
        img_atlas_np = arrs_3d[0]
        img_labels_np = arrs_3d[1]
        img_atlas = replace_sitk_with_numpy(img_atlas, img_atlas_np)
        img_labels = replace_sitk_with_numpy(img_labels, img_labels_np)
    
    # curate labels
    mask_lbls = None
    if extend_atlas:
        # include any lateral extension and mirroing
        img_labels_np, mirror_indices, borders_img_np, df_smoothing = (
            _curate_labels(
                img_labels, img_atlas, mirror, edge, expand, rotate, smooth, 
                affine))
    else:
        # turn off lateral extension and mirroing
        img_labels_np, _, borders_img_np, df_smoothing = _curate_labels(
            img_labels, img_atlas, None, None, expand, rotate, smooth, 
            affine)
        if crop and (mirror is not None or edge is not None):
            # separately get mirrored labels only for cropping to labels
            print("\nCurating labels with extension/mirroring only "
                  "for cropping:")
            lbls_np_mir, mirror_indices, _, _ = _curate_labels(
                img_labels, img_atlas, mirror, edge, expand, rotate, None, 
                affine, False)
            mask_lbls = lbls_np_mir != 0
            print()
    
    # adjust atlas with same settings
    img_atlas_np = sitk.GetArrayFromImage(img_atlas)
    if rotate:
        for rot in rotate:
            img_atlas_np = plot_3d.rotate_nd(img_atlas_np, rot[0], rot[1])
    if affine:
        for aff in affine:
            img_atlas_np = plot_3d.affine_nd(img_atlas_np, **aff)
    if extend_atlas and mirror is not None:
        # TODO: consider removing dup since not using
        dup = config.register_settings["labels_dup"]
        img_atlas_np = _mirror_planes(
            img_atlas_np, mirror_indices[1], start_dup=dup)
    
    if crop:
        # crop atlas to the mask of the labels with some padding; 
        # TODO: crop or deprecate borders image
        img_labels_np, img_atlas_np = plot_3d.crop_to_labels(
            img_labels_np, img_atlas_np, mask_lbls)
    
    imgs_np = (img_atlas_np, img_labels_np, borders_img_np)
    if pre_plane:
        # transpose back to original orientation
        imgs_np, _ = plot_support.transpose_images(
            pre_plane, imgs_np, rev=True)
    
    # convert back to sitk img and transpose if necessary
    img_borders = None if borders_img_np is None else img_labels
    imgs_sitk = (img_atlas, img_labels, img_borders)
    imgs_sitk_replaced = []
    for img_np, img_sitk in zip(imgs_np, imgs_sitk):
        if img_np is not None:
            img_sitk = replace_sitk_with_numpy(img_sitk, img_np)
            if pre_plane is None:
                # plane settings is for post-processing; 
                # TODO: check if 90deg rot is nec for yz
                rotate = 1 if config.plane in config.PLANE[1:] else 0
                if flip: rotate += 2
                img_sitk = transpose_img(
                    img_sitk, config.plane, rotate, flipud=True)
        imgs_sitk_replaced.append(img_sitk)
    img_atlas, img_labels, img_borders = imgs_sitk_replaced
    
    return img_atlas, img_labels, img_borders, df_smoothing

def import_atlas(atlas_dir, show=True):
    """Import atlas from the given directory, processing it according 
    to the register settings specified at :attr:``config.register_settings``.
    
    The imported atlas will be saved to a directory of the same path as 
    ``atlas_dir`` except with ``_import`` appended to the end. DSC 
    will be calculated and saved as a CSV file in this directory as well.
    
    Args:
        atlas_dir: Path to atlas directory.
        show: True to show the imported atlas.
    """
    # load atlas and corresponding labels
    img_atlas, path_atlas = read_sitk(os.path.join(atlas_dir, IMG_ATLAS))
    img_labels, _ = read_sitk(os.path.join(atlas_dir, IMG_LABELS))
    orig = "_raw" in config.register_settings["settings_name"]
    overlap_meas_add = config.register_settings["overlap_meas_add_lbls"]
    if orig:
        # baseline DSC of atlas to labels before any processing
        cond = "original"  
        dsc = _measure_overlap_combined_labels(
            img_atlas, img_labels, overlap_meas_add)
    else:
        # defer DSC until after processing
        cond = "extended" 
    
    # match atlas and labels to one another
    img_atlas, img_labels, img_borders, df_smoothing = match_atlas_labels(
        img_atlas, img_labels)
    
    truncate = config.register_settings["truncate_labels"]
    if truncate:
        # truncate labels
        img_labels_np = _truncate_labels(
            sitk.GetArrayFromImage(img_labels), *truncate)
        img_labels = replace_sitk_with_numpy(img_labels, img_labels_np)
    
    # show labels
    img_labels_np = sitk.GetArrayFromImage(img_labels)
    label_ids = np.unique(img_labels_np)
    print("number of labels: {}".format(label_ids.size))
    print(label_ids)
    
    # prep export paths
    target_dir = atlas_dir + "_import"
    basename = os.path.basename(atlas_dir)
    df_base_path = os.path.join(target_dir, basename) + "_{}"
    df_metrics_path = df_base_path.format(config.PATH_ATLAS_IMPORT_METRICS)
    name_prefix = os.path.join(target_dir, basename) + ".czi"
    
    # whole atlas stats
    if not orig:
        # measure DSC if processed and prep dict for data frame
        dsc = _measure_overlap_combined_labels(
            img_atlas, img_labels, overlap_meas_add)
    # use lower threshold for compactness measurement to minimize noisy 
    # surface artifacts
    img_atlas_np = sitk.GetArrayFromImage(img_atlas)
    thresh = config.register_settings["atlas_threshold_all"]
    thresh_atlas = img_atlas_np > thresh
    compactness = plot_3d.compactness(
        plot_3d.perimeter_nd(thresh_atlas), thresh_atlas)
    metrics = {
        AtlasMetrics.SAMPLE: [basename], 
        AtlasMetrics.REGION: config.REGION_ALL, 
        AtlasMetrics.CONDITION: cond, 
        AtlasMetrics.DSC_ATLAS_LABELS: [dsc], 
        SmoothingMetrics.COMPACTNESS: [compactness]
    }
    
    # write images with atlas saved as Clrbrain/Numpy format to 
    # allow opening as an image within Clrbrain alongside the labels image
    imgs_write = {
        IMG_ATLAS: img_atlas, IMG_LABELS: img_labels, IMG_BORDERS: img_borders}
    write_reg_images(
        imgs_write, name_prefix, copy_to_suffix=True, 
        ext=os.path.splitext(path_atlas)[1])
    detector.resolutions = [img_atlas.GetSpacing()[::-1]]
    img_ref_np = sitk.GetArrayFromImage(img_atlas)
    img_ref_np = img_ref_np[None]
    importer.save_np_image(img_ref_np, name_prefix, 0)
    
    if df_smoothing is not None:
        # write smoothing metrics to CSV with identifier columns
        df_smoothing_path = df_base_path.format(config.PATH_SMOOTHING_METRICS)
        df_smoothing[AtlasMetrics.SAMPLE.value] = basename
        df_smoothing[AtlasMetrics.REGION.value] = config.REGION_ALL
        df_smoothing[AtlasMetrics.CONDITION.value] = "smoothed"
        df_smoothing.loc[
            df_smoothing[SmoothingMetrics.FILTER_SIZE.value] == 0, 
            AtlasMetrics.CONDITION.value] = "unsmoothed"
        stats.data_frames_to_csv(
            df_smoothing, df_smoothing_path, 
            sort_cols=SmoothingMetrics.FILTER_SIZE.value)

    print("\nImported {} whole atlas stats:".format(basename))
    stats.dict_to_data_frame(metrics, df_metrics_path, show="\t")
    
    if show:
       sitk.Show(img_atlas)
       sitk.Show(img_labels)
    
def register_duo(fixed_img, moving_img):
    """Register two images to one another using ``SimpleElastix``.
    
    Args:
        fixed_img: The image to be registered to in :class``SimpleITK.Image`` 
            format.
        moving_img: The image to register to ``fixed_img`` in 
            :class``SimpleITK.Image`` format.
    
    Returns:
        Tuple of the registered image in SimpleITK Image format and a 
        Transformix filter with the registration's parameters to 
        reapply them on other images.
    """
    # basic info from images just prior to SimpleElastix filtering for 
    # registration; to view raw images, show these images rather than merely 
    # turning all iterations to 0 since simply running through the filter 
    # will alter images
    print("fixed image (type {}):\n{}".format(
        fixed_img.GetPixelIDTypeAsString(), fixed_img))
    print("moving image (type {}):\n{}".format(
        moving_img.GetPixelIDTypeAsString(), moving_img))
    
    # set up SimpleElastix filter
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_img)
    elastix_img_filter.SetMovingImage(moving_img)
    
    settings = config.register_settings
    param_map_vector = sitk.VectorOfParameterMap()
    # translation to shift and rotate
    param_map = sitk.GetDefaultParameterMap("translation")
    param_map["MaximumNumberOfIterations"] = [settings["translation_iter_max"]]
    '''
    # TESTING: minimal registration
    param_map["MaximumNumberOfIterations"] = ["0"]
    '''
    param_map_vector.append(param_map)
    # affine to sheer and scale
    param_map = sitk.GetDefaultParameterMap("affine")
    param_map["MaximumNumberOfIterations"] = [settings["affine_iter_max"]]
    param_map_vector.append(param_map)
    # bspline for non-rigid deformation
    param_map = sitk.GetDefaultParameterMap("bspline")
    param_map["FinalGridSpacingInVoxels"] = [
        settings["bspline_grid_space_voxels"]]
    del param_map["FinalGridSpacingInPhysicalUnits"] # avoid conflict with vox
    param_map["MaximumNumberOfIterations"] = [settings["bspline_iter_max"]]
    _config_reg_resolutions(
        settings["grid_spacing_schedule"], param_map, fixed_img.GetDimension())
    if settings["point_based"]:
        # point-based registration added to b-spline, which takes point sets 
        # found in name_prefix's folder; note that coordinates are from the 
        # originally set fixed and moving images, not after transformation up 
        # to this point
        fix_pts_path = os.path.join(os.path.dirname(name_prefix), "fix_pts.txt")
        move_pts_path = os.path.join(
            os.path.dirname(name_prefix), "mov_pts.txt")
        if os.path.isfile(fix_pts_path) and os.path.isfile(move_pts_path):
            metric = list(param_map["Metric"])
            metric.append("CorrespondingPointsEuclideanDistanceMetric")
            param_map["Metric"] = metric
            #param_map["Metric2Weight"] = ["0.5"]
            elastix_img_filter.SetFixedPointSetFileName(fix_pts_path)
            elastix_img_filter.SetMovingPointSetFileName(move_pts_path)
    
    param_map_vector.append(param_map)
    elastix_img_filter.SetParameterMap(param_map_vector)
    elastix_img_filter.PrintParameterMap()
    transform = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    
    # prep filter to apply transformation to label files
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    
    return transformed_img, transformix_img_filter

def register(fixed_file, moving_file_dir, plane=None, flip=False, 
             show_imgs=True, write_imgs=True, name_prefix=None, 
             new_atlas=False):
    """Register an atlas and associated labels to a sample image 
    using the SimpleElastix library.
    
    Args:
        fixed_file: The image to register, given as a Numpy archive file to 
            be read by :importer:`read_file`.
        moving_file_dir: Directory of the atlas images, including the 
            main image and labels. The atlas was chosen as the moving file
            since it is likely to be lower resolution than the Numpy file.
        plane: Planar orientation to which the atlas will be transposed, 
            considering the atlas' original plane as "xy". Defaults to 
            None to avoid planar transposition.
        flip: True if the moving files (does not apply to fixed file) should 
            be flipped/rotated; defaults to False.
        show_imgs: True if the output images should be displayed; defaults to 
            True.
        write_imgs: True if the images should be written to file; defaults to 
            False.
        name_prefix: Path with base name where registered files will be output; 
            defaults to None, in which case the fixed_file path will be used.
        new_atlas: True to generate registered images that will serve as a 
            new atlas; defaults to False.
    """
    start_time = time()
    if name_prefix is None:
        name_prefix = fixed_file
    settings = config.register_settings
    
    # load fixed image, assumed to be experimental image
    fixed_img = _load_numpy_to_sitk(fixed_file)
    
    # preprocessing; store original fixed image for overlap measure
    fixed_img_orig = fixed_img
    if settings["preprocess"]:
        img_np = sitk.GetArrayFromImage(fixed_img)
        #img_np = plot_3d.saturate_roi(img_np)
        img_np = plot_3d.denoise_roi(img_np)
        fixed_img = replace_sitk_with_numpy(fixed_img, img_np)
    fixed_img_size = fixed_img.GetSize()
    
    # load moving image, assumed to be atlas
    moving_file = os.path.join(moving_file_dir, IMG_ATLAS)
    moving_img = sitk.ReadImage(moving_file)
    
    # load labels image and match with atlas
    labels_img = sitk.ReadImage(os.path.join(moving_file_dir, IMG_LABELS))
    moving_img, labels_img, _, _ = match_atlas_labels(
        moving_img, labels_img, flip)
    
    transformed_img, transformix_img_filter = register_duo(
        fixed_img, moving_img)
    # turn off to avoid overshooting the interpolation for the labels image 
    # (see Elastix manual section 4.3)
    transform_param_map = transformix_img_filter.GetTransformParameterMap()
    transform_param_map[-1]["FinalBSplineInterpolationOrder"] = ["0"]
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    
    # overlap stats comparing original and registered samples (eg histology)
    print("DSC of original and registered sample images")
    dsc_sample = measure_overlap(
        fixed_img_orig, transformed_img, 
        transformed_thresh=settings["atlas_threshold"])
    
    def make_labels(truncate):
        nonlocal transformed_img
        truncation = settings["truncate_labels"] if truncate else None
        img = _transform_labels(
            transformix_img_filter, labels_img, truncation=truncation, 
            flip=flip)
        print(img.GetSpacing())
        # WORKAROUND: labels img floating point vals may be more rounded 
        # than transformed moving img for some reason; assume transformed 
        # labels and moving image should match exactly, so replace labels 
        # with moving image's transformed spacing
        img.SetSpacing(transformed_img.GetSpacing())
        print(img.GetSpacing())
        print(fixed_img_orig.GetSpacing(), transformed_img.GetSpacing())
        dsc = None
        if settings["curate"]:
            img, transformed_img = _curate_img(
                fixed_img_orig, img, imgs=[transformed_img], inpaint=new_atlas)
            print("DSC of original and registered sample images after curation")
            dsc = measure_overlap(
                fixed_img_orig, transformed_img, 
                transformed_thresh=settings["atlas_threshold"])
        return img, dsc
    
    labels_img_full, dsc_sample_curated = make_labels(False)
    labels_img, _ = labels_img_full if new_atlas else make_labels(True)
    
    imgs_write = (fixed_img, transformed_img, labels_img_full, labels_img)
    if show_imgs:
        # show individual SimpleITK images in default viewer
        for img in imgs_write: sitk.Show(img)
    
    if write_imgs:
        # write atlas and labels files, transposed according to plane setting
        if new_atlas:
            imgs_names = (IMG_ATLAS, IMG_LABELS)
            imgs_write = [transformed_img, labels_img]
        else:
            imgs_names = (IMG_EXP, IMG_ATLAS, IMG_LABELS, IMG_LABELS_TRUNC)
        for i in range(len(imgs_write)):
            out_path = imgs_names[i]
            if new_atlas:
                out_path = os.path.join(os.path.dirname(name_prefix), out_path)
            else:
                out_path = _reg_out_path(name_prefix, out_path)
            print("writing {}".format(out_path))
            sitk.WriteImage(imgs_write[i], out_path, False)

    # show 2D overlay for registered images
    imgs = [
        sitk.GetArrayFromImage(fixed_img),
        sitk.GetArrayFromImage(moving_img), 
        sitk.GetArrayFromImage(transformed_img), 
        sitk.GetArrayFromImage(labels_img)]
    # save transform parameters and attempt to find the original position 
    # that corresponds to the final position that will be displayed
    _, translation = _handle_transform_file(name_prefix, transform_param_map)
    translation = _translation_adjust(
        moving_img, transformed_img, translation, flip=True)
    
    # compare original atlas with registered labels taken as a whole
    dsc_labels = _measure_overlap_combined_labels(
        fixed_img_orig, labels_img_full)
    
    # measure compactness of fixed image
    fixed_img_orig_np = sitk.GetArrayFromImage(fixed_img_orig)
    thresh_atlas = fixed_img_orig_np > filters.threshold_mean(fixed_img_orig_np)
    compactness = plot_3d.compactness(
        plot_3d.perimeter_nd(thresh_atlas), thresh_atlas)
    
    # save basic metrics in CSV file
    basename = lib_clrbrain.get_filename_without_ext(fixed_file)
    metrics = {
        AtlasMetrics.SAMPLE: [basename], 
        AtlasMetrics.REGION: config.REGION_ALL, 
        AtlasMetrics.CONDITION: [np.nan], 
        AtlasMetrics.DSC_ATLAS_SAMPLE: [dsc_sample], 
        AtlasMetrics.DSC_ATLAS_SAMPLE_CUR: [dsc_sample_curated], 
        AtlasMetrics.DSC_ATLAS_LABELS: [dsc_labels], 
        SmoothingMetrics.COMPACTNESS: [compactness]
    }
    df_path = lib_clrbrain.combine_paths(
        name_prefix, config.PATH_ATLAS_IMPORT_METRICS)
    print("\nImported {} whole atlas stats:".format(basename))
    stats.dict_to_data_frame(metrics, df_path, show="\t")
    
    # show overlays last since blocks until fig is closed
    #_show_overlays(imgs, translation, fixed_file, None)
    print("time elapsed for single registration (s): {}"
          .format(time() - start_time))

def register_reg(fixed_path, moving_path, reg_base=None, reg_names=None, 
                 plane=None, flip=False, prefix=None, suffix=None, show=True):
    """Using registered images including the unregistered copies of 
    the original image, register these images to another image.
    
    For example, registered images can be registered back to the atlas. 
    This method can also be used to move unregistered original images 
    that have simply been copied as ``IMG_EXP`` during registration. 
    This copy can be registered "back" to the atlas, reversing the 
    fixed/moving images in :meth:``register`` to move all experimental 
    images into the same space.
    
    Args:
        fixed_path: Path to he image to be registered to in 
            :class``SimpleITK.Image`` format.
        moving_path: Path to the image in :class``SimpleITK.Image`` format 
            to register to the image at ``fixed_path``.
        reg_base: Registration suffix to combine with ``moving_path``. 
            Defaults to None to use ``moving_path`` as-is, with 
            output name based on :const:``IMG_EXP``.
        reg_names: List of additional registration suffixes associated 
            with ``moving_path`` to be registered using the same 
            transformation. Defaults to None.
        plane: Planar orientation to which the atlas will be transposed, 
            considering the atlas' original plane as "xy". Defaults to 
            None to avoid planar transposition.
        flip: True if the moving files (does not apply to fixed file) should 
            be flipped/rotated; defaults to False.
        prefix: Base path to use for output; defaults to None to 
            use ``moving_path`` instead.
        suffix: String to combine with ``moving_path`` to load images; 
            defaults to None.
        show: True to show images after registration; defaults to True.
    """
    fixed_img = sitk.ReadImage(fixed_path)
    mod_path = moving_path
    if suffix is not None:
        # adjust image path to load with suffix
        mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
    if reg_base is None:
        # load the image directly from given path
        moving_img = sitk.ReadImage(mod_path)
    else:
        # treat the path as a base path to which a reg suffix will be combined
        moving_img = load_registered_img(
            mod_path, get_sitk=True, reg_name=reg_base)
    
    # register the images and apply the transformation to any 
    # additional images previously registered to the moving path
    rotate = 2 if flip else 0
    moving_img = transpose_img(moving_img, plane, rotate)
    transformed_img, transformix_img_filter = register_duo(
        fixed_img, moving_img)
    reg_imgs = [transformed_img]
    names = [IMG_EXP if reg_base is None else reg_base]
    if reg_names is not None:
        for reg_name in reg_names:
            img = load_registered_img(
                mod_path, get_sitk=True, reg_name=reg_name)
            img = transpose_img(img, plane, rotate)
            transformix_img_filter.SetMovingImage(img)
            transformix_img_filter.Execute()
            img_result = transformix_img_filter.GetResultImage()
            reg_imgs.append(img_result)
            names.append(reg_name)
    
    # use prefix as base output path if given and append distiguishing string 
    # to differentiate from original files
    output_base = lib_clrbrain.insert_before_ext(
        moving_path if prefix is None else prefix, REREG_SUFFIX, "_")
    if suffix is not None:
        output_base = lib_clrbrain.insert_before_ext(output_base, suffix)
    imgs_write = {}
    for name, img in zip(names, reg_imgs):
        # use the same reg suffixes, assuming that output_base will give a 
        # distinct name to avoid overwriting previously registered images
        imgs_write[name] = img
    write_reg_images(imgs_write, output_base)
    if show:
        for img in imgs_write.values(): sitk.Show(img)
    
def measure_overlap(fixed_img, transformed_img, fixed_thresh=None, 
                    transformed_thresh=None, add_fixed_mask=None):
    """Measure the Dice Similarity Coefficient (DSC) between two foreground 
    of two images.
    
    Args:
        fixed_img: Image as a SimpleITK ``Image`` object.
        transformed_img: Image as a SimpleITK ``Image`` object to compare.
        fixed_thresh: Threshold to determine the foreground of ``fixed_img``; 
            defaults to None to determine by a mean threshold.
        transformed_thresh: Threshold to determine the foreground of 
            ``transformed_img``; defaults to None to determine by a mean 
            threshold.
        add_fixed_mask: Boolean mask to add to fixed image, after 
            thresholding; defaults to None. Useful to treat as foreground 
            regions that would be thresholded as background but 
            included in labels.
    
    Returns:
        The DSC of the foreground of the two given images.
    """
    # upper threshold does not seem be set with max despite docs for 
    # sitk.BinaryThreshold, so need to set with max explicitly
    fixed_img_np = sitk.GetArrayFromImage(fixed_img)
    fixed_thresh_up = float(np.amax(fixed_img_np))
    transformed_img_np = sitk.GetArrayFromImage(transformed_img)
    transformed_thresh_up = float(np.amax(transformed_img_np))
    
    # use threshold mean if lower thresholds not given
    if not fixed_thresh:
        fixed_thresh = float(filters.threshold_mean(fixed_img_np))
    if not transformed_thresh:
        transformed_thresh = float(filters.threshold_mean(transformed_img_np))
    print("measuring overlap with thresholds of {} (fixed) and {} (transformed)"
          .format(fixed_thresh, transformed_thresh))
    
    # similar to simple binary thresholding via Numpy
    fixed_binary_img = sitk.BinaryThreshold(
        fixed_img, fixed_thresh, fixed_thresh_up)
    if add_fixed_mask is not None:
        # add mask to foreground of fixed image
        fixed_binary_np = sitk.GetArrayFromImage(fixed_binary_img)
        print(np.unique(fixed_binary_np), fixed_binary_np.dtype)
        fixed_binary_np[add_fixed_mask] = True
        fixed_binary_img = replace_sitk_with_numpy(
            fixed_binary_img, fixed_binary_np)
    transformed_binary_img = sitk.BinaryThreshold(
        transformed_img, transformed_thresh, transformed_thresh_up)
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(fixed_binary_img, transformed_binary_img)
    total_dsc = overlap_filter.GetDiceCoefficient()
    #sitk.Show(fixed_binary_img)
    #sitk.Show(transformed_binary_img)
    print("Foreground DSC: {}".format(total_dsc))
    return total_dsc

def measure_overlap_labels(fixed_img, transformed_img):
    """Measure the mean Dice Similarity Coefficient (DSC) between two 
    labeled images.
    
    Args:
        fixed_img: Image as a SimpleITK ``Image`` object.
        transformed_img: Image as a SimpleITK ``Image`` object to compare.
    
    Returns:
        The mean label-by-label DSC of the two given images.
    """
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(fixed_img, transformed_img)
    mean_region_dsc = overlap_filter.GetDiceCoefficient()
    print("Mean regional (label-by-label) DSC: {}".format(mean_region_dsc))
    return mean_region_dsc

def make_labels_fg(labels_sitk):
    """Make a labels foreground image.
    
    Args:
        labels_sitk: Labels image as a SimpleITK ``Image`` object, where 
            0 = background, and all other values are considered foreground.
    
    Returns:
        Labels foreground as a SimpleITK ``Image`` object.
    """
    fg_img = sitk.GetArrayFromImage(labels_sitk)
    fg_img[fg_img != 0] = 1
    fg_img_sitk = replace_sitk_with_numpy(labels_sitk, fg_img)
    return fg_img_sitk

def _measure_overlap_combined_labels(fixed_img, labels_img, add_lbls=None):
    # check overlap based on combined labels images; should be 1.0 by def 
    # when using labels img to curate fixed img
    lbls_fg = make_labels_fg(labels_img)
    mask = None
    if add_lbls is not None:
        # build mask from labels to add to fixed image's foreground, such 
        # as labeled ventricles; TODO: get children of labels rather than 
        # taking labels range, but would need to load labels reference; 
        # TODO: consider using "atlas_threshold_all" profile setting 
        # instead, but would need to ensure that fixed thresholds work 
        # for both atlas and sample histology
        labels_np_abs = np.absolute(sitk.GetArrayFromImage(labels_img))
        mask = np.zeros_like(labels_np_abs, dtype=bool)
        for lbl in add_lbls:
            print("adding abs labels within", lbl)
            mask[np.all([labels_np_abs >= lbl[0], 
                         labels_np_abs < lbl[1]], axis=0)] = True
    print("\nDSC of thresholded fixed image compared with combined labels:")
    return measure_overlap(
        fixed_img, lbls_fg, transformed_thresh=1, add_fixed_mask=mask)

def _crop_image(img_np, labels_img, axis, eraser=None):
    """Crop image by removing the empty space at the start of the given axis.
    
    Args:
        img_np: 3D image array in Numpy format.
        labels_img: 3D image array in Numpy format of the same shape as 
            ``img_np``, typically a labels image from which to find the empty 
            region to crop.
        axis: Axis along which to crop.
        eraser: Erase rather than crop, changing pixels that would have been 
            cropped to the given intensity value instead; defaults to None.
    
    Returns:
        Tuple of ``img_crop, i``, where ``img_crop is the cropped image with 
        planes removed along the start of the given axis until the first 
        non-empty plane is reached, or erased if ``eraser`` is given, and 
        ``i`` is the index of the first non-cropped/erased plane.
    """
    # find the first non-zero plane in the labels image along the given axis, 
    # expanding slices to the include the rest of the image; 
    # TODO: consider using mask from given region in labels to zero out 
    # corresponding region in img_np
    slices = [slice(None)] * labels_img.ndim
    shape = labels_img.shape
    for i in range(shape[axis]):
        slices[axis] = i
        plane = labels_img[tuple(slices)]
        if not np.allclose(plane, 0):
            print("found first non-zero plane at i of {}".format(i))
            break
    
    # crop image if a region of empty planes is found at the start of the axis
    img_crop = img_np
    if i < shape[axis]:
        slices = [slice(None)] * img_np.ndim
        if eraser is None:
            slices[axis] = slice(i, shape[axis])
            img_crop = img_crop[tuple(slices)]
            print("cropped image from shape {} to {}"
                  .format(shape, img_crop.shape))
        else:
            slices[axis] = slice(0, i)
            img_crop[tuple(slices)] = eraser
            print("erased image outside of {} to {} intensity value"
                  .format(slices, eraser))
    else:
        print("could not find non-empty plane at which to crop")
    return img_crop, i

def register_group(img_files, flip=None, show_imgs=True, 
             write_imgs=True, name_prefix=None, scale=None):
    """Group registers several images to one another.
    
    Args:
        img_files: Paths to image files to register.
        flip: Boolean list corresponding to ``img_files`` flagging 
            whether to flip the image or not; defaults to None, in which 
            case no images will be flipped.
        show_imgs: True if the output images should be displayed; defaults to 
            True.
        write_imgs: True if the images should be written to file; defaults to 
            True.
        name_prefix: Path with base name where registered files will be output; 
            defaults to None, in which case the fixed_file path will be used.
        scale: Rescaling factor as a scalar value, used to find the rescaled, 
            smaller images corresponding to ``img_files``. Defaults to None.
    """
    start_time = time()
    if name_prefix is None:
        name_prefix = img_files[0]
    target_size = config.register_settings["target_size"]
    
    '''
    # TESTING: assuming first file is a raw groupwise registered image, 
    # import it for post-processing
    img = sitk.ReadImage(img_files[0])
    img_np = sitk.GetArrayFromImage(img)
    print("thresh mean: {}".format(filters.threshold_mean(img_np)))
    carve_threshold = config.register_settings["carve_threshold"]
    holes_area = config.register_settings["holes_area"]
    img_np, img_np_unfilled = plot_3d.carve(
        img_np, thresh=carve_threshold, holes_area=holes_area, 
        return_unfilled=True)
    sitk.Show(replace_sitk_with_numpy(img, img_np_unfilled))
    sitk.Show(replace_sitk_with_numpy(img, img_np))
    return
    '''
    
    img_vector = sitk.VectorOfImage()
    flip_img = False
    # image properties of 1st image, in SimpleITK format
    origin = None
    size_orig = None
    size_cropped = None
    start_y = None
    spacing = None
    img_np_template = None # first image, used as template for rest
    for i in range(len(img_files)):
        # load image, fipping if necessary and using tranpsosed img if specified
        img_file = img_files[i]
        img_file = transformer.get_transposed_image_path(
            img_file, scale, target_size)
        if flip is not None:
            flip_img = flip[i]
        img = _load_numpy_to_sitk(img_file, flip_img)
        size = img.GetSize()
        img_np = sitk.GetArrayFromImage(img)
        if img_np_template is None:
            img_np_template = np.copy(img_np)
        
        # crop y-axis based on registered labels to ensure that sample images 
        # have the same structures since variable amount of tissue posteriorly; 
        # cropping appears to work better than erasing for groupwise reg, 
        # preventing some images from being stretched into the erased space
        labels_img = load_registered_img(
            img_files[i], reg_name=IMG_LABELS_TRUNC)
        img_np, y_cropped = _crop_image(img_np, labels_img, 1)#, eraser=0)
        '''
        # crop anterior region
        rotated = np.rot90(img_np, 2, (1, 2))
        rotated, _ = _crop_image(rotated, np.rot90(labels_img, 2, (1, 2)), 1)
        img_np = np.rot90(rotated, 2, (1, 2))
        '''
        
        # force all images into same size and origin as first image 
        # to avoid groupwise registration error on physical space mismatch
        if size_cropped is not None:
            # use default interpolation, but should change to nearest neighbor 
            # if using for labels
            img_np = transform.resize(
                img_np, size_cropped[::-1], anti_aliasing=True, mode="reflect")
            print(img_file, img_np.shape)
        img = replace_sitk_with_numpy(img, img_np)
        if origin is None:
            origin = img.GetOrigin()
            size_orig = size
            size_cropped = img.GetSize()
            spacing = img.GetSpacing()
            start_y = y_cropped
            print("size_cropped: ", size_cropped, ", size_orig", size_orig)
        else:
            # force images into space of first image; may not be exactly 
            # correct but should be close since resized to match first image, 
            # and spacing of resized images and atlases largely ignored in 
            # favor of comparing shapes of large original and registered images
            img.SetOrigin(origin)
            img.SetSpacing(spacing)
        print("img_file: {}\n{}".format(img_file, img))
        img_vector.push_back(img)
        #sitk.Show(img)
    #sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(1)
    #sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(100)
    img_combined = sitk.JoinSeries(img_vector)
    
    settings = config.register_settings
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(img_combined)
    elastix_img_filter.SetMovingImage(img_combined)
    param_map = sitk.GetDefaultParameterMap("groupwise")
    param_map["FinalGridSpacingInVoxels"] = [
        settings["bspline_grid_space_voxels"]]
    del param_map["FinalGridSpacingInPhysicalUnits"] # avoid conflict with vox
    param_map["MaximumNumberOfIterations"] = [settings["groupwise_iter_max"]]
    # TESTING:
    #param_map["MaximumNumberOfIterations"] = ["0"]
    _config_reg_resolutions(
        settings["grid_spacing_schedule"], param_map, img_np_template.ndim)
    elastix_img_filter.SetParameterMap(param_map)
    elastix_img_filter.PrintParameterMap()
    transform_filter = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    
    # extract individual 3D images from 4D result image
    extract_filter = sitk.ExtractImageFilter()
    size = list(transformed_img.GetSize())
    size[3] = 0 # set t to 0 to collapse this dimension
    extract_filter.SetSize(size)
    imgs = []
    num_images = len(img_files)
    for i in range(num_images):
        extract_filter.SetIndex([0, 0, 0, i]) # x, y, z, t
        img = extract_filter.Execute(transformed_img)
        img_np = sitk.GetArrayFromImage(img)
        # resize to original shape of first image, all aligned to position 
        # of subject within first image
        img_large_np = np.zeros(size_orig[::-1])
        img_large_np[:, start_y:start_y+img_np.shape[1]] = img_np
        if show_imgs:
            sitk.Show(replace_sitk_with_numpy(img, img_large_np))
        imgs.append(img_large_np)
    
    # combine all images by taking their mean
    img_mean = np.mean(imgs, axis=0)
    extend_borders = settings["extend_borders"]
    carve_threshold = settings["carve_threshold"]
    if extend_borders and carve_threshold:
        # merge in specified border region from first image for pixels below 
        # carving threshold to prioritize groupwise image
        slices = []
        for border in extend_borders[::-1]:
            slices.append(slice(*border) if border else slice(None))
        slices = tuple(slices)
        region = img_mean[tuple(slices)]
        region_template = img_np_template[tuple(slices)]
        mask = region < carve_threshold
        region[mask] = region_template[mask]
    img_raw = replace_sitk_with_numpy(transformed_img, img_mean)
    
    # carve groupwise registered image if given thresholds
    imgs_to_show = []
    imgs_to_show.append(img_raw)
    holes_area = settings["holes_area"]
    if carve_threshold and holes_area:
        img_mean, _, img_mean_unfilled = plot_3d.carve(
            img_mean, thresh=carve_threshold, holes_area=holes_area, 
            return_unfilled=True)
        img_unfilled = replace_sitk_with_numpy(
            transformed_img, img_mean_unfilled)
        transformed_img = replace_sitk_with_numpy(transformed_img, img_mean)
        # will show unfilled and filled in addition to raw image
        imgs_to_show.append(img_unfilled)
        imgs_to_show.append(transformed_img)
    
    if show_imgs:
        for img in imgs_to_show: sitk.Show(img)
    
    #transformed_img = img_raw
    if write_imgs:
        # write both the .mhd and Numpy array files to a separate folder to 
        # mimic the atlas folder format
        out_path = os.path.join(name_prefix, IMG_GROUPED)
        if not os.path.exists(name_prefix):
            os.makedirs(name_prefix)
        print("writing {}".format(out_path))
        sitk.WriteImage(transformed_img, out_path, False)
        img_np = sitk.GetArrayFromImage(transformed_img)
        detector.resolutions = [transformed_img.GetSpacing()[::-1]]
        importer.save_np_image(img_np[None], out_path, config.series)
    
    print("time elapsed for groupwise registration (s): {}"
          .format(time() - start_time))

def register_labels_to_atlas(path_fixed):
    """Register annotation labels to its own atlas.
    
    Labels may not match their own underlying atlas image well, 
    particularly in the orthogonal directions in which the labels 
    were not constructed. To improve alignment between the labels 
    and the atlas itself, register the edge-detected versions of the 
    labels and its atlas.
    
    Edge files are assumed to have been generated by 
    :func:``make_edge_images``.
    
    Args:
        path_fixed: Path to the fixed file, typically the atlas file 
            with stained sections. The corresponding edge and labels 
            files will be loaded based on this path.
    """
    # load corresponding edge files
    fixed_sitk = load_registered_img(
        path_fixed, get_sitk=True, reg_name=IMG_ATLAS_EDGE)
    moving_sitk = load_registered_img(
        path_fixed, get_sitk=True, reg_name=IMG_LABELS_EDGE)
    
    # set up SimpleElastix filter
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_sitk)
    elastix_img_filter.SetMovingImage(moving_sitk)
    
    # set up registration parameters
    settings = config.register_settings
    param_map_vector = sitk.VectorOfParameterMap()
    # bspline for non-rigid deformation
    param_map = sitk.GetDefaultParameterMap("bspline")
    param_map["FinalGridSpacingInVoxels"] = [
        settings["bspline_grid_space_voxels"]]
    del param_map["FinalGridSpacingInPhysicalUnits"] # avoid conflict with vox
    param_map["MaximumNumberOfIterations"] = [settings["bspline_iter_max"]]
    #param_map["MaximumNumberOfIterations"] = "0"
    _config_reg_resolutions(
        settings["grid_spacing_schedule"], param_map, fixed_sitk.GetDimension())
    param_map_vector.append(param_map)
    elastix_img_filter.SetParameterMap(param_map_vector)
    elastix_img_filter.PrintParameterMap()
    
    # perform the registration
    transform = elastix_img_filter.Execute()
    transformed_img = elastix_img_filter.GetResultImage()
    sitk.Show(transformed_img)
    
    # set up filter to apply the same transformation to label file, 
    # ensuring that no new labels are interpolated
    transform_param_map = elastix_img_filter.GetTransformParameterMap()
    transform_param_map[-1]["FinalBSplineInterpolationOrder"] = ["0"]
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    
    # apply transformation, manually resetting spacing in case of rounding
    labels_sitk = load_registered_img(
        path_fixed, get_sitk=True, reg_name=IMG_LABELS)
    transformed_labels = _transform_labels(transformix_img_filter, labels_sitk)
    transformed_labels.SetSpacing(transformed_img.GetSpacing())
    #sitk.Show(transformed_labels)
    
    # write transformed labels file
    out_path_base = os.path.splitext(path_fixed)[0] + "_edgereg.mhd"
    imgs_write = {
        IMG_LABELS: transformed_labels}
    write_reg_images(imgs_write, out_path_base)
    # copy original atlas metadata file to allow opening this atlas 
    # alongside new labels image for comparison
    shutil.copy(_reg_out_path(path_fixed, IMG_ATLAS), out_path_base)

def _mirror_imported_labels(labels_img_np, start):
    # mirror labels that have been imported and transformed with x and z swapped
    labels_img_np = _mirror_planes(
        np.swapaxes(labels_img_np, 0, 2), start, mirror_mult=-1, 
        check_equality=True)
    labels_img_np = np.swapaxes(labels_img_np, 0, 2)
    return labels_img_np

def make_edge_images(path_img, show=True, atlas=True, suffix=None, 
                     path_atlas_dir=None):
    """Make edge-detected atlas and associated labels images.
    
    The atlas is assumed to be a sample (eg microscopy) image on which 
    an edge-detection filter will be applied. The labels image is 
    assumed to be an annotated image whose edges will be found by 
    obtaining the borders of all separate labels.
    
    Args:
        path_img: Path to the image atlas. The labels image will be 
            found as a corresponding, registered image, unless 
            ``path_atlas_dir`` is given.
        show_imgs: True if the output images should be displayed; defaults 
            to True.
        atlas: True if the primary image is an atlas, which is assumed 
            to be symmetrical. False if the image is an experimental/sample 
            image, in which case erosion will be performed on the full 
            images, and stats will not be performed.
        suffix: Modifier to append to end of ``path_img`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
        path_atlas_dir: Path to atlas directory to use labels from that 
            directory rather than from labels image registered to 
            ``path_img``, such as when the sample image is registered 
            to an atlas rather than the other way around. Typically 
            coupled with ``suffix`` to compare same sample against 
            different labels. Defaults to None.
    """
    
    # load atlas image, assumed to be a histology image
    if atlas:
        print("generating edge images for atlas")
        atlas_suffix = IMG_ATLAS
    else:
        print("generating edge images for experiment/sample image")
        atlas_suffix = IMG_EXP
    
    # adjust image path with suffix
    mod_path = path_img
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
    
    labels_from_atlas_dir = path_atlas_dir and os.path.isdir(path_atlas_dir)
    if labels_from_atlas_dir:
        # load labels from atlas directory
        # TODO: consider applying suffix to labels dir
        path_atlas = path_img
        path_labels = os.path.join(path_atlas_dir, IMG_LABELS)
        print("loading labels from", path_labels)
        labels_sitk = sitk.ReadImage(path_labels)
    else:
        # load labels registered to sample image
        path_atlas = mod_path
        labels_sitk = load_registered_img(
            mod_path, get_sitk=True, reg_name=IMG_LABELS)
    labels_img_np = sitk.GetArrayFromImage(labels_sitk)
    
    # load atlas image, set resolution from it
    atlas_sitk = load_registered_img(
        path_atlas, get_sitk=True, reg_name=atlas_suffix)
    detector.resolutions = np.array([atlas_sitk.GetSpacing()[::-1]])
    atlas_np = sitk.GetArrayFromImage(atlas_sitk)
    
    # output images
    atlas_sitk_log = None
    atlas_sitk_edge = None
    labels_sitk_edge = None
    labels_sitk_interior = None
    
    log_sigma = config.register_settings["log_sigma"]
    if log_sigma is not None and suffix is None:
        # generate LoG and edge-detected images for original image
        print("generating LoG edge-detected images with sigma", log_sigma)
        thresh = (config.register_settings["atlas_threshold"] 
                  if config.register_settings["log_atlas_thresh"] else None)
        atlas_log = plot_3d.laplacian_of_gaussian_img(
            atlas_np, sigma=log_sigma, labels_img=labels_img_np, thresh=thresh)
        atlas_sitk_log = replace_sitk_with_numpy(atlas_sitk, atlas_log)
        atlas_edge = plot_3d.zero_crossing(atlas_log, 1).astype(np.uint8)
        atlas_sitk_edge = replace_sitk_with_numpy(atlas_sitk, atlas_edge)
    else:
        # if sigma not set or if using suffix to compare two images, 
        # load from original image to copmare against common image
        atlas_edge = load_registered_img(path_img, reg_name=IMG_ATLAS_EDGE)
    
    # make map of label interiors for interior/border comparisons
    erosion = config.register_settings["marker_erosion"]
    erosion_frac = config.register_settings["erosion_frac"]
    interior = erode_labels(labels_img_np, erosion, erosion_frac, atlas)
    labels_sitk_interior = replace_sitk_with_numpy(labels_sitk, interior)
    
    # make labels edge and edge distance images
    dist_to_orig, labels_edge = edge_distances(
        labels_img_np, atlas_edge, spacing=atlas_sitk.GetSpacing()[::-1])
    dist_sitk = replace_sitk_with_numpy(atlas_sitk, dist_to_orig)
    labels_sitk_edge = replace_sitk_with_numpy(labels_sitk, labels_edge)
    
    # show all images
    imgs_write = {
        IMG_ATLAS_LOG: atlas_sitk_log, 
        IMG_ATLAS_EDGE: atlas_sitk_edge, 
        IMG_LABELS_EDGE: labels_sitk_edge, 
        IMG_LABELS_INTERIOR: labels_sitk_interior, 
        IMG_LABELS_DIST: dist_sitk, 
    }
    if show:
        for img in imgs_write.values():
            if img: sitk.Show(img)
    
    # write images to same directory as atlas with appropriate suffix
    write_reg_images(imgs_write, mod_path)

def erode_labels(labels_img_np, erosion, erosion_frac=None, atlas=True):
    """Erode labels image for use as markers or a map of the interior.
    
    Args:
        labels_img_np: Numpy image array of labels in z,y,x format.
        erosion: Filter size for erosion.
        erosion_frac: Target erosion fraction; defaults to None.
        atlas: True if the primary image is an atlas, which is assumed 
            to be symmetrical. False if the image is an experimental/sample 
            image, in which case erosion will be performed on the full image.
    
    Returns:
        The eroded labels as a new array of same shape as that of 
        ``labels_img_np``.
    """
    labels_to_erode = labels_img_np
    if atlas:
        # for atlases, assume that labels image is symmetric across the x-axis
        len_half = labels_img_np.shape[2] // 2
        labels_to_erode = labels_img_np[..., :len_half]
    
    # convert labels image into markers
    #eroded = segmenter.labels_to_markers_blob(labels_img_np)
    eroded = segmenter.labels_to_markers_erosion(
        labels_to_erode, erosion, erosion_frac)
    if atlas:
        eroded = _mirror_imported_labels(eroded, len_half)
    
    return eroded

def edge_aware_segmentation(path_atlas, show=True, atlas=True, suffix=None):
    """Segment an atlas using its previously generated edge map.
    
    Labels may not match their own underlying atlas image well, 
    particularly in the orthogonal directions in which the labels 
    were not constructed. To improve alignment between the labels 
    and the atlas itself, register the labels to an automated, roughly 
    segmented version of the atlas. The goal is to improve the 
    labels' alignment so that the atlas/labels combination can be 
    used for another form of automated segmentation by registering 
    them to experimental brains via :func:``register``.
    
    Edge files are assumed to have been generated by 
    :func:``make_edge_images``.
    
    Args:
        path_atlas: Path to the fixed file, typically the atlas file 
            with stained sections. The corresponding edge and labels 
            files will be loaded based on this path.
        show_imgs: True if the output images should be displayed; defaults 
            to True.
        atlas: True if the primary image is an atlas, which is assumed 
            to be symmetrical. False if the image is an experimental/sample 
            image, in which case segmentation will be performed on the full 
            images, and stats will not be performed.
        suffix: Modifier to append to end of ``path_atlas`` basename for 
            registered image files that were output to a modified name; 
            defaults to None. If ``atlas`` is True, ``suffix`` will only 
            be applied to saved files, with files still loaded based on the 
            original path.
    """
    # adjust image path with suffix
    load_path = path_atlas
    mod_path = path_atlas
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
        if atlas: load_path = mod_path
    
    # load corresponding files via SimpleITK
    atlas_sitk = load_registered_img(
        load_path, get_sitk=True, reg_name=IMG_ATLAS)
    atlas_sitk_edge = load_registered_img(
        load_path, get_sitk=True, reg_name=IMG_ATLAS_EDGE)
    labels_sitk = load_registered_img(
        load_path, get_sitk=True, reg_name=IMG_LABELS)
    labels_sitk_markers = load_registered_img(
        load_path, get_sitk=True, reg_name=IMG_LABELS_MARKERS)
    
    # get Numpy arrays of images
    atlas_img_np = sitk.GetArrayFromImage(atlas_sitk)
    atlas_edge = sitk.GetArrayFromImage(atlas_sitk_edge)
    labels_img_np = sitk.GetArrayFromImage(labels_sitk)
    markers = sitk.GetArrayFromImage(labels_sitk_markers)
    
    # segment image from markers
    if atlas:
        # segment only half of image, assuming symmetry
        len_half = atlas_img_np.shape[2] // 2
        labels_seg = segmenter.segment_from_labels(
            atlas_edge[..., :len_half], markers[..., :len_half], 
            labels_img_np[..., :len_half])
    else:
        labels_seg = segmenter.segment_from_labels(
            atlas_edge, markers, labels_img_np)
    
    smoothing = config.register_settings["smooth"]
    if smoothing is not None:
        # smoothing by opening operation based on profile setting
        smooth_labels(labels_seg, smoothing, config.SmoothingModes.opening)
    
    if atlas:
        # mirror back to other half
        labels_seg = _mirror_imported_labels(labels_seg, len_half)
    
    # expand background to smoothed background of original labels to 
    # roughly match background while still allowing holes to be filled
    crop = config.register_settings["crop_to_orig"]
    _crop_to_orig(labels_img_np, labels_seg, crop)
    
    if labels_seg.dtype != labels_img_np.dtype:
        # watershed may give different output type, so cast back if so
        labels_seg = labels_seg.astype(labels_img_np.dtype)
    labels_sitk_seg = replace_sitk_with_numpy(labels_sitk, labels_seg)
    
    # show DSCs for labels
    _measure_overlap_combined_labels(atlas_sitk, labels_sitk_seg)
    print("\nMeasuring overlap of individual labels:")
    measure_overlap_labels(labels_sitk, labels_sitk_seg)
    print("\nMeasuring foreground overlap of labels:")
    measure_overlap_labels(
        make_labels_fg(labels_sitk), make_labels_fg(labels_sitk_seg))
    
    # show and write image to same directory as atlas with appropriate suffix
    write_reg_images({IMG_LABELS: labels_sitk_seg}, mod_path)
    if show: sitk.Show(labels_sitk_seg)
    return path_atlas

def merge_atlas_segmentations(img_paths, show=True, atlas=True, suffix=None):
    """Merge atlas segmentations for a list of files as a multiprocessing 
    wrapper for :func:``merge_atlas_segmentations``, after which 
    edge image post-processing is performed separately since it 
    contains tasks also performed in multiprocessing.
    
    Args:
        img_path: Path to image to load.
        show_imgs: True if the output images should be displayed; defaults 
            to True.
        atlas: True if the image is an atlas; defaults to True.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
    """
    start_time = time()
    
    # erode all labels images into markers from which to grown via watershed
    erosion = config.register_settings["marker_erosion"]
    erosion_frac = config.register_settings["erosion_frac"]
    for img_path in img_paths:
        mod_path = img_path
        if suffix is not None:
            mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
        print("Generating label markers for", mod_path)
        labels_sitk = load_registered_img(
            mod_path, get_sitk=True, reg_name=IMG_LABELS)
        # use default minimal post-erosion size
        markers = erode_labels(
            sitk.GetArrayFromImage(labels_sitk), erosion, atlas=atlas)
        labels_sitk_markers = replace_sitk_with_numpy(labels_sitk, markers)
        write_reg_images({IMG_LABELS_MARKERS: labels_sitk_markers}, mod_path)
    
    pool = mp.Pool()
    pool_results = []
    for img_path in img_paths:
        print("setting up atlas segmentation merge for", img_path)
        # convert labels image into markers
        pool_results.append(pool.apply_async(
            edge_aware_segmentation, args=(img_path, show, atlas, suffix)))
    for result in pool_results:
        # edge distance calculation and labels interior image generation 
        # are multiprocessed, so run them as post-processing tasks to 
        # avoid nested multiprocessing
        path = result.get()
        mod_path = path
        if suffix is not None:
            mod_path = lib_clrbrain.insert_before_ext(path, suffix)
        
        # make edge distance images and stats
        labels_sitk = load_registered_img(
            mod_path, get_sitk=True, reg_name=IMG_LABELS)
        labels_np = sitk.GetArrayFromImage(labels_sitk)
        dist_to_orig, labels_edge = edge_distances(
            labels_np, path=path, spacing=labels_sitk.GetSpacing()[::-1])
        dist_sitk = replace_sitk_with_numpy(labels_sitk, dist_to_orig)
        labels_sitk_edge = replace_sitk_with_numpy(labels_sitk, labels_edge)
        
        # make interior images from labels using given target post-erosion frac
        interior = erode_labels(
            labels_np, erosion, erosion_frac=erosion_frac, atlas=atlas)
        labels_sitk_interior = replace_sitk_with_numpy(labels_sitk, interior)
        
        # write images to same directory as atlas
        imgs_write = {
            IMG_LABELS_DIST: dist_sitk, 
            IMG_LABELS_EDGE: labels_sitk_edge, 
            IMG_LABELS_INTERIOR: labels_sitk_interior, 
        }
        write_reg_images(imgs_write, mod_path)
        if show:
            for img in imgs_write.values():
                if img: sitk.Show(img)
        print("finished {}".format(path))
    pool.close()
    pool.join()
    print("time elapsed for merging atlas segmentations:", time() - start_time)

def edge_distances(labels, atlas_edge=None, path=None, spacing=None):
    """Measure the distance between edge images.
    
    Args:
        labels: Labels image as Numpy array.
        atlas_edge: Image as a Numpy array of the atlas reduced to its edges. 
            Defaults to None to load from the corresponding registered 
            file path based on ``path``.
        path: Path from which to load ``atlas_edge`` if it is None.
        spacing: Grid spacing sequence of same length as number of image 
            axis dimensions; defaults to None.
    
    Returns:
        An image array of the same shape as ``labels_edge`` with 
        label edge values replaced by corresponding distance values.
    """
    if atlas_edge is None:
        atlas_edge = load_registered_img(path, reg_name=IMG_ATLAS_EDGE)
    
    # create distance map between edges of original and new segmentations
    labels_edge = vols.make_labels_edge(labels)
    dist_to_orig, _, _ = plot_3d.borders_distance(
        atlas_edge != 0, labels_edge != 0, spacing=spacing)
    
    return dist_to_orig, labels_edge

def make_density_image(img_path, scale=None, shape=None, suffix=None, 
                       labels_img_sitk=None):
    """Make a density image based on associated blobs.
    
    Uses the shape of the registered labels image by default to set 
    the voxel sizes for the blobs.
    
    Args:
        img_path: Path to image, which will be used to indentify the blobs file.
        scale: Rescaling factor as a scalar value to find the corresponding 
            full-sized image. Defaults to None to use the register 
            setting ``target_size`` instead if available, falling back 
            to load the full size image to find its shape if necessary.
        shape: Final shape size; defaults to None to use the shape of 
            the labels image.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
        labels_img_sitk: Labels image as a SimpleITK ``Image`` object; 
            defaults to None, in which case the registered labels image file 
            corresponding to ``img_path`` with any ``suffix`` modifier 
            will be opened.
    
    Returns:
        Tuple of the density image as a Numpy array in the same shape as 
        the opened image; Numpy array of blob IDs; and the original 
        ``img_path`` to track such as for multiprocessing.
    """
    mod_path = img_path
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(img_path, suffix)
    if labels_img_sitk is None:
        labels_img_sitk = load_registered_img(
            mod_path, get_sitk=True, reg_name=IMG_LABELS)
    labels_img = sitk.GetArrayFromImage(labels_img_sitk)
    # load blobs
    filename_base = importer.filename_to_base(
        img_path, config.series)
    info = np.load(filename_base + config.SUFFIX_INFO_PROC)
    blobs = info["segments"]
    print("loading {} blobs".format(len(blobs)))
    # get scaling from source image, which can be rescaled/resized image 
    # since contains scaling image
    load_size = config.register_settings["target_size"]
    img_path_transposed = transformer.get_transposed_image_path(
        img_path, scale, load_size)
    if scale is not None or load_size is not None:
        image5d, img_info = importer.read_file(
            img_path_transposed, config.series, return_info=True)
        scaling = img_info["scaling"]
    else:
        # fall back to scaling based on comparison to original image
        image5d = importer.read_file(
            img_path_transposed, config.series)
        scaling = importer.calc_scaling(image5d, labels_img)
    if shape is not None:
        # scale blobs to an alternative final size
        scaling = np.divide(shape, np.divide(labels_img.shape, scaling))
        labels_spacing = np.multiply(
            labels_img_sitk.GetSpacing()[::-1], 
            np.divide(labels_img.shape, shape))
        labels_img = np.zeros(shape, dtype=labels_img.dtype)
        labels_img_sitk.SetSpacing(labels_spacing[::-1])
    print("using scaling: {}".format(scaling))
    # annotate blobs based on position
    blobs_ids, coord_scaled = ontology.get_label_ids_from_position(
        blobs[:, :3], labels_img, scaling, 
        return_coord_scaled=True)
    print("blobs_ids: {}".format(blobs_ids))
    
    # build heat map to store densities per label px and save to file
    heat_map = plot_3d.build_heat_map(labels_img.shape, coord_scaled)
    out_path = _reg_out_path(mod_path, IMG_HEAT_MAP)
    print("writing {}".format(out_path))
    heat_map_sitk = replace_sitk_with_numpy(labels_img_sitk, heat_map)
    sitk.WriteImage(heat_map_sitk, out_path, False)
    return heat_map, blobs_ids, img_path

def make_density_images_mp(img_paths, scale=None, shape=None, suffix=None):
    """Make density images for a list of files as a multiprocessing 
    wrapper for :func:``make_density_image``
    
    Args:
        img_path: Path to image, which will be used to indentify the blobs file.
        scale: Rescaling factor as a scalar value. If set, the corresponding 
            image for this factor will be opened. If None, the full size 
            image will be used. Defaults to None.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
    """
    start_time = time()
    pool = mp.Pool()
    pool_results = []
    for img_path in img_paths:
        print("making image", img_path)
        pool_results.append(pool.apply_async(
            make_density_image, args=(img_path, scale, shape, suffix)))
    heat_maps = []
    for result in pool_results:
        _, _, path = result.get()
        print("finished {}".format(path))
    pool.close()
    pool.join()
    print("time elapsed for making density images:", time() - start_time)

def make_sub_segmented_labels(img_path, suffix=None):
    """Divide each label based on anatomical borders to create a 
    sub-segmented image.
    
    The segmented labels image will be loaded, or if not available, the 
    non-segmented labels will be loaded instead.
    
    Args:
        img_path: Path to main image from which registered images will 
            be loaded.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
    
    Returns:
        Sub-segmented image as a Numpy array of the same shape as 
        the image at ``img_path``.
    """
    # adjust image path with suffix
    mod_path = img_path
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
    
    # load labels
    labels_sitk = load_registered_img(
        mod_path, get_sitk=True, reg_name=IMG_LABELS)
    
    # atlas edge image is associated with original, not modified image
    atlas_edge = load_registered_img(img_path, reg_name=IMG_ATLAS_EDGE)
    
    # sub-divide the labels and save to file
    labels_img_np = sitk.GetArrayFromImage(labels_sitk)
    labels_subseg = segmenter.sub_segment_labels(labels_img_np, atlas_edge)
    labels_subseg_sitk = replace_sitk_with_numpy(labels_sitk, labels_subseg)
    write_reg_images({IMG_LABELS_SUBSEG: labels_subseg_sitk}, mod_path)
    return labels_subseg

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
    img_np_base = None
    img_nps = []
    for img_path in img_paths:
        mod_path = img_path
        if suffix is not None:
            # adjust image path with suffix
            mod_path = lib_clrbrain.insert_before_ext(mod_path, suffix)
        print("loading", mod_path)
        get_sitk = img_sitk is None
        img = load_registered_img(
            mod_path, get_sitk=get_sitk, reg_name=reg_name)
        if get_sitk:
            # use the first image as template
            img_np = sitk.GetArrayFromImage(img)
            img_np_base = img_np
            img_sitk = img
        else:
            # resize to first image
            img_np = transform.resize(
                img, img_np_base.shape, preserve_range=True, 
                anti_aliasing=True, mode="reflect")
        img_nps.append(img_np)
    
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
        output_base = lib_clrbrain.insert_before_ext(output_base, suffix)
    output_reg = lib_clrbrain.insert_before_ext(
        reg_name, COMBINED_SUFFIX, "_")
    write_reg_images({output_reg: combined_sitk}, output_base)
    return combined_sitk

def overlay_registered_imgs(fixed_file, moving_file_dir, plane=None, 
                            flip=False, name_prefix=None, out_plane=None):
    """Shows overlays of previously saved registered images.
    
    Should be run after :func:`register` has written out images in default
    (xy) orthogonal orientation. Also output the Dice similiarity coefficient.
    
    Args:
        fixed_file: Path to the fixed file.
        moving_file_dir: Moving files directory, from which the original
            atlas will be retrieved.
        plane: Orthogonal plane to flip the moving image.
        flip: If true, will flip the fixed file first; defaults to False.
        name_prefix: Path with base name where registered files are located; 
            defaults to None, in which case the fixed_file path will be used.
        out_plane: Output plane to view.
    """
    # get the experiment file
    if name_prefix is None:
        name_prefix = fixed_file
    image5d = importer.read_file(fixed_file, config.series)
    roi = image5d[0, ...] # not using time dimension
    
    # get the atlas file and transpose it to match the orientation of the 
    # experiment image
    out_path = os.path.join(moving_file_dir, IMG_ATLAS)
    print("Reading in {}".format(out_path))
    moving_sitk = sitk.ReadImage(out_path)
    moving_sitk = transpose_img(moving_sitk, plane, 2 if flip else 0)
    moving_img = sitk.GetArrayFromImage(moving_sitk)
    
    # get the registered atlas file, which should already be transposed
    transformed_sitk = load_registered_img(name_prefix, get_sitk=True)
    transformed_img = sitk.GetArrayFromImage(transformed_sitk)
    
    # get the registered labels file, which should also already be transposed
    labels_img = load_registered_img(name_prefix, reg_name=IMG_LABELS)
    
    # calculate the Dice similarity coefficient
    measure_overlap(_load_numpy_to_sitk(fixed_file), transformed_sitk)
    
    # overlay the images
    imgs = [roi, moving_img, transformed_img, labels_img]
    _, translation = _handle_transform_file(name_prefix)
    translation = _translation_adjust(
        moving_sitk, transformed_sitk, translation, flip=True)
    _show_overlays(imgs, translation, fixed_file, out_plane)

def write_reg_images(imgs_write, prefix, copy_to_suffix=False, ext=None):
    """Write registered images to file.
    
    Args:
        imgs_write: Dictionary of ``{suffix: image}``, where ``suffix`` 
            is a registered images suffix, such as :const:``IMAGE_LABELS``, 
            and ``image`` is a SimpleITK image object. If the image does 
            not exist, the file will not be written.
        prefix: Base path from which to construct registered file paths.
        copy_to_suffix: If True, copy the output path to a file in the 
            same directory with ``suffix`` as the filename, which may 
            be useful when setting the registered images as the 
            main images in the directory. Defaults to False.
        ext: Replace extension with this value if given; defaults to None.
    """
    target_dir = os.path.dirname(prefix)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for suffix in imgs_write.keys():
        img = imgs_write[suffix]
        if img is None: continue
        if ext: suffix = lib_clrbrain.match_ext(ext, suffix)
        out_path = _reg_out_path(prefix, suffix)
        sitk.WriteImage(img, out_path, False)
        print("wrote registered image to", out_path)
        if copy_to_suffix:
            # copy metadata file to allow opening images from bare suffix name, 
            # such as when this atlas becomes the new atlas for registration
            out_path_copy = os.path.join(target_dir, suffix)
            shutil.copy(out_path, out_path_copy)
            print("also copied to", out_path_copy)

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

def load_registered_img(img_path, get_sitk=False, reg_name=IMG_ATLAS, 
                        replace=None):
    """Load atlas-based image that has been registered to another image.
    
    Args:
        img_path: Path as had been given to generate the registered images, 
            with the parent path of the registered images and base name of 
            the original image.
        get_sitk: True if the image should be returned as a SimpleITK image; 
            defaults to False, in which case the corresponding Numpy array will 
            be extracted instead.
        reg_name: Atlas image type to open; defaults to :const:``IMG_ATLAS``, 
            which will open the main atlas.
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
    reg_img_path = _reg_out_path(img_path, reg_name, True)
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

def get_scaled_regionprops(img_region, scaling):
    """Get the scaled regionprops for an image mask
    
    Args:
        img_region: Mask of the region.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
    
    Returns:
        A tuple of ``props, bbox, centroid``. ``props`` is the ``regionprops`` 
        properties for the given region. ``bbox`` is the bounding box 
        of this region scaled back to the experiment image. ``centroid`` is 
        the centroid position also scaled to the experiment. Note that the 
        bounding box may encompass many pixels not included in the region 
        itself, including border pixels or even internal pixels in irregularly 
        shaped or sparsely scattered regions. If so, the centroid position 
        may in fact be outside of the region. To ensure that a coordinate 
        within the region tree for ``label_id`` is returned, use 
        :func:``ontology.get_region_middle`` instead.
    """
    # TODO: consider deprecating
    # TODO: use only deepest child?
    img_region = img_region.astype(np.int32)
    print("img_region 1's: {}".format(np.count_nonzero(img_region)))
    props = measure.regionprops(img_region)
    print("num props: {}".format(len(props)))
    scaling_inv = 1 / scaling
    if len(props) > 0:
        bbox_scaled = np.array(props[0].bbox)
        bbox_scaled[:3] = bbox_scaled[:3] * scaling_inv
        bbox_scaled[3:] = bbox_scaled[3:] * scaling_inv
        centroid_scaled = np.array(props[0].centroid) * scaling_inv
        bbox_scaled = np.around(bbox_scaled).astype(np.int)
        centroid_scaled = np.around(centroid_scaled).astype(np.int)
        print(props[0].bbox, props[0].centroid)
        print(bbox_scaled, centroid_scaled)
        return props, bbox_scaled, centroid_scaled
    return props, None, None

def _find_atlas_labels(load_labels, max_level, labels_ref_lookup):
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
    orig_labels_path = os.path.join(orig_atlas_dir, IMG_LABELS)
    # need all labels from a reference as registered image may have lost labels
    if max_level is None and os.path.exists(orig_labels_path):
        # use all drawn labels in original labels image
        orig_labels_sitk = sitk.ReadImage(orig_labels_path)
        orig_labels_np = sitk.GetArrayFromImage(orig_labels_sitk)
        label_ids = np.unique(orig_labels_np).tolist()
    else:
        # use all labels in ontology reference to include hierarchical 
        # labels or if original labels image isn't presenst
        label_ids = list(labels_ref_lookup.keys())
    return label_ids

def volumes_by_id(img_paths, labels_ref_lookup, suffix=None, unit_factor=None, 
                   groups=None, max_level=None, combine_sides=True):
    """Get volumes and additional label metrics for each single labels ID.
    
    Args:
        img_paths: Sequence of images.
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup` 
            to look up by ID while preserving key order to ensure that 
            parents of any child will be reached prior to the child.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None to use "original" as the condition.
        unit_factor: Factor by which volumes will be divided to adjust units; 
            defaults to None.
        groups: Dictionary of sample grouping metadata, where each 
            entry is a list with a values corresponding to ``img_paths``.
        max_level: Integer of maximum ontological level to measure. 
            Giving this value changes the measurement to volumes 
            by ontology, where all children of each label are included 
            in the label rather than taking the label at face value, 
            the default drawn label. Labels at levels up to and including 
            this value will be included. Defaults to None to take 
            labels at face value only.
        combine_sides: True to combine corresponding labels from opposite 
            sides of the sample; defaults to True. Corresponding positive 
            and negative numbers will always be included, and this 
            flag will only determine whether the sides are combined 
            rather than kept separate.
    
    Returns:
        Tuple of Pandas data frames with volume-related metrics for each 
        sample, the first with all each region for each sample in a 
        separate row, and the second with all regions combined in a 
        weighted fashion. This second data frame will be None if 
        ``max_levels`` is not None since subregions are already included 
        in each label.
    """
    start_time = time()
    
    # prepare data frame output path and condition column based on suffix
    out_path = SAMPLE_VOLS
    out_path_summary = SAMPLE_VOLS_SUMMARY
    if max_level is not None:
        out_path = SAMPLE_VOLS_LEVELS
    if suffix is None:
        condition = "original" 
    else:
        condition = config.suffix.replace("_", "")
        out_path += config.suffix
        out_path_summary += config.suffix
    
    # grouping metadata, which will be combined with groups
    grouping = {}
    grouping["Condition"] = condition
    
    # setup labels
    label_ids = _find_atlas_labels(
        config.load_labels, max_level, labels_ref_lookup)
    if not combine_sides:
        # include opposite side as separate labels; otherwise, defer to 
        # ontology (max_level flag) or labels metrics to get labels from 
        # opposite sides by combining them
        label_ids.extend([-1 * n for n in label_ids])
    if max_level is not None:
        # setup ontological labels
        ids_with_children = []
        for label_id in label_ids:
            label = labels_ref_lookup[abs(label_id)]
            label_level = label[ontology.NODE][config.ABAKeys.LEVEL.value]
            if label_level <= max_level:
                # get children (including parent first) if up through level
                ids_with_children.append(
                    ontology.get_children_from_id(
                        labels_ref_lookup, label_id, both_sides=combine_sides))
        label_ids = ids_with_children
    
    dfs = []
    dfs_all = []
    sort_cols=["Region", "Sample", "Side"]
    for i, img_path in enumerate(img_paths):
        # adjust image path with suffix
        mod_path = img_path
        if suffix is not None:
            mod_path = lib_clrbrain.insert_before_ext(img_path, suffix)
        
        # load data frame if available
        df_path = "{}_volumes.csv".format(os.path.splitext(mod_path)[0])
        df = None
        if max_level is not None:
            if os.path.exists(df_path):
                df = pd.read_csv(df_path)
            else:
                msg = ("Could not find raw stats for drawn labels from "
                       "{}, will measure stats for individual regions "
                       "repeatedly. To save processing time, consider "
                       "stopping and re-running first without levels"
                       .format(df_path))
                warnings.warn(msg)
        
        spacing = None
        img_np = None
        labels_img_np = None
        labels_edge = None
        dist_to_orig = None
        labels_interior = None
        heat_map = None
        subseg = None
        if df is None:
            # open images registered to the main image, staring with the 
            # experimental image if available and falling back to atlas
            try:
                img_sitk = load_registered_img(
                    mod_path, get_sitk=True, reg_name=IMG_EXP)
            except FileNotFoundError as e:
                print(e)
                print("will load atlas image instead")
                img_sitk = load_registered_img(
                    mod_path, get_sitk=True, reg_name=IMG_ATLAS)
            img_np = sitk.GetArrayFromImage(img_sitk)
            spacing = img_sitk.GetSpacing()[::-1]
            
            # load labels in order of priority: full labels > truncated labels
            try:
                labels_img_np = load_registered_img(
                    mod_path, reg_name=IMG_LABELS)
            except FileNotFoundError as e:
                print(e)
                print("will attempt to load trucated labels image instead")
                labels_img_np = load_registered_img(
                    mod_path, reg_name=IMG_LABELS_TRUNC)
            
            # load labels edge and edge distances images
            labels_edge = load_registered_img(
                mod_path, reg_name=IMG_LABELS_EDGE)
            dist_to_orig = load_registered_img(
                mod_path,reg_name=IMG_LABELS_DIST)
            
            # load labels marker image
            try:
                labels_interior = load_registered_img(
                    mod_path, reg_name=IMG_LABELS_INTERIOR)
            except FileNotFoundError as e:
                print(e)
                print("will ignore label markers")
            
            # load heat map of nuclei per voxel if available
            try:
                heat_map = load_registered_img(mod_path,reg_name=IMG_HEAT_MAP)
            except FileNotFoundError as e:
                print(e)
                print("will ignore nuclei stats")
            
            # load sub-segmentation labels if available
            try:
                subseg = load_registered_img(
                    mod_path,reg_name=IMG_LABELS_SUBSEG)
            except FileNotFoundError as e:
                print(e)
                print("will ignore labels sub-segmentations")
            print("tot blobs", np.sum(heat_map))
        
        # prepare sample name with original name for comparison across 
        # conditions and add an arbitrary number of metadata grouping cols
        sample = lib_clrbrain.get_filename_without_ext(img_path)
        if groups is not None:
             for key in groups.keys():
                 grouping[key] = groups[key][i]
            
        # measure stats per label for the given sample; max_level already 
        # takes care of combining sides
        df, df_all = vols.measure_labels_metrics(
            sample, img_np, labels_img_np, 
            labels_edge, dist_to_orig, labels_interior, heat_map, subseg, 
            spacing, unit_factor, 
            combine_sides and max_level is None, 
            label_ids, grouping, df)
        if max_level is None:
            stats.data_frames_to_csv([df], df_path, sort_cols=sort_cols)
        dfs.append(df)
        dfs_all.append(df_all)
    
    # combine data frames from all samples by region for each sample
    df_combined = stats.data_frames_to_csv(
        dfs, out_path, sort_cols=sort_cols)
    df_combined_all = None
    if max_level is None:
        # combine weighted combo of all regions per sample; 
        # not necessary for levels-based (ontological) volumes since they 
        # already accumulate from sublevels
        df_combined_all = stats.data_frames_to_csv(
            dfs_all, out_path_summary)
    print("time elapsed for volumes by ID: {}".format(time() - start_time))
    return df_combined, df_combined_all

def make_labels_diff_img(img_path, df_path, meas, fn_avg, prefix=None, 
                         show=False, level=None, meas_path_name=None):
    """Replace labels in an image with the differences in metrics for 
    each given region between two conditions.
    
    Args:
        img_path: Path to the base image from which the corresponding 
            registered image will be found.
        df_path: Path to data frame with metrics for the labels.
        meas: Name of colum in data frame with the chosen measurement.
        fn_avg: Function to apply to the set of measurements, such as a mean. 
            Can be None if ``df_path`` points to a stats file from which 
            to extract metrics directly in :meth:``vols.map_meas_to_labels``.
        prefix: Start of path for output image; defaults to None to 
            use ``img_path`` instead.
        show: True to show the images after generating them; defaults to False.
        level: Ontological level at which to look up and show labels. 
            Assume that labels level image corresponding to this value 
            has already been generated by :meth:``make_labels_level_img``. 
            Defaults to None to use only drawn labels.
        meas_path_name: Name to use in place of `meas` in output path; 
            defaults to None.
    """
    # load labels image and data frame before generating map for the 
    # given metric of the chosen measurement
    reg_name = IMG_LABELS if level is None else IMG_LABELS_LEVEL.format(level)
    labels_sitk = load_registered_img(
        img_path, reg_name=reg_name, get_sitk=True)
    labels_np = sitk.GetArrayFromImage(labels_sitk)
    df = pd.read_csv(df_path)
    labels_diff = vols.map_meas_to_labels(
        labels_np, df, meas, fn_avg, reverse=True)
    if labels_diff is None: return
    labels_diff_sitk = replace_sitk_with_numpy(labels_sitk, labels_diff)
    
    # save and show labels difference image using measurement name in 
    # output path or overriding with custom name
    meas_path = meas if meas_path_name is None else meas_path_name
    reg_diff = lib_clrbrain.insert_before_ext(IMG_LABELS_DIFF, meas_path, "_")
    if fn_avg is not None:
        # add function name to output path if given
        reg_diff = lib_clrbrain.insert_before_ext(
            reg_diff, fn_avg.__name__, "_")
    imgs_write = {reg_diff: labels_diff_sitk}
    out_path = prefix if prefix else img_path
    write_reg_images(imgs_write, out_path)
    if show:
        for img in imgs_write.values():
            if img: sitk.Show(img)

def make_labels_level_img(img_path, level, prefix=None, show=False):
    """Replace labels in an image with their parents at the given level.
    
    Labels that do not fall within a parent at that level will remain in place.
    
    Args:
        img_path: Path to the base image from which the corresponding 
            registered image will be found.
        level: Ontological level at which to group child labels. 
        prefix: Start of path for output image; defaults to None to 
            use ``img_path`` instead.
        show: True to show the images after generating them; defaults to False.
    """
    # load original labels image and setup ontology dictionary
    labels_sitk = load_registered_img(
        img_path, reg_name=IMG_LABELS, get_sitk=True)
    labels_np = sitk.GetArrayFromImage(labels_sitk)
    ref = ontology.load_labels_ref(config.load_labels)
    labels_ref_lookup = ontology.create_aba_reverse_lookup(ref)
    
    ids = list(labels_ref_lookup.keys())
    for key in ids:
        keys = [key, -1 * key]
        for region in keys:
            if region == 0: continue
            # get ontological label
            label = labels_ref_lookup[abs(region)]
            label_level = label[ontology.NODE][config.ABAKeys.LEVEL.value]
            if label_level == level:
                # get children (including parent first) at given level 
                # and replace them with parent
                label_ids = ontology.get_children_from_id(
                    labels_ref_lookup, region)
                labels_region = np.isin(labels_np, label_ids)
                print("replacing labels within", region)
                labels_np[labels_region] = region
    labels_level_sitk = replace_sitk_with_numpy(labels_sitk, labels_np)
    
    # generate an edge image at this level
    labels_edge = vols.make_labels_edge(labels_np)
    labels_edge_sikt = replace_sitk_with_numpy(labels_sitk, labels_edge)
    
    # write and optionally display labels level image
    imgs_write = {
        IMG_LABELS_LEVEL.format(level): labels_level_sitk, 
        IMG_LABELS_EDGE_LEVEL.format(level): labels_edge_sikt, 
    }
    out_path = prefix if prefix else img_path
    write_reg_images(imgs_write, out_path)
    if show:
        for img in imgs_write.values():
            if img: sitk.Show(img)

def export_region_ids(labels_ref_lookup, path, level):
    """Export region IDs from annotation reference reverse mapped dictionary 
    to CSV file.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup` 
            to look up by ID while preserving key order to ensure that 
            parents of any child will be reached prior to the child.
        path: Path to output CSV file; if does not end with ``.csv``, it will 
            be added.
        level: Level at which to find parent for each label. If None, 
            a parent level of -1 will be used, and label IDs will be 
            taken from the labels image rather than the full set of 
            labels from the ``labels_ref_lookup``.
    
    Returns:
        Pandas data frame of the region IDs and corresponding names.
    """
    ext = ".csv"
    if not path.endswith(ext): path += ext
    
    # find parents for label at the given level
    parent_level = -1 if level is None else level
    label_parents = ontology.labels_to_parent(labels_ref_lookup, parent_level)
    
    cols = ("Region", "RegionAbbr", "RegionName", "Level", "Parent")
    data = OrderedDict()
    label_ids = _find_atlas_labels(
        config.load_labels, level, labels_ref_lookup)
    for key in label_ids:
        # does not include laterality distinction, only using original IDs
        if key <= 0: continue
        label = labels_ref_lookup[key]
        # ID of parent at label_parents' level
        parent = label_parents[key]
        vals = (key, label[ontology.NODE][config.ABAKeys.ACRONYM.value], 
                label[ontology.NODE][config.ABAKeys.NAME.value], 
                label[ontology.NODE][config.ABAKeys.LEVEL.value], parent)
        for col, val in zip(cols, vals):
            data.setdefault(col, []).append(val)
    data_frame = stats.dict_to_data_frame(data, path)
    return data_frame

def export_region_network(labels_ref_lookup, path):
    """Export region network file showing relationships among regions 
    according to the SIF specification.
    
    See http://manual.cytoscape.org/en/stable/Supported_Network_File_Formats.html#sif-format
    for file format information.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`ontology.create_reverse_lookup` 
            to look up by ID while preserving key order to ensure that 
            parents of any child will be reached prior to the child.
        path: Path to output SIF file; if does not end with ``.sif``, it will 
            be added.
    """
    ext = ".sif"
    if not path.endswith(ext): path += ext
    network = {}
    for key in labels_ref_lookup.keys():
        if key < 0: continue # only use original, non-neg region IDs
        label = labels_ref_lookup[key]
        parents = label.get(ontology.PARENT_IDS)
        if parents:
            for parent in parents[::-1]:
                # work backward since closest parent listed last
                print("{} looking for parent {} in network".format(key, parent))
                network_parent = network.get(parent)
                if network_parent is not None:
                    # assume that all parents will have already been entered 
                    # into the network dict since the keys were entered in 
                    # hierarchical order and maintain their order of entry
                    network_parent.append(key)
                    break
        # all regions have a node, even if connected to no one
        network[key] = []
    
    with open(path, "w", newline="") as csv_file:
        stats_writer = csv.writer(csv_file, delimiter=" ")
        # each region will have a line along with any of its immediate children
        for key in network.keys():
            children = network[key]
            row = [str(key)]
            if children:
                row.extend(["pp", *children])
            stats_writer.writerow(row)
    print("output region network to {}".format(path))

def export_common_labels(img_paths, output_path):
    """Export data frame combining all label IDs from the given atlases, 
    showing the presence of labels in each atlas.
    
    Args:
        img_paths: Image paths from which to load the corresponding 
            labels images.
        output_path: Path to export data frame to .csv.
    
    Returns:
        Data frame with label IDs as indices, column for each atlas, and 
        cells where 1 indicates that the given atlas has the corresponding 
        label.
    """
    labels_dict = {}
    for img_path in img_paths:
        name = lib_clrbrain.get_filename_without_ext(img_path)
        labels_np = load_registered_img(img_path, reg_name=IMG_LABELS)
        # only use pos labels since assume neg labels are merely mirrored
        labels_unique = np.unique(labels_np[labels_np >= 0])
        labels_dict[name] = pd.Series(
            np.ones(len(labels_unique), dtype=int), index=labels_unique)
    df = pd.DataFrame(labels_dict)
    df.sort_index()
    df.to_csv(output_path)
    print("common labels exported to {}".format(output_path))
    return df

def extract_sample_metrics(df, cols):
    """Extract columns from data frame relevant to samples.
    
    Args:
        df: Data frame from which to extract columns.
        cols: Sequence of additional columns to extract.
    
    Returns:
        Data frame view with sample, region, condition, and any additional 
        given columns extracted.
    """
    df_sm = df[
        [AtlasMetrics.SAMPLE.value, AtlasMetrics.REGION.value, 
         AtlasMetrics.CONDITION.value, *cols]]
    return df_sm

def _test_labels_lookup():
    """Test labels reverse dictionary creation and lookup.
    """
    
    # create reverse lookup dictionary
    ref = ontology.load_labels_ref(config.load_labels)
    lookup_id = 15565 # short search path
    #lookup_id = 126652058 # last item
    time_dict_start = time()
    id_dict = ontology.create_aba_reverse_lookup(ref)
    labels_img = load_registered_img(config.filename, reg_name=IMG_LABELS)
    max_labels = np.max(labels_img)
    print("max_labels: {}".format(max_labels))
    time_dict_end = time()
    
    # look up a single ID
    time_node_start = time()
    found = id_dict[lookup_id]
    time_node_end = time()
    print("found {}: {} with parents {}"
          .format(lookup_id, found[ontology.NODE]["name"], 
                  found[ontology.PARENT_IDS]))
    
    # brute-force query
    time_direct_start = time()
    node = ontology.get_node(ref["msg"][0], "id", lookup_id, "children")
    time_direct_end = time()
    #print(node)
    
    print("time to create id_dict (s): {}".format(time_dict_end - time_dict_start))
    print("time to find node (s): {}".format(time_node_end - time_node_start))
    print("time to find node directly (s): {}".format(time_direct_end - time_direct_start))
    
    # get a list of IDs corresponding to each blob
    blobs = np.array([[300, 5000, 3000], [350, 5500, 4500], [400, 6000, 5000]])
    image5d = importer.read_file(config.filename, config.series)
    scaling = importer.calc_scaling(image5d, labels_img)
    ids, coord_scaled = ontology.get_label_ids_from_position(
        blobs[:, 0:3], labels_img, scaling, return_coord_scaled=True)
    print("blob IDs:\n{}".format(ids))
    print("coord_scaled:\n{}".format(coord_scaled))

def _test_region_from_id():
    """Test finding a region by ID in a labels image.
    """
    if len(config.filenames) > 1:
        # unregistered, original labels image; assume that atlas directory has 
        # been given similarly to register fn
        path = os.path.join(config.filenames[1], IMG_LABELS)
        labels_img = sitk.ReadImage(path)
        labels_img = sitk.GetArrayFromImage(labels_img)
        scaling = np.ones(3)
        print("loaded labels image from {}".format(path))
    else:
        # registered labels image and associated experiment file
        labels_img = load_registered_img(config.filename, reg_name=IMG_LABELS)
        if config.filename.endswith(".mhd"):
            img = sitk.ReadImage(config.filename)
            img = sitk.GetArrayFromImage(img)
            image5d = img[None]
        else:
            image5d = importer.read_file(config.filename, config.series)
        scaling = importer.calc_scaling(image5d, labels_img)
        print("loaded experiment image from {}".format(config.filename))
    ref = ontology.load_labels_ref(config.load_labels)
    id_dict = ontology.create_aba_reverse_lookup(ref)
    middle, img_region, region_ids = ontology.get_region_middle(
        id_dict, 16652, labels_img, scaling)
    atlas_label = ontology.get_label(
        middle, labels_img, id_dict, scaling, None, True)
    props, bbox, centroid = get_scaled_regionprops(img_region, scaling)
    print("bbox: {}, centroid: {}".format(bbox, centroid))

def _test_curate_img(path, prefix):
    fixed_img = _load_numpy_to_sitk(path)
    labels_img = load_registered_img(prefix, reg_name=IMG_LABELS, get_sitk=True)
    atlas_img = load_registered_img(prefix, reg_name=IMG_ATLAS, get_sitk=True)
    labels_img.SetSpacing(fixed_img.GetSpacing())
    holes_area = config.register_settings["holes_area"]
    result_imgs = _curate_img(
        fixed_img, labels_img, [atlas_img], inpaint=False, 
        holes_area=holes_area)
    sitk.Show(fixed_img)
    sitk.Show(labels_img)
    sitk.Show(result_imgs[0])
    sitk.Show(result_imgs[1])

def _test_smoothing_metric():
    img = np.zeros((6, 6, 6), dtype=int)
    img[1:5, 1:5, 1:5] = 1
    img_smoothed = morphology.binary_erosion(img)
    print("img:\n{}".format(img))
    print("img_smoothed:\n{}".format(img_smoothed))
    label_smoothing_metric(img, img_smoothed)

def main():
    """Handle registration processing tasks as specified in 
    :attr:``config.register_type``.
    """
    plot_2d.setup_style("default")
    unit_factor = np.power(1000.0, 3)
    
    # name prefix to use a different name from the input files, such as when 
    # registering transposed/scaled images but outputting paths corresponding 
    # to the original image
    if config.prefix is not None:
        print("Formatting registered filenames to match {}"
              .format(config.prefix))
    if config.suffix is not None:
        print("Modifying registered filenames with suffix {}"
              .format(config.suffix))
    flip = False
    if config.flip is not None:
        flip = config.flip[0]
    show = not config.no_show
    size = config.roi_sizes
    if size: size = size[0][:2]
    
    #_test_labels_lookup()
    #_test_region_from_id()
    #_test_curate_img(config.filenames[0], config.prefix)
    #_test_smoothing_metric()
    #os._exit(os.EX_OK)
    
    reg = config.RegisterTypes[config.register_type]
    if config.register_type is None:
        # explicitly require a registration type
        print("Please choose a registration type")
    
    elif reg in (
        config.RegisterTypes.single, config.RegisterTypes.new_atlas):
        # "single", basic registration of 1st to 2nd image, transposing the 
        # second image according to config.plane and config.flip_horiz; 
        # "new_atlas" registers similarly but outputs new atlas files
        new_atlas = reg is config.RegisterTypes.new_atlas
        register(*config.filenames[0:2], plane=config.plane, 
                 flip=flip, name_prefix=config.prefix, new_atlas=new_atlas, 
                 show_imgs=show)
    
    elif reg is config.RegisterTypes.group:
        # groupwise registration, which assumes that the last image 
        # filename given is the prefix and uses the full flip array
        prefix = config.filenames[-1]
        register_group(
            config.filenames[:-1], flip=config.flip, name_prefix=config.prefix, 
            scale=config.rescale, show_imgs=show)
    
    elif reg is config.RegisterTypes.overlays:
        # overlay registered images in each orthogonal plane
        for out_plane in config.PLANE:
            overlay_registered_imgs(
                *config.filenames[0:2], plane=config.plane, 
                flip=flip, name_prefix=config.prefix, 
                out_plane=out_plane)
    
    elif reg is config.RegisterTypes.export_regions:
        # export regions IDs to CSV files
        
        # use ABA levels up through the specified level to collect 
        # sub-regions to the given level
        if config.labels_level is None:
            levels = [None]
        else:
            levels = list(range(config.labels_level + 1))
        ref = ontology.load_labels_ref(config.load_labels)
        labels_ref_lookup = ontology.create_aba_reverse_lookup(ref)
        
        # export region IDs and parents at given level to CSV
        export_region_ids(
            labels_ref_lookup, "region_ids", config.labels_level)
        # export region IDs to network file
        export_region_network(labels_ref_lookup, "region_network")
    
    elif reg is config.RegisterTypes.import_atlas:
        # import original atlas, mirroring if necessary
        import_atlas(config.filename, show=show)
    
    elif reg is config.RegisterTypes.export_common_labels:
        # export common labels
        export_common_labels(config.filenames, config.PATH_COMMON_LABELS)
    
    elif reg is config.RegisterTypes.convert_itksnap_labels:
        # convert labels from ITK-SNAP to CSV format
        df = ontology.convert_itksnap_to_df(config.filename)
        output_path = lib_clrbrain.combine_paths(
            config.filename, ".csv", sep="")
        stats.data_frames_to_csv([df], output_path)
    
    elif reg is config.RegisterTypes.export_metrics_compactness:
        # export data frame with compactness to compare:
        # 1) whole histology image and unsmoothed labels
        # 2) unsmoothed and selected smoothed labels
        # 1st config.filenames element should be atlas import stats, 
        # and 2nd element should be smoothing stats
        
        # load data frames
        df_stats = pd.read_csv(config.filename) # atlas import stats
        df_smoothing = pd.read_csv(config.filenames[1]) # smoothing stats
        
        # compare histo vs unsmoothed labels
        df_stats_base = extract_sample_metrics(
            df_stats, [SmoothingMetrics.COMPACTNESS.value])
        df_smoothing_base = df_smoothing.loc[
            df_smoothing[SmoothingMetrics.FILTER_SIZE.value] == 0]
        df_smoothing_base = extract_sample_metrics(
            df_smoothing_base, [SmoothingMetrics.COMPACTNESS.value])
        df_baseline = pd.concat([df_stats_base, df_smoothing_base])
        df_baseline[config.GENOTYPE_KEY] = (
            "Histo Vs Orig Labels")
        
        # compare unsmoothed vs smoothed labels
        smooth = config.register_settings["smooth"]
        df_smoothing_sm = df_smoothing.loc[
            df_smoothing[SmoothingMetrics.FILTER_SIZE.value] == smooth]
        df_smoothing_sm = extract_sample_metrics(
            df_smoothing_sm, [SmoothingMetrics.COMPACTNESS.value])
        df_smoothing_vs = pd.concat([df_smoothing_base, df_smoothing_sm])
        df_smoothing_vs[config.GENOTYPE_KEY] = (
            "Smoothing")
        
        # compare histo vs smoothed labels
        df_histo_sm = pd.concat([df_stats_base, df_smoothing_sm])
        df_histo_sm[config.GENOTYPE_KEY] = (
            "Vs Smoothed Labels")
        
        # export data frames
        output_path = lib_clrbrain.combine_paths(
            config.filename, "compactness.csv")
        df = pd.concat([df_baseline, df_histo_sm, df_smoothing_vs])
        df[AtlasMetrics.REGION.value] = "all"
        stats.data_frames_to_csv(df, output_path)
    
    elif reg is config.RegisterTypes.plot_smoothing_metrics:
        # plot smoothing metrics
        title = "{} Label Smoothing".format(
            lib_clrbrain.str_to_disp(
                os.path.basename(config.filename).replace(
                    config.PATH_SMOOTHING_METRICS, "")))
        plot_2d.plot_lines(
            config.filename, SmoothingMetrics.FILTER_SIZE.value, 
            (SmoothingMetrics.COMPACTED.value, 
             SmoothingMetrics.DISPLACED.value, 
             SmoothingMetrics.SM_QUALITY.value), 
            ("--", "--", "-"), "Smoothing Filter Size", 
            "Fractional Change", title, size, not config.no_show, "_quality")
        plot_2d.plot_lines(
            config.filename, SmoothingMetrics.FILTER_SIZE.value, 
            (SmoothingMetrics.SA_VOL.value, 
             SmoothingMetrics.LABEL_LOSS.value), 
            ("-", "-"), "Smoothing Filter Size", 
            "Fractional Change", None, size, not config.no_show, "_extras", 
            ("C3", "C4"))
    
    elif reg is config.RegisterTypes.smoothing_peaks:
        # find peak smoothing qualities without label loss for a set 
        # of data frames and output to combined data frame
        dfs = []
        for path in config.filenames:
            df = pd.read_csv(path)
            dfs.append(smoothing_peak(df, 0, None))
        stats.data_frames_to_csv(dfs, "smoothing_peaks.csv")
    
    elif reg in (
        config.RegisterTypes.make_edge_images, 
        config.RegisterTypes.make_edge_images_exp):
        
        # convert atlas or experiment image and associated labels 
        # to edge-detected images; labels can be given as atlas dir from 
        # which labels will be extracted (eg import dir)
        atlas = reg is config.RegisterTypes.make_edge_images
        for img_path in config.filenames:
            make_edge_images(
                img_path, show, atlas, config.suffix, config.load_labels)
    
    elif reg is config.RegisterTypes.reg_labels_to_atlas:
        # register labels to its underlying atlas
        register_labels_to_atlas(config.filename)
    
    elif reg in (
        config.RegisterTypes.merge_atlas_segs, 
        config.RegisterTypes.merge_atlas_segs_exp):
        
        # merge various forms of atlas segmentations
        atlas = reg is config.RegisterTypes.merge_atlas_segs
        merge_atlas_segmentations(
            config.filenames, show=show, atlas=atlas, suffix=config.suffix)
    
    elif reg is config.RegisterTypes.vol_stats:
        # volumes stats
        # TODO: replace volumes/densities function
        ref = ontology.load_labels_ref(config.load_labels)
        labels_ref_lookup = ontology.create_aba_reverse_lookup(ref)
        groups_numeric = None
        groups = {}
        if config.groups is not None:
            groups[config.GENOTYPE_KEY] = [
                config.GROUPS_NUMERIC[geno] for geno in config.groups]
        # should generally leave uncombined for drawn labels to allow 
        # faster level building, where can combine sides
        combine_sides = config.register_settings["combine_sides"]
        volumes_by_id(
            config.filenames, labels_ref_lookup, suffix=config.suffix, 
            unit_factor=unit_factor, groups=groups, 
            max_level=config.labels_level, combine_sides=combine_sides)
    
    elif reg is config.RegisterTypes.make_density_images:
        # make density images
        size = config.roi_sizes
        if size: size = size[0][::-1]
        make_density_images_mp(
            config.filenames, config.rescale, size, config.suffix)
    
    elif reg is config.RegisterTypes.make_subsegs:
        # make sub-segmentations for all images
        for img_path in config.filenames:
            make_sub_segmented_labels(img_path, config.suffix)

    elif reg is config.RegisterTypes.merge_images:
        # take mean of separate experiments from all paths using the 
        # given registered image type, defaulting to experimental images
        suffix = IMG_EXP
        if config.reg_suffixes is not None:
            # use suffix assigned to atlas
            suffix_exp = config.reg_suffixes[config.RegSuffixes.ATLAS]
            if suffix_exp: suffix = suffix_exp
        merge_images(config.filenames, suffix, config.prefix, config.suffix)

    elif reg is config.RegisterTypes.merge_images_channels:
        # combine separate experiments from all paths into separate channels
        merge_images(
            config.filenames, IMG_EXP, config.prefix, config.suffix, 
            fn_combine=None)

    elif reg is config.RegisterTypes.register_reg:
        # register a group of registered images to another image, 
        # such as the atlas to which the images were originally registered
        suffixes = None
        if config.reg_suffixes is not None:
            # get additional suffixes to register the same as for exp img
            suffixes = [config.reg_suffixes[key] 
                        for key, val in config.reg_suffixes.items() 
                        if config.reg_suffixes[key] is not None]
        register_reg(
            *config.filenames[:2], IMG_EXP, suffixes, config.plane, 
            flip, config.prefix, config.suffix, not config.no_show)

    elif reg is config.RegisterTypes.make_labels_level:
        # make a labels image grouped at the given level
        make_labels_level_img(
            config.filename, config.labels_level, config.prefix, show)
    
    elif reg is config.RegisterTypes.labels_diff:
        # generate labels difference images for various measurements 
        # and metrics, using the output from volumes measurements for 
        # drawn labels
        path_df = (SAMPLE_VOLS if config.labels_level is None 
                   else SAMPLE_VOLS_LEVELS) + ".csv"
        metrics = [
            (vols.LabelMetrics.CoefVarNuc.name, np.nanmean), 
            (vols.LabelMetrics.CoefVarNuc.name, np.nanmedian), 
            (vols.LabelMetrics.CoefVarIntens.name, np.nanmean), 
            (vols.LabelMetrics.CoefVarIntens.name, np.nanmedian), 
        ]
        for metric in metrics:
            make_labels_diff_img(
                config.filename, path_df, *metric, config.prefix, show, 
                config.labels_level)

    elif reg is config.RegisterTypes.labels_diff_stats:
        # generate labels difference images for various measurements 
        # from a stats CSV generated by the R clrstats package
        metrics = (
            vols.LabelMetrics.EdgeDistSum.name, 
            vols.LabelMetrics.CoefVarNuc.name, 
            vols.LabelMetrics.CoefVarIntens.name, 
            #vols.MetricCombos.HOMOGENEITY.value[0], 
        )
        for metric in metrics:
            path_df = "{}_{}.csv".format("vols_stats", metric)
            if not os.path.exists(path_df): continue
            make_labels_diff_img(
                config.filename, path_df, "vals.effect", None, config.prefix, 
                show, meas_path_name=metric)
    
    elif reg is config.RegisterTypes.combine_cols:
        # normalize the given columns to original values in a data frame 
        # and combine columns for composite metrics
        df = pd.read_csv(config.filename)
        df = stats.normalize_df(
            df, ["Sample", "Region"], "Condition", "original", 
            (vols.LabelMetrics.VarIntensity.name, 
             vols.LabelMetrics.VarIntensDiff.name, 
             vols.LabelMetrics.EdgeDistSum.name, 
             vols.LabelMetrics.VarNuclei.name), 
            (vols.LabelMetrics.VarIntensity.Volume.name, ))
        df = stats.combine_cols(
            df, (vols.MetricCombos.HOMOGENEITY, ))
        stats.data_frames_to_csv(
            df, lib_clrbrain.insert_before_ext(config.filename, "_norm"))

    elif reg is config.RegisterTypes.zscores:
        # export z-scores for the given metrics to a new data frame 
        # and display as a scatter plot
        
        # generate z-scores
        df = pd.read_csv(config.filename)
        metric_cols = (
            vols.LabelMetrics.VarIntensity.name, 
            vols.LabelMetrics.VarIntensDiff.name,
            vols.LabelMetrics.VarNuclei.name,  
            vols.LabelMetrics.EdgeDistSum.name, 
        )
        extra_cols = (
            "Sample", "Condition", vols.LabelMetrics.Volume.name, 
        )
        df = stats.zscore_df(
            df, "Region", metric_cols, extra_cols, True)
        
        # generate composite score column
        df_comb = stats.combine_cols(
            df, (vols.MetricCombos.HOMOGENEITY, ), np.sum)
        stats.data_frames_to_csv(
            df_comb, 
            lib_clrbrain.insert_before_ext(config.filename, "_zhomogeneity"))
        
        # shift metrics from each condition to separate columns
        conds = np.unique(df["Condition"])
        df = stats.cond_to_cols_df(
            df, ["Sample", "Region"], "Condition", "original", metric_cols)
        path = lib_clrbrain.insert_before_ext(config.filename, "_zscore")
        stats.data_frames_to_csv(df, path)
        
        # display as probability plot
        lims = (-3, 3)
        plot_2d.plot_probability(path, conds, metric_cols, "Volume", 
            xlim=lims, ylim=lims, title="Region Match Z-Scores", 
            fig_size=size, show=show, suffix=None, df=df)
    
    elif reg is config.RegisterTypes.coefvar:
        # export coefficient of variation for the given metrics to a 
        # new data frame and display as a scatter plot
        
        # measure coefficient of variation
        df = pd.read_csv(config.filename)
        cols_orig = df.columns
        combos = (
            vols.MetricCombos.COEFVAR_INTENS, vols.MetricCombos.COEFVAR_NUC
        )
        df = stats.combine_cols(df, combos)
        stats.data_frames_to_csv(
            df, 
            lib_clrbrain.insert_before_ext(config.filename, "_coefvar"))

        metric_cols = (
            vols.LabelMetrics.VarIntensity.name, 
            vols.LabelMetrics.VarIntensMatch.name,
            vols.LabelMetrics.VarNuclei.name,  
            vols.LabelMetrics.EdgeDistSum.name, 
        )
        df = stats.coefvar_df(
            df, ["Region", "Condition"], metric_cols, 
            vols.LabelMetrics.Volume.name)
        
        # shift metrics from each condition to separate columns
        conds = np.unique(df["Condition"])
        df = stats.cond_to_cols_df(
            df, ["Region"], "Condition", "original", metric_cols)
        path = lib_clrbrain.insert_before_ext(config.filename, "_coefvarhom")
        stats.data_frames_to_csv(df, path)
        
        # display as probability plot
        lims = (0, 0.7)
        plot_2d.plot_probability(path, conds, metric_cols, "Volume", 
            xlim=lims, ylim=lims, title="Coefficient of Variation", 
            fig_size=size, show=show, suffix=None, df=df)

    elif reg is config.RegisterTypes.melt_cols:
        # melt columns specified in "groups" using ID columns from 
        # standard atlas metrics
        id_cols=[
            AtlasMetrics.SAMPLE.value, AtlasMetrics.REGION.value, 
            AtlasMetrics.CONDITION.value]
        df = stats.melt_cols(
            pd.read_csv(config.filename), id_cols, config.groups, 
            config.GENOTYPE_KEY)
        stats.data_frames_to_csv(
            df, lib_clrbrain.insert_before_ext(config.filename, "_melted"))

if __name__ == "__main__":
    print("Clrbrain image registration")
    from clrbrain import cli
    cli.main(True)
    main()
