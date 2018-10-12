#!/bin/bash
# Image registration
# Author: David Young, 2017, 2018
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
import json
import multiprocessing as mp
from collections import OrderedDict
from pprint import pprint
import shutil
from time import time
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
from clrbrain import plot_2d
from clrbrain import plot_3d
from clrbrain import stats

IMG_ATLAS = "atlasVolume.mhd"
IMG_LABELS = "annotation.mhd"
IMG_EXP = "exp.mhd"
IMG_GROUPED = "grouped.mhd"
IMG_ATLAS_TEMPLATE = "atlasVolumeTemplate.mhd"
IMG_BORDERS = "borders.mhd"

NODE = "node"
PARENT_IDS = "parent_ids"
MIRRORED = "mirrored"
RIGHT_SUFFIX = " (R)"
LEFT_SUFFIX = " (L)"
ABA_ID = "id"
ABA_PARENT = "parent_structure_id"
ABA_LEVEL = "st_level"
ABA_CHILDREN = "children"

SMOOTHING_MODES=("opening", "gaussian")
SMOOTHING_METRIC_MODES=("vol", "area_edt", "area_radial")
_SIGNAL_THRESHOLD = 0.01

def _reg_out_path(file_path, reg_name):
    """Generate a path for a file registered to another file.
    
    Args:
        file_name: Full path of file registered to.
        reg_name: Filename alone of registered file.
    
    Returns:
        Full path with the registered filename including extension at the end.
    """
    ext = lib_clrbrain.get_filename_ext(file_path)
    file_path_base = importer.filename_to_base(
        file_path, config.series, ext=ext)
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
    """Get the bounding box for the first object within an image.
    
    Since there does not appear to be a guarantee about the order of 
    objects found in :func:``measure.regionprops``, this bbox method 
    should only be used when only one object is expected.
    
    Args:
        img_np: Image as a Numpy array.
        threshold: Threshold level; defaults to 10. If None, assume 
            ``img_np`` is already binary.
    
    Returns:
        Bounding box of the first object in the image.
    """
    thresholded = img_np
    if threshold is not None:
        # threshold the image, removing any small object
        thresholded = img_np > threshold
        thresholded = morphology.remove_small_objects(thresholded, 200)
    # make labels for foreground and get label properties
    labels_props = measure.regionprops(measure.label(thresholded))
    if len(labels_props) < 1:
        return None
    labels_bbox = labels_props[0].bbox
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
            img_np[slices] = 0
            slices[axis] = slice(bound_abs[1], None)
            img_np[slices] = 0
            print("truncated axis {} outside of bounds {}"
                  .format(axis, bound_abs))
        axis += 1
    return img_np

def _mirror_planes(img_np, start, mirror_mult=1, resize=True):
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
    
    Returns:
        The mirrored image in Numpy format.
    """
    if resize:
        shape = img_np.shape
        shape_resized = np.copy(shape)
        shape_resized[0] = start * 2
        img_resized = np.zeros(shape_resized, dtype=img_np.dtype)
        img_resized[:shape[0]] = img_np
        print("original shape: {}, new shape: {}".format(shape, shape_resized))
        img_np = img_resized
    tot_planes = len(img_np)
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
    return img_np

def _mirror_labels(img, img_ref, extent=None, expand=None, rotate=None, 
                   smooth=True):
    """Mirror labels across sagittal midline and add lateral edges.
    
    Assume that the image is in sagittal sections and consists of only one 
    hemisphere, empty from the far z planes toward the middle but not 
    necessarily the exact middle of the image. Find the first plane that 
    doesn't have any intensity values and set this position as the mirror 
    plane.
    
    Also assume that the lateral edges of the image are also missing. Build 
    edges that match the size of the reference image on one side and mirror 
    over to the other side.
    
    Args:
        img: Labels image in SimpleITK format.
        img_ref: Reference atlas image in SimpleITK format.
        extent: Tuple (start, end) fractions demarcating the region to mirror; 
            defaults to None, in which case defaults will be found based on 
            first found non-zero planes at boundaries. If either value 
            within the tuple is None, the corresponding default will be 
            found.
        expand: Tuple of 
            ((x_pixels_start, end), (y, ...), ...), (next_region, ...)) 
            specifying slice boundaries for regions to expand the labels to 
            the size of the atlas. Defaults to None.
        rotate: Tuple of ((angle0, axis0), ...) by which to rotate the 
            labels. Defaults to None.
        smooth: True if labels should be smoothed; defaults to True.
    
    Returns:
        Tuple of ``img_np``, ``(extendi, mirrori)``, where ``img_np`` is 
        the mirrored image in Numpy format, and ``(extendi, mirrori)`` is 
        a tuple of the indices at which the edge was extended and the 
        image was mirrored, respectively.
    """
    # TODO: check to make sure values don't get wrapped around if np.int32
    # max value is less than data max val
    img_np = sitk.GetArrayFromImage(img).astype(np.int32)
    img_ref_np = sitk.GetArrayFromImage(img_ref).astype(np.int32)
    tot_planes = len(img_np)
    
    # extend lateral planes: make first plane with signal start even earlier
    extendi = 0
    if extent is None or extent[0] is None:
        # find the first non-zero plane
        for plane in img_np:
            if not np.allclose(plane, 0):
                print("found first non-zero plane at {}".format(extendi))
                break
            extendi += 1
    else:
        # based on settings
        extendi = int(extent[0] * tot_planes)
    
    # find the bounds of the reference image in the given plane and resize 
    # the corresponding section of the labels image to the bounds of the 
    # reference image in the next plane closer to the edge, essentially 
    # extending the last edge plane of the labels image
    plane_region = None
    while extendi >= 0:
        #print("plane_region max: {}".format(np.max(plane_region)))
        bbox = _get_bbox(img_ref_np[extendi])
        if bbox is None:
            break
        shape, slices = plot_3d.get_bbox_region(bbox)
        if plane_region is None:
            plane_region = img_np[extendi, slices[0], slices[1]]
            # remove ventricular space using empirically determined selem, 
            # which appears to be very sensitive to radius since values above 
            # or below lead to square shaped artifact along outer sample edges
            plane_region = morphology.closing(
                plane_region, morphology.square(12))
        else:
            # assume that the reference image background is about < 10, the 
            # default threshold
            plane_region = transform.resize(
                plane_region, shape, preserve_range=True, order=0, 
                anti_aliasing=True, mode="reflect")
            #print("plane_region max: {}".format(np.max(plane_region)))
            img_np[extendi, slices[0], slices[1]] = plane_region
        extendi -= 1
    
    if expand:
        # expand selected regions
        for expand_limits in expand:
            # get region from slices specified by tuple of (start, end) pixels
            expand_slices = tuple(
                slice(*limits) for limits in expand_limits[::-1])
            region = img_np[expand_slices]
            region_ref = img_ref_np[expand_slices]
            for extendi in range(len(region_ref)):
                # find bounding boxes for labels and atlas within region
                bbox = _get_bbox(region[extendi], 0) # assume pos labels region
                shape, slices = plot_3d.get_bbox_region(bbox)
                plane_region = region[extendi, slices[0], slices[1]]
                bbox_ref = _get_bbox(region_ref[extendi])
                shape_ref, slices_ref = plot_3d.get_bbox_region(bbox_ref)
                # expand bounding box region of labels to that of atlas
                plane_region = transform.resize(
                    plane_region, shape_ref, preserve_range=True, order=0, 
                    anti_aliasing=True, mode="reflect")
                region[extendi, slices_ref[0], slices_ref[1]] = plane_region
    
    # find approximate midline by locating the last zero plane from far edge 
    # at which to start mirroring across midline
    mirrori = tot_planes
    for plane in img_np[::-1]:
        if not np.allclose(plane, 0):
            print("found last zero plane from far border at {}".format(mirrori))
            break
        mirrori -= 1
    
    if rotate:
        # mirror labels with original values in case rotation will cause 
        # some labels to be cut off, then rotate for each specified axis
        for i in range(mirrori, tot_planes):
            img_np[i] = img_np[mirrori - 1]
        for rot in rotate:
            img_np = plot_3d.rotate_nd(img_np, rot[0], rot[1], order=0)
    
    # reset mirroring index baed either on previously found index or fractional 
    # profile setting
    if extent is not None and extent[1] is not None:
        mirrori = int(extent[1] * tot_planes)
    
    borders_img_np = None
    if smooth:
        # minimize jaggedness in labels, often seen outside of the original 
        # orthogonal direction, using pre-mirrored slices only since rest will 
        # be overwritten
        img_smoothed = img_np[:mirrori]
        img_smoothed_orig = np.copy(img_smoothed)
        smooth_labels(img_smoothed)
        
        # calculate smoothing metric with borders image
        borders, _, _ = label_smoothing_metric(
            img_smoothed_orig, img_smoothed)
        shape = list(borders.shape)
        shape[0] = img_np.shape[0]
        borders_img_np = np.zeros(shape, dtype=np.int32)
        borders_img_np[:mirrori] = borders
        borders_img_np = _mirror_planes(borders_img_np, mirrori, -1)
    
    # check that labels will fit in integer type
    lib_clrbrain.printv(
        "type: {}, max: {}, max avail: {}".format(
            img_np.dtype, np.max(img_np), np.iinfo(img_np.dtype).max))
    
    # mirror and check for label loss
    print("total labels before reflection: {}".format(np.unique(img_np).size))
    img_np = _mirror_planes(img_np, mirrori, -1)
    print("total labels after reflection up to set midline ({}): {}"
          .format(mirrori, np.unique(img_np[:mirrori]).size))
    print("total final labels: {}".format(np.unique(img_np).size))
    return img_np, (extendi, mirrori), borders_img_np

def replace_sitk_with_numpy(img_sitk, img_np):
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    img_sitk_back = sitk.GetImageFromArray(img_np)
    img_sitk_back.SetSpacing(spacing)
    img_sitk_back.SetOrigin(origin)
    return img_sitk_back

def smooth_labels(labels_img_np, filter_size=3, mode=SMOOTHING_MODES[0]):
    """Smooth each label within labels annotation image.
    
    Labels images created in one orthogonal direction may have ragged, 
    high-frequency edges when viewing in the other orthogonal directions. 
    Smooth these edges by applying a filter to each label.
    
    Args:
        labels_img_np: Labels image as a Numpy array.
        filter_size: Structuring element or kernel size.
        mode: One of :const:``SMOOTHING_MODES``, where ``opening`` applies 
            a morphological opening filter unless the size is severely 
            reduced, in which case a closing filter is applied instead; and 
            ``gaussian`` applies a Gaussian blur.
    """
    print("Smoothing labels with filter size of {}, mode {}"
          .format(filter_size, mode))
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
        region = labels_img_np[slices]
        label_mask_region = region == label_id
        region_size = np.sum(label_mask_region)
        if region_size == 0:
            print("no pixels to smooth, skipping")
            continue
        
        # smoothing based on mode
        region_size_smoothed = 0
        if mode == SMOOTHING_MODES[0]:
            # smooth region with opening filter, changing to closing filter 
            # if region would be lost or severely reduced
            selem = morphology.ball(filter_size)
            opened = morphology.binary_opening(label_mask_region, selem)
            region_size_smoothed = np.sum(opened)
            size_ratio = region_size_smoothed / region_size
            if size_ratio < 0.5:
                print("largest region would be lost or too small "
                      "(ratio {}), will use closing filter instead"
                      .format(size_ratio))
                opened = morphology.binary_closing(label_mask_region, selem)
                region_size_smoothed = np.sum(opened)
            
            # fill empty spaces with closest surrounding labels
            region = plot_3d.in_paint(region, label_mask_region)
        elif mode == SMOOTHING_MODES[1]:
            # smoothing with gaussian blur
            opened = filters.gaussian(
                label_mask_region, filter_size, mode="nearest", 
                multichannel=False).astype(bool)
            region_size_smoothed = np.sum(opened)
        
        region[opened] = label_id
        labels_img_np[slices] = region
        print("changed num of pixels from {} to {}"
              .format(region_size, region_size_smoothed))
    
    # show label loss metric
    print("\nMeasuring label loss:")
    label_ids_smoothed = np.unique(labels_img_np)
    print("number of labels changed from {} to {}"
          .format(label_ids.size, label_ids_smoothed.size))
    labels_lost = label_ids[np.isin(label_ids, label_ids_smoothed, invert=True)]
    print("labels lost during smoothing: {}".format(labels_lost))
    for lost in labels_lost:
        region_lost = labels_img_np_orig[labels_img_np_orig == lost]
        print("size of lost label {}: {}".format(lost, region_lost.size))
    
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
        weighted_size_ratio += size_smoothed / size_orig * size_orig
        tot_pxs += size_orig
    weighted_size_ratio /= tot_pxs
    print("\nVolume ratio (smoothed:orig) weighted by orig size: {}\n"
          .format(weighted_size_ratio))

def label_smoothing_metric(orig_img_np, smoothed_img_np, filter_size=4, 
                           penalty_wt=1.0, mode=SMOOTHING_METRIC_MODES[1]):
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
    
    Args:
        original_img_np: Unsmoothed labels image as Numpy array.
        smoothing_img_np: Smoothed labels image as Numpy array, which 
            should be of the same shape as ``original_img_np``.
        filter_size: Structuring element size for determining the filled, 
            broad volume of each label. Defaults to 4. Larger sizes 
            favor greater smoothing in the final labels.
        penalty_wt: Weighting factor for the penalty term. For ``vol`` 
            mode, larger  values favor labels that remain within their 
            original bounds. For ``area`` mode, this value is used as a 
            denominator for pixel perimeter displacement, where larger values 
            tolerate more displacement. Defaults to 1.0.
        mode: One of :const:``SMOOTHING_METRIC_MODES`` (see above for 
            description of the modes).
    
    Returns:
        Tuple of ``borders_img_np``, a Numpy array of the same same as 
        ``original_img_np`` except with an additional channel dimension at 
        the end, where channel 0 contains the broad borders of the 
        original image's labels, and channel 1 is that of the smoothed image; 
        ``tot_metric``, the smoothing metric as a float value; and 
        ``pd``, the metric components as a Pandas data frame.
    """
    print("Calculating smoothing metrics with filter size of {}, "
          "penalty weighting factor of {}".format(filter_size, penalty_wt))
    start_time = time()
    
    # prepare roughness images to track global overlap
    shape = list(orig_img_np.shape)
    roughs = [np.zeros(shape, dtype=np.int8)]
    roughs.append(np.copy(roughs[0]))
    
    # prepare borders image with channel for each set of borders
    shape.append(2)
    borders_img_np = np.zeros(shape, dtype=np.int32)
    
    # pepare labels and default selem used to find "broad volume"
    label_ids = np.unique(orig_img_np)
    
    def update_borders_img(borders, slices, label_id, channel):
        nonlocal borders_img_np
        borders_region = borders_img_np[slices]
        borders_region[borders, channel] = label_id
    
    def broad_borders(img_np, slices, label_id, channel, rough_img_np):
        # use closing filter to approximate volume encompassing rough edges
        # get region, skipping if no region left
        region = img_np[slices]
        label_mask_region = region == label_id
        filtered = morphology.binary_closing(label_mask_region, selem)
        rough_img_np[slices] = np.add(
            rough_img_np[slices], filtered.astype(np.int8))
        filtered_border = plot_3d.perimeter_nd(filtered)
        update_borders_img(filtered_border, slices, label_id, channel)
        return label_mask_region, filtered
    
    def surface_area(img_np, slices, label_id, rough_img_np):
        # use closing filter to approximate volume encompassing rough edges
        # get region, skipping if no region left
        region = img_np[slices]
        label_mask_region = region == label_id
        borders = plot_3d.perimeter_nd(label_mask_region)
        rough_img_np[slices] = np.add(
            rough_img_np[slices], borders.astype(np.int8))
        return label_mask_region, borders
    
    def gaus(distances):
        if penalty_wt is not None:
            distances = filters.gaussian(
                distances, sigma=penalty_wt, multichannel=False, 
                preserve_range=True)
        return distances
    
    tot_metric = 0
    tot_size = 0
    padding = 2 if filter_size is None else 2 * filter_size
    pxs = {}
    cols = ("label_id", "pxs_reduced", "pxs_expanded", "size_orig")
    for label_id in label_ids:
        # calculate metric for each label
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
        elif mode in SMOOTHING_METRIC_MODES[1:3]:
            # "area": measure surface area
            mask_orig, borders_orig = surface_area(
                orig_img_np, slices, label_id, roughs[0])
            update_borders_img(borders_orig, slices, label_id, 0)
            mask_smoothed, borders_smoothed = surface_area(
                smoothed_img_np, slices, label_id, roughs[1])
            # reduction in surface area (compaction)
            pxs_reduced = np.sum(borders_orig) - np.sum(borders_smoothed)
            # normalize to original surface area
            size_orig = np.sum(borders_orig)
            tot_size += size_orig
            # expansion past original borders (displacement penalty), 
            # using filter to give buffer around irregular borders to 
            # offset distances there
            dist_to_orig, indices, borders_orig_filled = (
                plot_3d.borders_distance(
                    borders_orig, borders_smoothed, mask_orig=mask_orig, 
                    filter_size=filter_size))
            if mode == SMOOTHING_METRIC_MODES[1]:
                # "area_edt": displacement determined using distance transform 
                # from shifted to original borrders
                if filter_size is not None:
                    # find distances around the original borders to show 
                    # distances potentially in appropriately compacted areas
                    update_borders_img(borders_orig_filled, slices, label_id, 1)
                dist_to_orig = gaus(dist_to_orig)
                dist_to_orig[dist_to_orig < 0] = 0
            elif mode == SMOOTHING_METRIC_MODES[2]:
                # "area_radial": displacement determined using difference 
                # in radial distances from center to get signed distances
                region = orig_img_np[slices]
                props = measure.regionprops((region == label_id).astype(np.int))
                centroid = props[0].centroid
                radial_dist_orig = plot_3d.radial_dist(borders_orig, centroid)
                radial_dist_smoothed = plot_3d.radial_dist(
                    borders_smoothed, centroid)
                # assuming that high freq regions are those likely targeted 
                # for compaction, smooth out those distances and use only 
                # pos vals to avoid cancelation
                radial_diff = plot_3d.radial_dist_diff(
                    radial_dist_orig, radial_dist_smoothed, indices)
                radial_diff = gaus(radial_diff)
                dist_to_orig = np.abs(radial_diff)
            # take square root of distances, first rounding numbers between 
            # 0-1 in case of Gaussian filtering to avoid inflating numbers
            mask = np.logical_and(
                np.greater(dist_to_orig, 0), np.less(dist_to_orig, 1))
            dist_to_orig[mask] = np.round(dist_to_orig[mask])
            pxs_expanded = np.sum(np.sqrt(dist_to_orig))
            sa_to_vol = (np.sum(borders_smoothed) / np.sum(mask_smoothed)
                         - np.sum(borders_orig) / np.sum(mask_orig))
            pxs.setdefault("SA_to_vol_diff", []).append(sa_to_vol)
        else:
            raise TypeError("no metric of mode {}".format(mode))
        
        metric = pxs_reduced - pxs_expanded
        tot_metric += metric
        vals = (label_id, pxs_reduced, pxs_expanded, size_orig)
        for col, val in zip(cols, vals):
            pxs.setdefault(col, []).append(val)
        print("pxs_reduced: {}, pxs_expanded: {}, metric: {}"
              .format(pxs_reduced, pxs_expanded, metric))
    roughs_metric = None
    if tot_size > 0:
        # normalize to total original label foreground
        tot_metric /= tot_size
        if mode == SMOOTHING_METRIC_MODES[0]:
            # find only amount of overlap, subtracting label count itself
            roughs = [rough - 1 for rough in roughs]
        roughs_metric = [np.sum(rough) / tot_size for rough in roughs]
    
    print()
    df = pd.DataFrame(pxs)
    print(df.to_csv(sep="\t", index=False))
    print("\nTotal foreground pxs: {}".format(tot_size))
    if roughs_metric is not None:
       print("Roughness original: {}".format(roughs_metric[0]))
       print("Roughness smoothed: {}".format(roughs_metric[1]))
       print("roughness diff: {}".format(roughs_metric[0] - roughs_metric[1]))
    print("Smoothing metric: {}".format(tot_metric))
    print("time elapsed for smoothing metric (s): {}"
          .format(time() - start_time))
    return borders_img_np, tot_metric, df

def transpose_img(img_sitk, plane, rotate=False, target_size=None):
    """Transpose a SimpleITK format image via Numpy and re-export to SimpleITK.
    
    Args:
        img_sitk: Image in SimpleITK format.
        plane: One of :attr:``config.PLANES`` elements, specifying the 
            planar orientation in which to transpose the image. The current 
            orientation is taken to be "xy".
        rotate: Rotate the final output image by 180 degrees; defaults to False.
        target_size: Size of target image, typically one to which ``img_sitk`` 
            will be registered, in (x,y,z, SimpleITK standard) ordering.
    
    Returns:
        Transposed image in SimpleITK format.
    """
    img = sitk.GetArrayFromImage(img_sitk)
    img_dtype = img.dtype
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    transposed = img
    if plane is not None and plane != config.PLANE[0]:
        # swap z-y to get (y, z, x) order for xz orientation
        transposed = np.swapaxes(transposed, 0, 1)
        # sitk convention is opposite of numpy with (x, y, z) order
        spacing = lib_clrbrain.swap_elements(spacing, 1, 2)
        origin = lib_clrbrain.swap_elements(origin, 1, 2)
        if plane == config.PLANE[1]:
            # rotate
            if transposed.ndim >=4: # multichannel
                transposed = transposed[..., ::-1, :]
            else:
                transposed = transposed[..., ::-1]
            transposed = np.swapaxes(transposed, 1, 2)
            spacing = lib_clrbrain.swap_elements(spacing, 0, 1)
            origin = lib_clrbrain.swap_elements(origin, 0, 1)
        elif plane == config.PLANE[2]:
            # swap new y-x to get (x, z, y) order for yz orientation
            transposed = np.swapaxes(transposed, 0, 2)
            spacing = lib_clrbrain.swap_elements(spacing, 0, 2)
            origin = lib_clrbrain.swap_elements(origin, 0, 2)
            # rotate
            transposed = np.swapaxes(transposed, 1, 2)
            spacing = lib_clrbrain.swap_elements(spacing, 0, 1)
        if plane == config.PLANE[1] or plane == config.PLANE[2]:
            # flip upside-down
            transposed[:] = np.flipud(transposed[:])
        else:
            transposed[:] = transposed[:]
    if rotate:
        # rotate the final output image by 180 deg
        # TODO: need to change origin? make axes accessible (eg (0, 2) for 
        # horizontal rotation)
        transposed = np.rot90(transposed, 2, (1, 2))
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
                multichannel=False, anti_aliasing=True, 
                order=0).astype(img_dtype)
            spacing = np.divide(spacing, rescale)
        # casted back since transpose changes data type even when 
        # preserving range
        print(transposed.dtype, np.min(transposed), np.max(transposed))
    transposed = sitk.GetImageFromArray(transposed)
    transposed.SetSpacing(spacing)
    transposed.SetOrigin(origin)
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

def _curate_img(fixed_img, labels_img, imgs=None, inpaint=True, carve=True):
    """Curate images by in-painting where corresponding pixels are present in 
    fixed image but not labels or other images and removing pixels 
    present in those images but not the fixed image.
    
    Args:
        fixed_img: Image in SimpleITK format by which to curate other images.
        labels_img: Labels image in SimpleITK format, used to determine 
            missing pixels and measure overlap.
        imgs: Array of additional images to curate corresponding pixels 
            as those curated in ``labels_img``. Defaults to None.
    
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
            result_img_np[fixed_img_np <= thresh] = 0
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

def _measure_overlap_combined_labels(fixed_img, labels_img):
    # check overlap based on combined labels images; should be 1.0 by def 
    # when using labels img to curate fixed img
    result_img_np = sitk.GetArrayFromImage(labels_img)
    result_img_np[result_img_np != 0] = 2
    result_img_for_overlap = replace_sitk_with_numpy(
        labels_img, result_img_np)
    print("\nDSC compared with combined labels:")
    measure_overlap(
        fixed_img, result_img_for_overlap, transformed_thresh=1)

def _transform_labels(transformix_img_filter, labels_img, settings, 
                      truncate=False):
    if truncate:
        # truncate ventral and posterior portions since variable 
        # amount of tissue or quality of imaging in these regions
        labels_img_np = sitk.GetArrayFromImage(labels_img)
        _truncate_labels(labels_img_np, *settings["truncate_labels"])
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

def match_atlas_labels(img_atlas, img_labels):
    mirror = config.register_settings["labels_mirror"]
    img_borders = None
    if mirror:
        # mirror and truncate labels for labels for only half the brain, 
        # such as for ABA E18pt5, unlike P56
        expand = config.register_settings["expand_labels"]
        rotate = config.register_settings["rotate"]
        img_labels_np, mirror_indices, borders_img_np = _mirror_labels(
            img_labels, img_atlas, mirror, expand, rotate)
        img_labels = replace_sitk_with_numpy(img_labels, img_labels_np)
        img_atlas_np = sitk.GetArrayFromImage(img_atlas)
        if rotate:
            for rot in rotate:
                img_atlas_np = plot_3d.rotate_nd(img_atlas_np, rot[0], rot[1])
        img_atlas_np = _mirror_planes(img_atlas_np, mirror_indices[1])
        img_atlas = replace_sitk_with_numpy(img_atlas, img_atlas_np)
        
        if borders_img_np is not None:
            img_borders = replace_sitk_with_numpy(img_labels, borders_img_np)
            img_borders = transpose_img(img_borders, config.plane, False)
    
    # transpose to given plane
    img_atlas = transpose_img(img_atlas, config.plane, False)
    img_labels = transpose_img(img_labels, config.plane, False)
    return img_atlas, img_labels, img_borders

def import_atlas(atlas_dir, show=True):
    # load atlas and corresponding labels
    img_atlas = sitk.ReadImage(os.path.join(atlas_dir, IMG_ATLAS))
    img_labels = sitk.ReadImage(os.path.join(atlas_dir, IMG_LABELS))
    
    #img_labels_np = None
    img_atlas, img_labels, img_borders = match_atlas_labels(img_atlas, img_labels)
    
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
    
    _measure_overlap_combined_labels(img_atlas, img_labels)
    
    if show:
       sitk.Show(img_atlas)
       sitk.Show(img_labels)
    
    # write images with atlas saved as Clrbrain/Numpy format to 
    # allow opening as an image within Clrbrain alongside the labels image
    target_dir = atlas_dir + "_import"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    name_prefix = os.path.join(target_dir, os.path.basename(atlas_dir)) + ".czi"
    imgs_write = {
        IMG_ATLAS: img_atlas, IMG_LABELS: img_labels, IMG_BORDERS: img_borders}
    for suffix in imgs_write.keys():
        img = imgs_write[suffix]
        if img is None: continue
        out_path = _reg_out_path(name_prefix, suffix)
        sitk.WriteImage(img, out_path, False)
        # copy metadata file to allow opening images from bare suffix name, 
        # such as when this atlas becomes the new atlas for registration
        shutil.copy(out_path, os.path.join(target_dir, suffix))
    detector.resolutions = [img_atlas.GetSpacing()[::-1]]
    img_ref_np = sitk.GetArrayFromImage(img_atlas)
    img_ref_np = img_ref_np[None]
    importer.save_np_image(img_ref_np, name_prefix, 0)

def register(fixed_file, moving_file_dir, plane=None, flip=False, 
             show_imgs=True, write_imgs=True, name_prefix=None, 
             new_atlas=False):
    """Registers two images to one another using the SimpleElastix library.
    
    Args:
        fixed_file: The image to register, given as a Numpy archive file to 
            be read by :importer:`read_file`.
        moving_file_dir: Directory of the atlas images, including the 
            main image and labels. The atlas was chosen as the moving file
            since it is likely to be lower resolution than the Numpy file.
        plane: Planar orientation to which the atlas will be transposed, 
            considering the atlas' original plane as "xy".
        flip: True if the moving files (does not apply to fixed file) should 
            be flipped/rotated; defaults to False.
        show_imgs: True if the output images should be displayed; defaults to 
            True.
        write_imgs: True if the images should be written to file; defaults to 
            False.
        name_prefix: Path with base name where registered files are located; 
            defaults to None, in which case the fixed_file path will be used.
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
    moving_img, labels_img, _ = match_atlas_labels(moving_img, labels_img)
    
    # basic info from images just prior to SimpleElastix filtering for 
    # registration; to view raw images, show these images rather than merely 
    # turning all iterations to 0 since simply running through the filter 
    # will alter images
    print("fixed image from {} (type {}):\n{}".format(
        fixed_file, fixed_img.GetPixelIDTypeAsString(), fixed_img))
    print("moving image from {} (type {}):\n{}".format(
        moving_file, moving_img.GetPixelIDTypeAsString(), moving_img))
    
    # set up SimpleElastix filter
    elastix_img_filter = sitk.ElastixImageFilter()
    elastix_img_filter.SetFixedImage(fixed_img)
    elastix_img_filter.SetMovingImage(moving_img)
    
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
        move_pts_path = os.path.join(os.path.dirname(name_prefix), "mov_pts.txt")
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
    # turn off to avoid overshooting the interpolation for the labels image 
    # (see Elastix manual section 4.3)
    transform_param_map[-1]["FinalBSplineInterpolationOrder"] = ["0"]
    transformix_img_filter = sitk.TransformixImageFilter()
    transformix_img_filter.SetTransformParameterMap(transform_param_map)
    
    def make_labels(truncate):
        nonlocal transformed_img
        img = _transform_labels(
            transformix_img_filter, labels_img, settings, truncate=truncate)
        print(img.GetSpacing())
        # WORKAROUND: labels img may be more rounded than transformed moving 
        # img for some reason; assume transformed labels and moving image 
        # should match exactly, so replace labels with moving image's 
        # transformed spacing
        img.SetSpacing(transformed_img.GetSpacing())
        print(img.GetSpacing())
        print(fixed_img_orig.GetSpacing(), transformed_img.GetSpacing())
        img, transformed_img = _curate_img(
            fixed_img_orig, img, imgs=[transformed_img], inpaint=new_atlas)
        return img
    
    labels_img_full = make_labels(False)
    labels_img = labels_img_full if new_atlas else make_labels(True)
    
    if show_imgs:
        # show individual SimpleITK images in default viewer
        sitk.Show(fixed_img)
        #sitk.Show(moving_img)
        sitk.Show(transformed_img)
        sitk.Show(labels_img)
    
    if write_imgs:
        # write atlas and labels files, transposed according to plane setting
        imgs_names = (IMG_EXP, IMG_ATLAS, IMG_LABELS)
        imgs_write = [fixed_img, transformed_img, labels_img]
        if new_atlas:
            imgs_names = (IMG_ATLAS, IMG_LABELS)
            imgs_write = [transformed_img, labels_img]
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
    
    # overlap stats
    print("DSC compared with atlas")
    measure_overlap(
        fixed_img_orig, transformed_img, 
        transformed_thresh=settings["atlas_threshold"])
    
    _measure_overlap_combined_labels(fixed_img_orig, labels_img_full)
    
    # show overlays last since blocks until fig is closed
    #_show_overlays(imgs, translation, fixed_file, None)
    print("time elapsed for single registration (s): {}"
          .format(time() - start_time))

def measure_overlap(fixed_img, transformed_img, fixed_thresh=None, 
                    transformed_thresh=None):
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    '''
    # mean Dice Similarity Coefficient (DSC) of labeled regions;
    # not really applicable here since don't have moving labels;
    # fixed_img is 64-bit float (double), while transformed_img is 32-bit
    overlap_filter.Execute(sitk.Cast(fixed_img, sitk.sitkFloat32), transformed_img)
    mean_region_dsc = overlap_filter.GetDiceCoefficient()
    '''
    # Dice Similarity Coefficient (DSC) of total volume by applying 
    # simple binary mask for estimate of background vs foreground
    if not fixed_thresh:
        fixed_thresh = float(
            filters.threshold_mean(sitk.GetArrayFromImage(fixed_img)))
    if not transformed_thresh:
        transformed_thresh = float(
            filters.threshold_mean(sitk.GetArrayFromImage(transformed_img)))
    print("measuring overlap with thresholds of {} (fixed) and {} (transformed)"
          .format(fixed_thresh, transformed_thresh))
    # similar to simple binary thresholding via Numpy
    fixed_binary_img = sitk.BinaryThreshold(fixed_img, fixed_thresh)
    transformed_binary_img = sitk.BinaryThreshold(
        transformed_img, transformed_thresh)
    overlap_filter.Execute(fixed_binary_img, transformed_binary_img)
    #sitk.Show(fixed_binary_img)
    #sitk.Show(transformed_binary_img)
    total_dsc = overlap_filter.GetDiceCoefficient()
    print("Total DSC: {}".format(total_dsc))
    return total_dsc

def measure_overlap_labels(fixed_img, transformed_img):
    # mean Dice Similarity Coefficient (DSC) of labeled regions
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    # fixed_img is 64-bit float (double), while transformed_img is 32-bit
    overlap_filter.Execute(fixed_img, transformed_img)
    mean_region_dsc = overlap_filter.GetDiceCoefficient()
    print("Mean regional (label-by-label) DSC: {}".format(mean_region_dsc))
    return mean_region_dsc

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
        plane = labels_img[slices]
        if not np.allclose(plane, 0):
            print("found first non-zero plane at i of {}".format(i))
            break
    
    # crop image if a region of empty planes is found at the start of the axis
    img_crop = img_np
    if i < shape[axis]:
        slices = [slice(None)] * img_np.ndim
        if eraser is None:
            slices[axis] = slice(i, shape[axis])
            img_crop = img_crop[slices]
            print("cropped image from shape {} to {}"
                  .format(shape, img_crop.shape))
        else:
            slices[axis] = slice(0, i)
            img_crop[slices] = eraser
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
        name_prefix: Path with base name where registered files are located; 
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
        img_file = importer.get_transposed_image_path(
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
        labels_img = load_registered_img(img_files[i], reg_name=IMG_LABELS)
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
        region = img_mean[slices]
        region_template = img_np_template[slices]
        mask = region < carve_threshold
        region[mask] = region_template[mask]
    img_raw = replace_sitk_with_numpy(transformed_img, img_mean)
    
    # carve groupwise registered image if given thresholds
    imgs_to_show = []
    imgs_to_show.append(img_raw)
    holes_area = settings["holes_area"]
    if carve_threshold and holes_area:
        img_mean, img_mean_unfilled = plot_3d.carve(
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
    moving_sitk = transpose_img(moving_sitk, plane, flip)
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
    """
    reg_img_path = _reg_out_path(img_path, reg_name)
    print("loading registered image from {}".format(reg_img_path))
    if not os.path.exists(reg_img_path):
        raise FileNotFoundError(
            "{} registered image file not found".format(reg_img_path))
    reg_img = sitk.ReadImage(reg_img_path)
    if replace is not None:
        reg_img = replace_sitk_with_numpy(reg_img, replace)
        sitk.WriteImage(reg_img, reg_img_path, False)
        print("replaced {} with current registered image".format(reg_img_path))
    if get_sitk:
        return reg_img
    return sitk.GetArrayFromImage(reg_img)

def load_labels_ref(path):
    labels_ref = None
    with open(path, "r") as f:
        labels_ref = json.load(f)
        #pprint(labels_ref)
    return labels_ref

def create_reverse_lookup(nested_dict, key, key_children, id_dict=OrderedDict(), 
                          parent_list=None):
    """Create a reveres lookup dictionary with the values of the original 
    dictionary as the keys of the new dictionary.
    
    Each value of the new dictionary is another dictionary that contains 
    "node", the dictionary with the given key-value pair, and "parent_ids", 
    a list of all the parents of the given node. This entry can be used to 
    track all superceding dictionaries, and the node can be used to find 
    all its children.
    
    Args:
        nested_dict: A dictionary that contains a list of dictionaries in
            the key_children entry.
        key: Key that contains the values to use as keys in the new dictionary. 
            The values of this key should be unique throughout the entire 
            nested_dict and thus serve as IDs.
        key_children: Name of the children key, which contains a list of 
            further dictionaries but can be empty.
        id_dict: The output dictionary as an OrderedDict to preserve key 
            order (though not hierarchical structure) so that children 
            will come after their parents; if None is given, an empty 
            dictionary will be created.
        parent_list: List of values for the given key in all parent 
            dictionaries.
    
    Returns:
        A dictionary with the original values as the keys, which each map 
        to another dictionary containing an entry with the dictionary 
        holding the given value and another entry with a list of all parent 
        dictionary values for the given key.
    """
    value = nested_dict[key]
    sub_dict = {NODE: nested_dict}
    if parent_list is not None:
        sub_dict[PARENT_IDS] = parent_list
    id_dict[value] = sub_dict
    try:
        children = nested_dict[key_children]
        parent_list = [] if parent_list is None else list(parent_list)
        parent_list.append(value)
        for child in children:
            #print("parents: {}".format(parent_list))
            create_reverse_lookup(
                child, key, key_children, id_dict, parent_list)
    except KeyError as e:
        print(e)
    return id_dict

def mirror_reverse_lookup(labels_ref, offset, name_modifier):
    # NOT CURRENTLY USED: replaced with neg values for mirrored side
    keys = list(labels_ref.keys())
    for key in keys:
        mirrored_key = key + offset
        mirrored_val = copy.deepcopy(labels_ref[key])
        node = mirrored_val[NODE]
        node[ABA_ID] = mirrored_key
        node[config.ABA_NAME] += name_modifier
        parent = node[ABA_PARENT]
        if parent is not None:
            node[ABA_PARENT] += offset
        try:
            parent_ids = mirrored_val[PARENT_IDS]
            parent_ids = np.add(parent_ids, offset).tolist()
        except KeyError as e:
            pass
        labels_ref[mirrored_key] = mirrored_val

def get_node(nested_dict, key, value, key_children):
    """Get a node from a nested dictionary by iterating through all 
    dictionaries until the specified value is found.
    
    Args:
        nested_dict: A dictionary that contains a list of dictionaries in
            the key_children entry.
        key: Key to check for the value.
        value: Value to find, assumed to be unique for the given key.
        key_children: Name of the children key, which contains a list of 
            further dictionaries but can be empty.
    
    Returns:
        The node matching the key-value pair, or None if not found.
    """
    try:
        #print("checking for key {}...".format(key), end="")
        found_val = nested_dict[key]
        #print("found {}".format(found_val))
        if found_val == value:
            return nested_dict
        children = nested_dict[key_children]
        for child in children:
            result = get_node(child, key, value, key_children)
            if result is not None:
                return result
    except KeyError as e:
        print(e)
    return None

def create_aba_reverse_lookup(labels_ref):
    """Create a reverse lookup dictionary for Allen Brain Atlas style
    ontology files.
    
    Args:
        labels_ref: The ontology file as a parsed JSON dictionary.
    
    Returns:
        Reverse lookup dictionary as output by :func:`create_reverse_lookup`.
    """
    return create_reverse_lookup(labels_ref["msg"][0], ABA_ID, ABA_CHILDREN)

def get_label_ids_from_position(coord, labels_img, scaling, rounding=False):
    """Get the atlas label IDs for the given coordinates.
    
    Args:
        coord: Coordinates of experiment image in (z, y, x) order. Can be an 
            [n, 3] array of coordinates.
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
        rounding: True to round coordinates after scaling, which should be 
            used rounding to reverse coordinates that were previously scaled 
            inversely to avoid size degredation with repeated scaling. 
            Typically rounding is False (default) so that coordinates fall 
            evenly to their lowest integer, without exceeding max bounds.
    
    Returns:
        An array of label IDs corresponding to ``coords``, or a scalar of 
        one ID if only one coordinate is given.
    """
    lib_clrbrain.printv("getting label IDs from coordinates")
    # scale coordinates to atlas image size
    coord_scaled = np.multiply(coord, scaling)
    if rounding: 
        # round when extra precision is necessary, such as during reverse 
        # scaling, which requires clipping so coordinates don't exceed labels 
        # image shape
        coord_scaled = np.around(coord_scaled).astype(np.int)
        coord_scaled = np.clip(
            coord_scaled, None, np.subtract(labels_img.shape, 1))
    else:
        # tpyically don't round to stay within bounds
        coord_scaled = coord_scaled.astype(np.int)
    '''
    exceeding = np.greater_equal(coord_scaled, labels_img.shape)
    print("exceeding:\n{}".format(exceeding))
    print("cood_scaled exceeding:\n{}".format(coord_scaled[np.any(exceeding, axis=1)]))
    print("y exceeding:\n{}".format(coord_scaled[coord_scaled[:, 1] >= labels_img.shape[1]]))
    '''
    
    # split coordinates into lists by dimension to index the labels image
    # at once
    coord_scaled = np.transpose(coord_scaled)
    coord_scaled = np.split(coord_scaled, coord_scaled.shape[0])
    '''
    print("coord:\n{}".format(coord))
    print("coord_scaled:\n{}".format(coord_scaled))
    print("max coord scaled: {}".format(np.max(coord_scaled, axis=2)))
    print("labels_img shape: {}".format(labels_img.shape))
    '''
    return labels_img[coord_scaled][0]

def get_label(coord, labels_img, labels_ref, scaling, level=None, 
              rounding=False):
    """Get the atlas label for the given coordinates.
    
    Args:
        coord: Coordinates of experiment image in (z, y, x) order.
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        labels_ref: The labels reference lookup, assumed to be generated by 
            :func:`create_reverse_lookup` to look up by ID.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
        level: The ontology level as an integer to target; defaults to None. 
            If None, level will be ignored, and the exact matching label 
            to the given coordinates will be returned. If a level is given, 
            the label at the highest (numerically lowest) level encompassing 
            this region will be returned.
        rounding: True to round coordinates after scaling (see 
            :func:``get_label_ids_from_position``); defaults to False.
    
    Returns:
        The label dictionary at those coordinates, or None if no label is 
        found.
    """
    label_id = get_label_ids_from_position(coord, labels_img, scaling, rounding)
    lib_clrbrain.printv("found label_id: {}".format(label_id))
    mirrored = label_id < 0
    if mirrored:
        label_id = -1 * label_id
    label = None
    try:
        label = labels_ref[label_id]
        if level is not None and label[NODE][ABA_LEVEL] > level:
            # search for parent at "higher" (numerically lower) level 
            # that matches the target level
            parents = label[PARENT_IDS]
            label = None
            if label_id < 0:
                parents = np.multiply(parents, -1)
            for parent in parents:
                parent_label = labels_ref[parent]
                if parent_label[NODE][ABA_LEVEL] == level:
                    label = parent_label
                    break
        if label is not None:
            label[MIRRORED] = mirrored
            lib_clrbrain.printv(
                "label ID at level {}: {}".format(level, label_id))
    except KeyError as e:
        lib_clrbrain.printv(
            "could not find label id {} or its parent (error {})"
            .format(label_id, e))
    return label

def get_label_name(label):
    """Get the atlas region name from the label.
    
    Args:
        label: The label dictionary.
    
    Returns:
        The atlas region name, or None if not found.
    """
    name = None
    try:
        if label is not None:
            node = label[NODE]
            if node is not None:
                name = node[config.ABA_NAME]
                print("name: {}".format(name))
                if label[MIRRORED]:
                    name += LEFT_SUFFIX
                else:
                    name += RIGHT_SUFFIX
    except KeyError as e:
        print(e, name)
    return name

def get_children(labels_ref_lookup, label_id, children_all=[]):
    """Get the children of a given atlas ID.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be 
            generated by :func:`create_reverse_lookup` to look up by ID.
        label_id: ID of the label to find.
        children_all: List of all children of this ID, used recursively; 
            defaults to an empty list. To include the ID itself, pass in a 
            list with this ID alone.
    
    Returns:
        A list of all children of the given ID, in order from highest 
        (numerically lowest) level to lowest.
    """
    label = labels_ref_lookup.get(label_id)
    if label:
        children = label[NODE][ABA_CHILDREN]
        for child in children:
            child_id = child[ABA_ID]
            #print("child_id: {}".format(child_id))
            children_all.append(child_id)
            get_children(labels_ref_lookup, child_id, children_all)
    return children_all

def get_region_middle(labels_ref_lookup, label_id, labels_img, scaling):
    """Approximate the middle position of a region by taking the middle 
    value of its sorted list of coordinates.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be 
            generated by :func:`create_reverse_lookup` to look up by ID.
        label_id: ID of the label to find.
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
    
    Returns:
        The middle value of a list of all coordinates in the region at the 
        given ID. All children of this ID are included in the region. The 
        region's coordinate sorting prioritizes z, followed by y, etc, meaning 
        that the middle value will be closest to the middle of z but may fall 
        be slightly away from midline in the other axes if this z does not 
        contain y/x's around midline. Getting the coordinate at the middle 
        of this list rather than another coordinate midway between other values 
        in the region ensures that the returned coordinate will reside within 
        the region itself, including non-contingous regions that may be 
        intermixed with coordinates not part of the region.
    """
    # gather IDs for label and all its children
    id_abs = abs(label_id)
    region_ids = get_children(labels_ref_lookup, id_abs, [id_abs])
    if label_id < 0: region_ids = np.multiply(-1, region_ids)
    print("region IDs: {}".format(region_ids))
    
    # get a list of all the region's coordinates to sort
    img_region = np.isin(labels_img, region_ids)
    region_coords = np.where(img_region)
    print("region_coords:\n{}".format(region_coords))
    
    def get_middle(region_coords):
        # recursively get value at middle of list for each axis
        sort_ind = np.lexsort(region_coords[::-1]) # last axis is primary key
        num_coords = len(sort_ind)
        if num_coords > 0:
            mid_ind = sort_ind[int(num_coords / 2)]
            mid = region_coords[0][mid_ind]
            if len(region_coords) > 1:
                # shift to next axis in tuple of coords
                mask = region_coords[0] == mid
                region_coords = tuple(
                    coords[mask] for coords in region_coords[1:])
                return (mid, *get_middle(region_coords))
            return (mid, )
        return None
    
    coord = None
    coord_labels = get_middle(region_coords)
    if coord_labels:
        print("coord_labels (unscaled): {}".format(coord_labels))
        print("ID at middle coord: {} (in region? {})"
              .format(labels_img[coord_labels], img_region[coord_labels]))
        coord = tuple(np.around(coord_labels / scaling).astype(np.int))
    print("coord at middle: {}".format(coord))
    return coord, img_region

def get_region_from_id(img_region, scaling):
    """Get the entire region encompassing a given label ID.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be 
            generated by :func:`create_reverse_lookup` to look up by ID.
        label_id: ID of the label to find.
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        scaling: Scaling factor for the labels image size compared with the 
            experiment image.
    
    Returns:
        A tuple of ``props, bbox, centroid``. ``props`` is the first 
        ``regionprops`` for the given region. ``bbox`` is the bounding box 
        of this region scaled back to the experiment image. ``centroid`` is 
        the centroid position also scaled to the experiment. Note that the 
        bounding box may encompass many pixels not included in the region 
        itself, including border pixels or even internal pixels in irregularly 
        shaped or sparsely scattered regions. If so, the centroid position 
        may in fact be outside of the region. To ensure that a coordinate 
        within the region tree for ``label_id`` is returned, use 
        :func:``get_region_middle`` instead.
    """
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

def volumes_by_id(labels_img, labels_ref_lookup, resolution, level=None, 
                  blobs_ids=None, image5d=None):
    """Get volumes by labels IDs.
    
    Args:
        labels_img: The registered image whose intensity values correspond to 
            label IDs.
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`create_reverse_lookup` to look up 
            by ID while preserving key order to ensure that parents of any 
            child will be reached prior to the child.
        resolution: The image resolutions as an array in (z, y, x) order, 
            typically the spacing of the SimpleITK image, not the 
            original image's resolution.
        level: The ontology level as an integer; defaults to None. If None, 
            volumes for all label IDs will be returned. If a level is 
            given, only regions from that level will be returned, while 
            children will be collapsed into the parent at that level, and 
            regions above this level will be ignored. Recommended practice is 
            to generate this volumes dict with level of None to use as a 
            master dictionary, followed by volumes at a given level using 
            :func:``volumes_dict_level_grouping``.
        blobs_ids: List of label IDs for blobs. If None, blob densities will 
            not be calculated.
        image5d: Image to which ``labels_img`` is registered, used to 
            verify the actual volume present corresponding to each label. 
            Defaults to None, in which case volume will be based solely on 
            the labels themselves.
    
    Returns:
        Nested dictionary of {ID: {:attr:`config.ABA_NAME`: name, 
        :attr:`config.VOL_KEY`: volume, 
        :attr:`config.BLOBS_KEY`: number of blobs}}, where volume is in the 
        cubed units of :attr:`detector.resolutions`.
    """
    ids = list(labels_ref_lookup.keys())
    #print("ids: {}".format(ids))
    volumes_dict = {}
    scaling_vol = np.prod(resolution)
    scaling_vol_image5d = scaling_vol
    scaling_inv = None
    # default to simple threshold since costly to calc for large image
    thresh = _SIGNAL_THRESHOLD
    if image5d is not None and image5d.shape[1:4] != labels_img.shape:
        # find scale between larger image5d and labels image
        scaling = importer.calc_scaling(image5d, labels_img)
        scaling_inv = np.divide(1, scaling)
        #scaling_vol_image5d = np.prod(detector.resolutions[0])
        scaling_vol_image5d = scaling_vol * np.prod(scaling)
        print("images have different shapes so will scale to compare "
              "with scaling of {}".format(scaling_inv))
    elif image5d is not None:
        # find global threshold from scaled (presumably smaller) image
        thresh = filters.threshold_mean(image5d)
        print("using signal threshold of {}".format(thresh))
    for key in ids:
        label_ids = [key, -1 * key]
        for label_id in label_ids:
            label = labels_ref_lookup[key] # always use pos val
            mask_id = labels_img == label_id
            region = labels_img[mask_id]
            vol = len(region) * scaling_vol
            
            if image5d is not None:
                # find volume of corresponding region in experiment image
                # where significant signal (eg actual tissue) is present; 
                # assume that rest of region does not contain blobs since 
                # not checking for blobs there
                vol_image5d = 0
                vol_theor = 0 # to verify
                if scaling_inv is not None:
                    # scale back to full-sized image
                    coords = list(np.where(mask_id))
                    coords = np.transpose(coords)
                    coords = np.around(
                        np.multiply(coords, scaling_inv)).astype(np.int32)
                    coords_end = np.around(
                        np.add(coords, scaling_inv)).astype(np.int32)
                    for i in range(len(coords)):
                        #print("slicing from {} to {}".format(coords[i], coords_end[i]))
                        region_image5d = image5d[
                            0, coords[i][0]:coords_end[i][0],
                            coords[i][1]:coords_end[i][1],
                            coords[i][2]:coords_end[i][2]]
                        present = region_image5d[
                            region_image5d > thresh]
                        #print("len of region with tissue: {}".format(len(region_present)))
                        vol_image5d += len(present) * scaling_vol_image5d
                        vol_theor += region_image5d.size * scaling_vol_image5d
                else:
                    # use scaled image, whose size should match the labels image
                    image5d_in_region = image5d[0, mask_id]
                    present = image5d_in_region[
                        image5d_in_region >= thresh]
                    vol_image5d = len(present) * scaling_vol_image5d
                    vol_theor = len(image5d_in_region) * scaling_vol_image5d
                    pixels = len(image5d_in_region)
                    print("{} of {} pixels under threshold"
                          .format(pixels - len(present), pixels))
                print("Changing labels vol of {} to image5d vol of {} "
                      "(theor max of {})".format(vol, vol_image5d, vol_theor))
                vol = vol_image5d
            
            # get blobs annotated to the given label
            blobs = None
            blobs_len = 0
            if blobs_ids is not None:
                blobs = blobs_ids[blobs_ids == label_id]
                blobs_len = len(blobs)
            #print("checking id {} with vol {}".format(label_id, vol))
            
            # insert a regional dict for the given label
            label_level = label[NODE][ABA_LEVEL]
            name = label[NODE][config.ABA_NAME]
            if level is None or label_level == level or label_level == -1:
                # include region in volumes dict if at the given level, no 
                # level specified (to build a mster dict), or at the default 
                # (organism) level, which is used to catch all children without 
                # a parent at the given level
                region_dict = {
                    config.ABA_NAME: label[NODE][config.ABA_NAME],
                    config.VOL_KEY: vol,
                    config.BLOBS_KEY: blobs_len
                }
                volumes_dict[label_id] = region_dict
                print("inserting region {} (id {}) with {} vol and {} blobs "
                      .format(name, label_id, vol, blobs_len))
            else:
                # add vol and blobs to parent with a regional dict, which is 
                # parent at the given level; assume that these dicts will have 
                # already been made for these parents since key order is 
                # hierarchical, while regions higher than given level or in 
                # another hierachical branch will not find a parent with dict
                parents = label.get(PARENT_IDS)
                if parents is not None:
                    if label_id < 0:
                        parents = np.multiply(parents, -1)
                    # start from last parent to avoid level -1 unless no 
                    # parent found and stop checking as soon as parent found
                    found_parent = False
                    for parent in parents[::-1]:
                        region_dict = volumes_dict.get(parent)
                        if region_dict is not None:
                            region_dict[config.VOL_KEY] += vol
                            region_dict[config.BLOBS_KEY] += blobs_len
                            print("added {} vol and {} blobs from {} (id {}) "
                                  "to {}".format(vol, blobs_len, name, 
                                  label_id, region_dict[config.ABA_NAME]))
                            found_parent = True
                            break
                    if not found_parent:
                        print("could not find parent for {} with blobs {}"
                              .format(label_id, blobs_len))
    
    # blobs summary
    blobs_tot = 0
    for key in volumes_dict.keys():
        # all blobs matched to a region at the given level; 
        # TODO: find blobs that had a label (ie not 0) but did not match any 
        # parent, including -1
        if key >= 0:
            blobs_side = volumes_dict[key][config.BLOBS_KEY]
            blobs_mirrored = volumes_dict[-1 * key][config.BLOBS_KEY]
            print("{} (id {}), {}: volume {}, blobs {}; {}: volume {}, blobs {}"
                .format(volumes_dict[key][config.ABA_NAME], key, 
                RIGHT_SUFFIX, volumes_dict[key][config.VOL_KEY], blobs_side, 
                LEFT_SUFFIX, volumes_dict[-1 * key][config.VOL_KEY], 
                blobs_mirrored))
            blobs_tot += blobs_side + blobs_mirrored
    # all unlabeled blobs
    blobs_unlabeled = blobs_ids[blobs_ids == 0]
    blobs_unlabeled_len = len(blobs_unlabeled)
    print("unlabeled blobs (id 0): {}".format(blobs_unlabeled_len))
    blobs_tot += blobs_unlabeled_len
    print("total blobs accounted for: {}".format(blobs_tot))
    return volumes_dict

def labels_to_parent(labels_ref_lookup, level):
    """Generate a dictionary mapping label IDs to parent IDs at a given level.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`create_reverse_lookup` to look up 
            by ID while preserving key order to ensure that parents of any 
            child will be reached prior to the child.
        level: Level at which to find parent for each label.
    
    Returns:
        Dictionary of label IDs to parent IDs at the given level. Labels at 
        the given level will be assigned to their own ID, and labels above 
        (numerically lower) or without a parent at the level will be 
        given a default level of 0.
    """
    # similar to volumes_dict_level_grouping but without checking for neg 
    # keys or grouping values
    label_parents = {}
    ids = list(labels_ref_lookup.keys())
    for label_id in ids:
        parent_at_level = 0
        label = labels_ref_lookup[label_id]
        label_level = label[NODE][ABA_LEVEL]
        if label_level == level:
            parent_at_level = label_id
        elif label_level > level:
            parents = label.get(PARENT_IDS)
            for parent in parents[::-1]:
                parent_level = labels_ref_lookup[parent][NODE][ABA_LEVEL]
                if parent_level < level:
                    break
                elif parent_level == level:
                    parent_at_level = parent
        label_parents[label_id] = parent_at_level
    return label_parents

def volumes_dict_level_grouping(volumes_dict, level):
    """Group child volumes into the given parent level from a volumes 
    dictionary.
    
    Args:
        volumes_dict: Volumes dictionary as generated by 
            :func:``volumes_by_id``.
        level: Level at which to group volumes.
    
    Returns:
        The grouped dictionary.
    """
    level_dict = {}
    ids = list(labels_ref_lookup.keys())
    for key in ids:
        label_ids = [key, -1 * key]
        for label_id in label_ids:
            label = labels_ref_lookup[key] # always use pos val
            label_level = label[NODE][ABA_LEVEL]
            name = label[NODE][config.ABA_NAME]
            if label_level == level or label_level == -1:
                print("found region at given level with id {}".format(label_id))
                level_dict[label_id] = volumes_dict.get(label_id)
            elif label_level > level:
                parents = label.get(PARENT_IDS)
                if parents is not None:
                    if label_id < 0:
                        parents = np.multiply(parents, -1)
                    # start from last parent, assuming parents listed in order 
                    # with last parent at highest numerical level, stopping 
                    # once parent at level found or parent exceeds given level
                    found_parent = False
                    for parent in parents[::-1]:
                        parent_level = labels_ref_lookup[
                            abs(parent)][NODE][ABA_LEVEL]
                        if parent_level < level:
                            break
                        region_dict_level = level_dict.get(parent)
                        if region_dict_level is not None:
                            region_dict = volumes_dict.get(label_id)
                            vol = region_dict[config.VOL_KEY]
                            blobs_len = region_dict[config.BLOBS_KEY]
                            region_dict_level[config.VOL_KEY] += vol
                            region_dict_level[config.BLOBS_KEY] += blobs_len
                            print("added {} vol and {} blobs from {} (id {}) "
                                  "to {}".format(vol, blobs_len, name, 
                                  label_id, region_dict[config.ABA_NAME]))
                            found_parent = True
                            break
                    if not found_parent:
                        print("could not find parent for {}".format(label_id))
            else:
                print("skipping {} as its level is above target"
                      .format(label_id))
    return level_dict

def get_volumes_dict_path(img_path, level):
    """Get the path of the volumes dictionary corresponding to the given image 
    and level.
    
    Args:
        img_path: Path to image file.
        level: Level; if None, "All" will be substituted.
    
    Returns:
        Path to volumes dictionary corresponding to the image at the given 
        level.
    """
    if level is None:
        level = "All"
    return "{}_volumes_level{}.json".format(os.path.splitext(img_path)[0], level)

def open_json(json_path):
    """Open a JSON file from the given path, with indices converted to ints.
    
    Args:
        json_path: Path to JSON file.
    
    Returns:
        The JSON file as a dictionary, None if the path does not point to a 
        file.
    """
    volumes_dict = None
    if os.path.isfile(json_path):
        with open(json_path, "r") as fp:
            print("reloading saved volumes dict from {}".format(json_path))
            volumes_dict = json.load(
                fp, object_hook=lib_clrbrain.convert_indices_to_int)
            print(volumes_dict)
    return volumes_dict

def register_volumes(img_path, labels_ref_lookup, level, scale=None, 
                     densities=False, index=None, suffix=None):
    """Register volumes and densities.
    
    If a volumes dictionary from the path generated by 
    :func:``get_volumes_dict_path`` exists, this dictionary will be loaded 
    instead. If a dictionary will volumes for each ID exists, this 
    dictionary will be used to generate a volumes dictionary for the given 
    level, unless level is None.
    
    Args:
        img_path: Path to the original image file.
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`create_reverse_lookup` to look up 
            by ID while preserving key order to ensure that parents of any 
            child will be reached prior to the child.
        level: Ontology level at which to show volumes and densities. Can be 
            None, in which case volumes for each individual ID will be 
            stored rather than grouping by level.
        scale: Rescaling factor as a scalar value. If set, the corresponding 
            image for this factor will be opened. If None, the full size 
            image will be used. Defaults to None.
        densities: True if densities should be displayed; defaults to False.
        index: Index to report back as output for tracking order in 
            multiprocessing.
        suffix: Modifier to append to end of ``img_path`` basename for 
            registered image files that were output to a modified name; 
            defaults to None.
    
    Returns:
        A tuple of ``volumes_dict``, the volumes dictionary in the format 
        of ``{[ID as int]: {name: [name as str], volume: [volume as float], 
        blobs: [blobs as int]}, ...}``; ``json_path``, the path to the JSON 
        file where the dictionary is saved; and ``index``, which is the same 
        as the ``index`` parameter to identify the order in which this 
        function was called if launched from multiprocessing.
    """
    mod_path = img_path
    if suffix is not None:
        mod_path = lib_clrbrain.insert_before_ext(img_path, suffix)
    # reload saved volumes dictionary if possible for the given level
    volumes_dict = None
    json_path = get_volumes_dict_path(mod_path, level)
    volumes_dict = open_json(json_path)
    
    if volumes_dict is None:
        # reload the saved volumes dictionary for all levels if possible
        json_path_all = get_volumes_dict_path(mod_path, None)
        volumes_dict = open_json(json_path_all)
        
        if volumes_dict is None:
            # build volumes dictionary for the given level, which can be None
            
            # load labels image and setup labels dictionary
            labels_img_sitk = load_registered_img(
                mod_path, get_sitk=True, reg_name=IMG_LABELS)
            spacing = labels_img_sitk.GetSpacing()
            labels_img = sitk.GetArrayFromImage(labels_img_sitk)
            print("{} shape: {}".format(img_path, labels_img.shape))
            
            # load blob densities by region if flagged
            blobs_ids = None
            if densities:
                # load blobs
                filename_base = importer.filename_to_base(
                    img_path, config.series)
                info = np.load(filename_base + config.SUFFIX_INFO_PROC)
                blobs = info["segments"]
                print("loading {} blobs".format(len(blobs)))
                #print("blobs range:\n{}".format(np.max(blobs, axis=0)))
                target_size = config.register_settings["target_size"]
                img_path_transposed = importer.get_transposed_image_path(
                    img_path, scale, target_size)
                if scale is not None or target_size is not None:
                    image5d, img_info = importer.read_file(
                        img_path_transposed, config.series, return_info=True)
                    scaling = img_info["scaling"]
                else:
                    # fall back to pixel comparison based on original image, 
                    # but **very slow**
                    image5d = importer.read_file(
                        img_path_transposed, config.series)
                    scaling = importer.calc_scaling(image5d, labels_img)
                print("using scaling: {}".format(scaling))
                # annotate blobs based on position
                blobs_ids = get_label_ids_from_position(
                    blobs[:, 0:3], labels_img, scaling)
                print("blobs_ids: {}".format(blobs_ids))
            
            # calculate and plot volumes and densities
            volumes_dict = volumes_by_id(
                labels_img, labels_ref_lookup, spacing, level=level, 
                blobs_ids=blobs_ids, image5d=image5d)
        
        elif level is not None:
            # use the all levels volumes dictionary to group child levels 
            # into their parent at the given level
            volumes_dict = volumes_dict_level_grouping(volumes_dict, level)
        
        if volumes_dict is not None:
            # output volumes dictionary to file with pretty printing but no 
            # sorting to ensure that IDs are in the same hierarchical order 
            # as labels reference, even if not grouped hierarchically
            print(volumes_dict)
            with open(json_path, "w") as fp:
                json.dump(volumes_dict, fp, indent=4)
    
    return volumes_dict, json_path, index

def register_volumes_mp(img_paths, labels_ref_lookup, level, scale=None, 
                        densities=False, suffix=None):
    start_time = time()
    '''
    # testing capping processes when calculating volumes for levels since 
    # appeared to run slowly with multiple files in parallel during some 
    # tests but not others
    processes = 4 if level is None else None
    pool = mp.Pool(processes=processes)
    '''
    pool = mp.Pool()
    pool_results = []
    img_paths_len = len(img_paths)
    for i in range(img_paths_len):
        img_path = img_paths[i]
        pool_results.append(pool.apply_async(
            register_volumes, 
            args=(img_path, labels_ref_lookup, level, scale, densities, 
                  i, suffix)))
    vols = [None] * img_paths_len
    paths = [None] * img_paths_len
    for result in pool_results:
        vol_dict, path, i = result.get()
        vols[i] = vol_dict
        paths[i] = path
        print("finished {}".format(path))
    #print("vols: {}".format(vols))
    pool.close()
    pool.join()
    print("time elapsed for volumes by ID: {}".format(time() - start_time))
    return vols, paths

def group_volumes(labels_ref_lookup, vol_dicts):
    """Group volumes from multiple volume dictionaries.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`create_reverse_lookup` to look up 
            by ID while preserving key order to ensure that parents of any 
            child will be reached prior to the child.
        vol_dicts: List of volume dictionaries, from which values from 
            identical IDs will be pooled.
    Returns:
        Grouped volumes dictionaries with a set of IDs containing all of the 
        unique IDs in the individual volume dictionaries and lists of values 
        in places of individual values.
    """
    ids = list(labels_ref_lookup.keys())
    grouped_vol_dict = {}
    for key in ids:
        # check all IDs, including negative versions for mirrored labels
        label_ids = [key, -1 * key]
        for label_id in label_ids:
            # entry associated with the ID, which should be identical for 
            # each dictionary except the numerical values; eg all volume 
            # lengths should be the same as one another since if one 
            # vol_dict has an entry for the given label_id, all vol dicts 
            # should have an entry as well
            entry_group = None
            for vol_dict in vol_dicts:
                entry_vol = vol_dict.get(label_id)
                if entry_vol is not None:
                    if entry_group is None:
                        # shallow copy since only immutable values within dict
                        entry_group = dict(entry_vol)
                        entry_group[config.VOL_KEY] = [
                            entry_group[config.VOL_KEY]]
                        entry_group[config.BLOBS_KEY] = [
                            entry_group[config.BLOBS_KEY]]
                        grouped_vol_dict[label_id] = entry_group
                    else:
                        # append numerical values to existing lists
                        entry_group[config.VOL_KEY].append(
                            entry_vol[config.VOL_KEY])
                        entry_group[config.BLOBS_KEY].append(
                            entry_vol[config.BLOBS_KEY])
    #print("grouped_vol_dict: {}".format(grouped_vol_dict))
    return grouped_vol_dict

def export_region_ids(labels_ref_lookup, path, level):
    """Export region IDs from annotation reference reverse mapped dictionary 
    to CSV file.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`create_reverse_lookup` to look up 
            by ID while preserving key order to ensure that parents of any 
            child will be reached prior to the child.
        path: Path to output CSV file; if does not end with ``.csv``, it will 
            be added.
        level: Level at which to find parent for each label.
    
    Returns:
        Pandas data frame of the region IDs and corresponding names.
    """
    ext = ".csv"
    if not path.endswith(ext): path += ext
    label_parents = labels_to_parent(labels_ref_lookup, level)
    header = ["Region", "RegionName", "Parent"]
    data = OrderedDict()
    for h in header:
        data[h] = []
    for key in labels_ref_lookup.keys():
        # does not include laterality distinction, only using original IDs
        label = labels_ref_lookup[key]
        data[header[0]].append(key)
        data[header[1]].append(label[NODE][config.ABA_NAME])
        # ID of parent at label_parents' level
        parent = label_parents[key]
        data[header[2]].append(parent)
    data_frame = pd.DataFrame(data=data, columns=header)
    data_frame.to_csv(path, index=False)
    print("exported volume data per sample to CSV at {}".format(path))
    return data_frame

def export_region_network(labels_ref_lookup, path):
    """Export region network file showing relationships among regions 
    according to the SIF specification.
    
    See http://manual.cytoscape.org/en/stable/Supported_Network_File_Formats.html#sif-format
    for file format information.
    
    Args:
        labels_ref_lookup: The labels reference lookup, assumed to be an 
            OrderedDict generated by :func:`create_reverse_lookup` to look up 
            by ID while preserving key order to ensure that parents of any 
            child will be reached prior to the child.
        path: Path to output SIF file; if does not end with ``.sif``, it will 
            be added.
    """
    ext = ".sif"
    if not path.endswith(ext): path += ext
    network = {}
    for key in labels_ref_lookup.keys():
        if key < 0: continue # only use original, non-neg region IDs
        label = labels_ref_lookup[key]
        parents = label.get(PARENT_IDS)
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

def _test_labels_lookup():
    """Test labels reverse dictionary creation and lookup.
    """
    
    # create reverse lookup dictionary
    ref = load_labels_ref(config.load_labels)
    #pprint(ref)
    lookup_id = 15565 # short search path
    #lookup_id = 126652058 # last item
    time_dict_start = time()
    id_dict = create_aba_reverse_lookup(ref)
    labels_img = load_registered_img(config.filename, reg_name=IMG_LABELS)
    max_labels = np.max(labels_img)
    print("max_labels: {}".format(max_labels))
    #mirror_reverse_lookup(id_dict, max_labels, " (R)")
    #pprint(id_dict)
    time_dict_end = time()
    
    # look up a single ID
    time_node_start = time()
    found = id_dict[lookup_id]
    time_node_end = time()
    print("found {}: {} with parents {}".format(lookup_id, found[NODE]["name"], found[PARENT_IDS]))
    
    # brute-force query
    time_direct_start = time()
    node = get_node(ref["msg"][0], "id", lookup_id, "children")
    time_direct_end = time()
    #print(node)
    
    print("time to create id_dict (s): {}".format(time_dict_end - time_dict_start))
    print("time to find node (s): {}".format(time_node_end - time_node_start))
    print("time to find node directly (s): {}".format(time_direct_end - time_direct_start))
    
    # get a list of IDs corresponding to each blob
    blobs = np.array([[300, 5000, 3000], [350, 5500, 4500], [400, 6000, 5000]])
    image5d = importer.read_file(config.filename, config.series)
    scaling = importer.calc_scaling(image5d, labels_img)
    ids = get_label_ids_from_position(blobs[:, 0:3], labels_img, scaling)
    print("blob IDs:\n{}".format(ids))

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
    ref = load_labels_ref(config.load_labels)
    id_dict = create_aba_reverse_lookup(ref)
    middle, img_region = get_region_middle(id_dict, 16652, labels_img, scaling)
    atlas_label = get_label(middle, labels_img, id_dict, scaling, None, True)
    props, bbox, centroid = get_region_from_id(img_region, scaling)
    print("bbox: {}, centroid: {}".format(bbox, centroid))

def _test_curate_img(path):
    fixed_img = _load_numpy_to_sitk(path)
    moving_dir = os.path.dirname(path)
    labels_img = sitk.ReadImage(os.path.join(moving_dir, IMG_LABELS))
    atlas_img = sitk.ReadImage(os.path.join(moving_dir, IMG_ATLAS))
    result_imgs = _curate_img(fixed_img, labels_img, [atlas_img])
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

if __name__ == "__main__":
    print("Clrbrain image registration")
    from clrbrain import cli
    cli.main(True)
    plot_2d.setup_style()
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
    
    #_test_labels_lookup()
    #_test_region_from_id()
    #_test_curate_img(config.filenames[0])
    #_test_smoothing_metric()
    #os._exit(os.EX_OK)
    
    if config.register_type is None:
        # explicitly require a registration type
        print("Please choose a registration type")
    elif config.register_type in (
        config.REGISTER_TYPES[0], config.REGISTER_TYPES[7]):
        # "single", basic registration of 1st to 2nd image, transposing the 
        # second image according to config.plane and config.flip_horiz; 
        # "new_atlas" registers similarly but outputs new atlas files
        new_atlas = config.register_type == config.REGISTER_TYPES[7]
        register(*config.filenames[0:2], plane=config.plane, 
                 flip=flip, name_prefix=config.prefix, new_atlas=new_atlas, 
                 show_imgs=show)
    elif config.register_type == config.REGISTER_TYPES[1]:
        # groupwise registration, which assumes that the last image 
        # filename given is the prefix and uses the full flip array
        prefix = config.filenames[-1]
        register_group(
            config.filenames[:-1], flip=config.flip, name_prefix=config.prefix, 
            scale=config.rescale, show_imgs=show)
    elif config.register_type == config.REGISTER_TYPES[2]:
        # overlay registered images in each orthogonal plane
        for out_plane in config.PLANE:
            overlay_registered_imgs(
                *config.filenames[0:2], plane=config.plane, 
                flip=flip, name_prefix=config.prefix, 
                out_plane=out_plane)
    elif config.register_type in (
        config.REGISTER_TYPES[3], config.REGISTER_TYPES[4]):
        
        # compute grouped volumes/densities by ontology level
        densities = config.register_type == config.REGISTER_TYPES[4]
        ref = load_labels_ref(config.load_labels)
        labels_ref_lookup = create_aba_reverse_lookup(ref)
        vol_dicts, json_paths = register_volumes_mp(
            config.filenames, labels_ref_lookup, 
            config.labels_level, config.rescale, densities, 
            suffix=config.suffix)
        
        # plot volumes for individual experiments for each region
        exps = []
        for vol, path in zip(vol_dicts, json_paths):
            exp_name = os.path.basename(path)
            vol_stats = tuple(stats.volume_stats(
                vol, densities, unit_factor=unit_factor))
            plot_2d.plot_volumes(
                vol_stats, title=os.path.splitext(exp_name)[0], 
                densities=densities, show=show)
            # experiment identifiers, assumed to be at the start of the image 
            # filename, separated by a "-"; if no dash, will use the whole name
            exps.append(exp_name.split("-")[0])
            #stats.volume_stats_to_csv(vol_stats, "csvtest")
        
        # plot mean volumes of all experiments for each region
        group_vol_dict = group_volumes(labels_ref_lookup, vol_dicts)
        vol_stats = tuple(stats.volume_stats(
            group_vol_dict, densities, config.groups, unit_factor))
        title = "Volume Means from {} at Level {}".format(
            ", ".join(exps), config.labels_level)
        plot_2d.plot_volumes(
            vol_stats, title=title, densities=densities, show=show, 
            groups=config.groups)
        
        # write stats to CSV file
        stats.volume_stats_to_csv(vol_stats, title, config.groups)
        
    elif config.register_type in (
        config.REGISTER_TYPES[5], config.REGISTER_TYPES[6]):
        
        # export registered stats and regions IDs to CSV files
        
        # use all ABA levels (0-13) and find parent at the given level
        levels = list(range(14))
        ref = load_labels_ref(config.load_labels)
        labels_ref_lookup = create_aba_reverse_lookup(ref)
        
        if config.register_type == config.REGISTER_TYPES[5]:
            # export volumes to CSV
            
            # convert groupings from strings to numerical format for stats
            groups_numeric = [
                config.GROUPS_NUMERIC[geno] for geno in config.groups]
            dfs = []
            for level in levels:
                # register segments (or simply load if already done), combine 
                # from all samples, and convert to Pandas data frame for 
                # the given level
                vol_dicts, json_paths = register_volumes_mp(
                    config.filenames, labels_ref_lookup, level, config.rescale, 
                    True, suffix=config.suffix)
                group_vol_dict = group_volumes(labels_ref_lookup, vol_dicts)
                dfs.append(
                    stats.regions_to_pandas(
                        group_vol_dict, level, groups_numeric, unit_factor))
            
            # combine data frames and export to CSV
            stats.data_frames_to_csv(dfs, "vols_by_sample")
        
        elif config.register_type == config.REGISTER_TYPES[6]:
            # export region IDs and parents at given level to CSV
            export_region_ids(
                labels_ref_lookup, "region_ids", config.labels_level)
            # export region IDs to network file
            export_region_network(labels_ref_lookup, "region_network")
        
    elif config.register_type == config.REGISTER_TYPES[8]:
        # import original atlas, mirroring if necessary
        import_atlas(config.filename, show=show)
