#!/bin/bash
# Atlas refinement
# Author: David Young, 2019
"""Refine atlases in 3D.
"""
import multiprocessing as mp
import os
from collections import OrderedDict
from time import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import transform

from clrbrain import config
from clrbrain import detector
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import plot_3d
from clrbrain import plot_support
from clrbrain import sitk_io
from clrbrain import stats


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


def truncate_labels(img_np, x_frac=None, y_frac=None, z_frac=None):
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


def mirror_planes(img_np, start, mirror_mult=1, resize=True, start_dup=None,
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
        edge (Dict[str, float]): Lateral edge extension parameters passed 
            to :meth:`plot_3d.extend_edge`. The value from ``start`` 
            specifies the fraction of z-planes from which to start 
            extending the edge to lower z-planes, and -1 will cause the 
            edge plane to be found automatically based on the first 
            plane with any labels, starting from the lowest plane. 
            Defaults to None, in which case edge extension will be skipped.
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
         indices from which the edge was extended (labels used as template 
         from this plane index) and the image was mirrored (index where 
         mirrored hemisphere starts); and a data frame of smoothing stats, 
         or None if smoothing was not performed.
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
    
    # lateral edges of atlas labels (ie low z in sagittal orientation) are 
    # missing in many ABA developing mouse atlases, requiring extension
    edgei = 0
    edgei_first = None
    edge_start = None if edge is None else edge["start"]
    if edge_start == -1 or edge is None:
        # find the first non-zero plane
        for plane in img_np:
            if not np.allclose(plane, 0):
                if edgei_first is None:
                    edgei_first = edgei
                elif edgei - edgei_first >= 1:
                    # require at least 2 contiguous planes with signal
                    edgei = edgei_first
                    break
            else:
                edgei_first = None
            edgei += 1
        print("found start of contiguous non-zero planes at {}".format(edgei))
    else:
        # based on profile settings
        edgei = int(edge_start * tot_planes)
        print("will extend near edge from plane {}".format(edgei))
    
    if edge is not None:
        # find the bounds of the reference image in the given plane and resize 
        # the corresponding section of the labels image to the bounds of the 
        # reference image in the next plane closer to the edge, recursively 
        # extending the nearest plane with labels based on the underlying 
        # atlas; assume that each nearer plane is the same size or smaller 
        # than the next farther plane, such as a tapering specimen
        plot_3d.extend_edge(
            img_np, img_ref_np, config.register_settings["atlas_threshold"], 
            None, edgei, edge["surr_size"], edge["closing_size"])
    
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
            _, df_smoothing = _smoothing(
                img_smoothed, img_smoothed_orig, smooth)
    
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
        img_np = mirror_planes(
            img_np, mirrori, mirror_mult=-1, check_equality=True, resize=resize)
        print("total final labels: {}".format(np.unique(img_np).size))
    return img_np, (edgei, mirrori), df_smoothing


def crop_to_orig(labels_img_np_orig, labels_img_np, crop):
    # crop new labels to extent of original labels unless crop is False
    print("cropping to original labels' extent with filter size of", crop)
    if crop is False: return
    mask = labels_img_np_orig == 0
    if crop > 0:
        # smooth mask
        mask = morphology.binary_opening(mask, morphology.ball(crop))
    labels_img_np[mask] = 0


def _smoothing(img_np, img_np_orig, filter_size):
    """Smooth image and calculate smoothing metric for use individually or 
    in multiprocessing.
    
    Args:
        img_np: Image as Numpy array, which will be directly updated.
        img_np_orig: Original image as Numpy array for comparison with 
            smoothed image in metric.
        filter_size: Structuring element size for smoothing.
    
    Returns:
        Tuple of ``filter_size`` and a data frame of smoothing metrices.
    """
    smoothing_mode = config.register_settings["smoothing_mode"]
    smooth_labels(img_np, filter_size, smoothing_mode)
    df_metrics, df_raw = label_smoothing_metric(img_np_orig, img_np)
    df_metrics[config.SmoothingMetrics.FILTER_SIZE.value] = [filter_size]
    
    # curate back to lightly smoothed foreground of original labels
    crop = config.register_settings["crop_to_orig"]
    crop_to_orig(img_np_orig, img_np, crop)
    
    print("\nMeasuring foreground overlap of labels after smoothing:")
    measure_overlap_labels(
        make_labels_fg(sitk.GetImageFromArray(img_np)), 
        make_labels_fg(sitk.GetImageFromArray(img_np_orig)))
    
    return filter_size, df_metrics


def _smoothing_mp(img_np, img_np_orig, filter_sizes):
    """Smooth image and calculate smoothing metric for a list of smoothing 
    strengths.
    
    Args:
        img_np: Image as Numpy array, which will be directly updated.
        img_np_orig: Original image as Numpy array for comparison with 
            smoothed image in metric.
        filter_sizes: Tuple or list of structuring element sizes.
    
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
        filter_size, df_metrics = result.get()
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


def label_smoothing_metric(orig_img_np, smoothed_img_np):
    """Measure degree of appropriate smoothing, defined as smoothing that 
    retains the general shape and placement of the region.
    
    Compare the difference in compactness before and after the smoothing 
    algorithm, termed "compaction," while penalizing inappropriate smoothing, 
    the smoothed volume lying outside of the original broad volume, termed 
    "displacement."
    
    Args:
        orig_img_np: Unsmoothed labels image as Numpy array.
        smoothed_img_np: Smoothed labels image as Numpy array, which 
            should be of the same shape as ``original_img_np``.
    
    Returns:
        Tuple of a data frame of the smoothing metrics and another data 
        frame of the raw metric components.
    """
    def meas_compactness(img_np):
        # get the borders of the label and add them to a rough image
        region = img_np[tuple(slices)]
        label_mask_region = region == label_id
        area = plot_3d.surface_area_3d(label_mask_region)
        compactness = plot_3d.compactness(None, label_mask_region, area)
        return label_mask_region, area, compactness
    
    pxs = {}
    cols = ("label_id", "pxs_reduced", "pxs_expanded", "size_orig")
    label_ids = np.unique(orig_img_np)
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
            props[0].bbox, 2, orig_img_np.shape)
        
        # measure surface area for SA:vol and to get vol mask
        mask_orig, area_orig, compact_orig = meas_compactness(
            orig_img_np)
        mask_smoothed, area_sm, compact_smoothed = meas_compactness(
            smoothed_img_np)
        
        # "compaction": reduction in compactness, multiplied by 
        # orig vol for wt avg
        size_orig = np.sum(mask_orig)
        pxs.setdefault("compactness_orig", []).append(compact_orig)
        pxs.setdefault("compactness_smoothed", []).append(compact_smoothed)
        pxs_reduced = (compact_orig - compact_smoothed) / compact_orig
        
        # "displacement": fraction of displaced volume
        displ = np.sum(np.logical_and(mask_smoothed, ~mask_orig))
        pxs.setdefault("displacement", []).append(displ)
        pxs_expanded = displ / size_orig
        
        # SA:vol metrics
        sa_to_vol_orig = area_orig / np.sum(mask_orig)
        vol_smoothed = np.sum(mask_smoothed)
        sa_to_vol_smoothed = 0
        if vol_smoothed > 0:
            sa_to_vol_smoothed = area_sm / vol_smoothed
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
    
    metrics = dict.fromkeys(config.SmoothingMetrics, np.nan)
    tot_size = totals["size_orig"]
    if tot_size > 0:
        frac_reduced = totals["pxs_reduced"] / tot_size
        frac_expanded = totals["pxs_expanded"] / tot_size
        metrics[config.SmoothingMetrics.COMPACTED] = [frac_reduced]
        metrics[config.SmoothingMetrics.DISPLACED] = [frac_expanded]
        metrics[config.SmoothingMetrics.SM_QUALITY] = [
            frac_reduced - frac_expanded]
        metrics[config.SmoothingMetrics.COMPACTNESS] = [
            totals["compactness_smoothed"] / tot_size]
        metrics[config.SmoothingMetrics.DISPLACEMENT] = [
            totals["displacement"] / tot_size]
        metrics[config.SmoothingMetrics.SA_VOL_ABS] = [
            totals["SA_to_vol_smoothed"] / tot_size]
        metrics[config.SmoothingMetrics.SA_VOL] = [
            totals["SA_to_vol_ratio"] / tot_size]
        num_labels_orig = len(label_ids)
        metrics[config.SmoothingMetrics.LABEL_LOSS] = [
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
    return df_metrics, df_pxs


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
        flipud: True to flip along the final z-axis for mirrorred axes; 
            defaults to False.
    
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
        if flipud:
            # flip along z-axis for mirrored orientations
            transposed = np.flipud(transposed)
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


def match_atlas_labels(img_atlas, img_labels, flip=False, metrics=None):
    """Apply register profile settings to labels and match atlas image 
    accordingly.
    
    Args:
        img_labels (:obj:`sitk.Image`): Labels image.
        img_ref (:obj:`sitk.Image`): Reference image, such as histology.
        flip (bool): True to rotate images 180deg around the final z axis; 
            defaults to False.
        metrics (:obj:`dict`): Dictionary to store metrics; defaults to 
            None, in which case metrics will not be measured.
    
    Returns:
        Tuple of ``img_atlas``, the updated atlas; ``img_labels``, the 
        updated labels; ``img_borders``, a new (:obj:`sitk.Image`) of the 
        same shape as the prior images except an extra channels dimension 
        as given by :func:``_curate_labels``; and ``df_smoothing``, a 
        data frame of smoothing stats, or None if smoothing was not performed.
    """
    pre_plane = config.register_settings["pre_plane"]
    extend_labels = config.register_settings["extend_labels"]
    mirror = config.register_settings["labels_mirror"]
    is_mirror = extend_labels["mirror"]
    edge = config.register_settings["labels_edge"]
    expand = config.register_settings["expand_labels"]
    rotate = config.register_settings["rotate"]
    smooth = config.register_settings["smooth"]
    crop = config.register_settings["crop_to_labels"]
    affine = config.register_settings["affine"]
    far_hem_neg = config.register_settings["make_far_hem_neg"]
    
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
        img_atlas = sitk_io.replace_sitk_with_numpy(img_atlas, img_atlas_np)
        img_labels = sitk_io.replace_sitk_with_numpy(img_labels, img_labels_np)
    
    # curate labels
    mask_lbls = None  # mask of fully extended/mirrored labels for cropping
    extis = None  # extension indices of 1st labeled, then unlabeled planes
    if all(extend_labels.values()):
        # include any lateral extension and mirroring
        img_labels_np, extis, df_smoothing = (
            _curate_labels(
                img_labels, img_atlas, mirror, edge, expand, rotate, smooth, 
                affine))
    else:
        # turn off lateral extension and/or mirroring
        img_labels_np, _, df_smoothing = _curate_labels(
            img_labels, img_atlas, mirror if is_mirror else None, 
            edge if extend_labels["edge"] else None, expand, rotate, smooth, 
            affine)
        if metrics or crop and (mirror is not None or edge is not None):
            print("\nCurating labels with extension/mirroring only "
                  "for measurements and any cropping:")
            resize = is_mirror and mirror is not None
            lbls_np_mir, extis, _ = _curate_labels(
                img_labels, img_atlas, mirror, edge, expand, rotate, None, 
                affine, resize)
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
    if is_mirror and mirror is not None:
        # TODO: consider removing dup since not using
        dup = config.register_settings["labels_dup"]
        img_atlas_np = mirror_planes(
            img_atlas_np, extis[1], start_dup=dup)
    
    if crop:
        # crop atlas to the mask of the labels with some padding
        img_labels_np, img_atlas_np, crop_sl = plot_3d.crop_to_labels(
            img_labels_np, img_atlas_np, mask_lbls)
        if crop_sl[0].start > 0:
            # offset extension indices and crop labels mask
            extis = tuple(n - crop_sl[0].start for n in extis)
            if mask_lbls is not None:
                mask_lbls = mask_lbls[tuple(crop_sl)]

    if far_hem_neg and np.all(img_labels_np >= 0):
        # unmirrored images may have only pos labels, while metrics assume 
        # that the far hem is neg; invert pos labels there if they are >=1/3 of 
        # total labels, not just spillover from the near side
        half_lbls = img_labels_np[extis[1]:]
        if (np.sum(half_lbls < 0) == 0 and
                np.sum(half_lbls != 0) > np.sum(img_labels_np != 0) / 3):
            print("less than half of labels in right hemisphere are neg; "
                  "inverting all pos labels in x >= {} (shape {}) for "
                  "sided metrics".format(extis[1], img_labels_np.shape))
            half_lbls[half_lbls > 0] *= -1
    
    if metrics is not None:
        
        # meas DSC of labeled hemisphere, using the sagittal midline 
        # to define the hemispheric boundaries
        dsc = measure_overlap_combined_labels(
            sitk.GetImageFromArray(img_atlas_np[:extis[1]]), 
            sitk.GetImageFromArray(img_labels_np[:extis[1]]), 
            config.register_settings["overlap_meas_add_lbls"])
        metrics[config.AtlasMetrics.DSC_ATLAS_LABELS_HEM] = [dsc]
        
        # meas frac of hemisphere that is unlabeled using "mirror" bounds
        thresh = config.register_settings["atlas_threshold_all"]
        thresh_atlas = img_atlas_np > thresh
        # some edge planes may be partially labeled
        lbl_edge = np.logical_and(
            img_labels_np[:extis[0]] != 0, thresh_atlas[:extis[0]])
        # simply treat rest of hem as labeled to focus on unlabeled lat portion
        metrics[config.AtlasMetrics.LAT_UNLBL_VOL] = 1 - (
            (np.sum(lbl_edge) + np.sum(thresh_atlas[extis[0]:extis[1]])) 
            / np.sum(thresh_atlas[:extis[1]]))

        # meas frac of planes that are at least partially labeled, using 
        # mask labels since from atlas portion that should be labeled
        frac = 0   # mask labels fully covered, so fully labeled by this def
        if mask_lbls is not None:
            mask_lbls_unrot = mask_lbls
            lbls_unrot = img_labels_np
            if rotate:
                # un-rotate so sagittal planes are oriented as orig drawn
                for rot in rotate[::-1]:
                    mask_lbls_unrot = plot_3d.rotate_nd(
                        mask_lbls_unrot, -rot[0], rot[1], 0)
                    lbls_unrot = plot_3d.rotate_nd(
                        lbls_unrot, -rot[0], rot[1], 0)
            planes_lbl = 0
            planes_tot = 0.
            for i in range(extis[1]):
                if not np.all(mask_lbls_unrot[i] == 0):
                    # plane that should be labeled
                    planes_tot += 1
                    if not np.all(lbls_unrot[i] == 0):
                        # plane is at least partially labeled
                        planes_lbl += 1
            if planes_tot > 0:
                frac = 1 - (planes_lbl / planes_tot)
        metrics[config.AtlasMetrics.LAT_UNLBL_PLANES] = frac
    
    imgs_np = (img_atlas_np, img_labels_np)
    if pre_plane:
        # transpose back to original orientation
        imgs_np, _ = plot_support.transpose_images(
            pre_plane, imgs_np, rev=True)
    
    # convert back to sitk img and transpose if necessary
    imgs_sitk = (img_atlas, img_labels)
    imgs_sitk_replaced = []
    for img_np, img_sitk in zip(imgs_np, imgs_sitk):
        if img_np is not None:
            img_sitk = sitk_io.replace_sitk_with_numpy(img_sitk, img_np)
            if pre_plane is None:
                # plane settings is for post-processing; 
                # TODO: check if 90deg rot is nec for yz
                rotate = 1 if config.plane in config.PLANE[1:] else 0
                if flip: rotate += 2
                img_sitk = transpose_img(
                    img_sitk, config.plane, rotate, flipud=True)
        imgs_sitk_replaced.append(img_sitk)
    img_atlas, img_labels = imgs_sitk_replaced
    
    return img_atlas, img_labels, df_smoothing


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
    img_atlas, path_atlas = sitk_io.read_sitk(
        os.path.join(atlas_dir, config.RegNames.IMG_ATLAS.value))
    img_labels, _ = sitk_io.read_sitk(
        os.path.join(atlas_dir, config.RegNames.IMG_LABELS.value))
    
    # prep export paths
    target_dir = atlas_dir + "_import"
    basename = os.path.basename(atlas_dir)
    df_base_path = os.path.join(target_dir, basename) + "_{}"
    df_metrics_path = df_base_path.format(config.PATH_ATLAS_IMPORT_METRICS)
    name_prefix = os.path.join(target_dir, basename) + ".czi"
    
    # set up condition
    overlap_meas_add = config.register_settings["overlap_meas_add_lbls"]
    extend_labels = config.register_settings["extend_labels"]
    if any(extend_labels.values()):
        cond = "extended" 
    else:
        # show baseline DSC of atlas to labels before any processing
        cond = "original"
        print("\nRaw DSC before import:")
        measure_overlap_combined_labels(
            img_atlas, img_labels, overlap_meas_add)
    
    # prep metrics
    metrics = {
        config.AtlasMetrics.SAMPLE: [basename], 
        config.AtlasMetrics.REGION: config.REGION_ALL, 
        config.AtlasMetrics.CONDITION: cond, 
    }
    
    # match atlas and labels to one another
    img_atlas, img_labels, df_smoothing = match_atlas_labels(
        img_atlas, img_labels, metrics=metrics)
    
    truncate = config.register_settings["truncate_labels"]
    if truncate:
        # truncate labels
        img_labels_np = truncate_labels(
            sitk.GetArrayFromImage(img_labels), *truncate)
        img_labels = sitk_io.replace_sitk_with_numpy(img_labels, img_labels_np)
    
    # show labels
    img_labels_np = sitk.GetArrayFromImage(img_labels)
    label_ids = np.unique(img_labels_np)
    print("number of labels: {}".format(label_ids.size))
    print(label_ids)
    
    # whole atlas stats; measure DSC if processed and prep dict for data frame
    print("\nDSC after import:")
    dsc = measure_overlap_combined_labels(
        img_atlas, img_labels, overlap_meas_add)
    # use lower threshold for compactness measurement to minimize noisy 
    # surface artifacts
    img_atlas_np = sitk.GetArrayFromImage(img_atlas)
    thresh = config.register_settings["atlas_threshold_all"]
    thresh_atlas = img_atlas_np > thresh
    compactness = plot_3d.compactness(
        plot_3d.perimeter_nd(thresh_atlas), thresh_atlas)
    metrics[config.AtlasMetrics.DSC_ATLAS_LABELS] = [dsc]
    metrics[config.SmoothingMetrics.COMPACTNESS] = [compactness]
    
    # write images with atlas saved as Clrbrain/Numpy format to 
    # allow opening as an image within Clrbrain alongside the labels image
    imgs_write = {
        config.RegNames.IMG_ATLAS.value: img_atlas, 
        config.RegNames.IMG_LABELS.value: img_labels}
    sitk_io.write_reg_images(
        imgs_write, name_prefix, copy_to_suffix=True, 
        ext=os.path.splitext(path_atlas)[1])
    detector.resolutions = [img_atlas.GetSpacing()[::-1]]
    img_ref_np = sitk.GetArrayFromImage(img_atlas)
    img_ref_np = img_ref_np[None]
    importer.save_np_image(img_ref_np, name_prefix, 0)
    
    if df_smoothing is not None:
        # write smoothing metrics to CSV with identifier columns
        df_smoothing_path = df_base_path.format(config.PATH_SMOOTHING_METRICS)
        df_smoothing[config.AtlasMetrics.SAMPLE.value] = basename
        df_smoothing[config.AtlasMetrics.REGION.value] = config.REGION_ALL
        df_smoothing[config.AtlasMetrics.CONDITION.value] = "smoothed"
        df_smoothing.loc[
            df_smoothing[config.SmoothingMetrics.FILTER_SIZE.value] == 0,
            config.AtlasMetrics.CONDITION.value] = "unsmoothed"
        stats.data_frames_to_csv(
            df_smoothing, df_smoothing_path, 
            sort_cols=config.SmoothingMetrics.FILTER_SIZE.value)

    print("\nImported {} whole atlas stats:".format(basename))
    stats.dict_to_data_frame(metrics, df_metrics_path, show="\t")
    
    if show:
        sitk.Show(img_atlas)
        sitk.Show(img_labels)


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
        fixed_binary_img = sitk_io.replace_sitk_with_numpy(
            fixed_binary_img, fixed_binary_np)
    transformed_binary_img = sitk.BinaryThreshold(
        transformed_img, transformed_thresh, transformed_thresh_up)
    overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_filter.Execute(fixed_binary_img, transformed_binary_img)
    total_dsc = overlap_filter.GetDiceCoefficient()
    #sitk.Show(fixed_binary_img)
    #sitk.Show(transformed_binary_img)
    print("foreground DSC: {}\n".format(total_dsc))
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
    fg_img_sitk = sitk_io.replace_sitk_with_numpy(labels_sitk, fg_img)
    return fg_img_sitk


def measure_overlap_combined_labels(fixed_img, labels_img, add_lbls=None):
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
    print("DSC of thresholded fixed image compared with combined labels:")
    return measure_overlap(
        fixed_img, lbls_fg, transformed_thresh=1, add_fixed_mask=mask)
