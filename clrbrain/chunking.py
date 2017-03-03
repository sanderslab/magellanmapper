#!/bin/bash
# Chunking image stacks
# Author: David Young, 2017
"""Divides a region into smaller chunks and reassembles it.

Attributes:
    max_pixels: Maximum number of pixels in (x, y, z) dimensions.
    overlap_base: Base number of pixels for overlap, which will
        be scaled by detector.resolutions.
"""

import numpy as np

from clrbrain import detector

max_pixels_factor_denoise = 25
max_pixels_factor_segment = 100
overlap_factor = 5

def _num_units(size, max_pixels):
    """Calculates number of sub regions.
    
    Params:
        size: Size of the entire region
    
    Returns:
        The size of sub-ROIs array.
    """
    num = np.floor_divide(size, max_pixels)
    num[np.remainder(size, max_pixels) > 0] += 1
    return num

def _bounds_side(size, max_pixels, overlap, coord, axis):
    """Calculates the boundaries of a side based on where in the
    ROI the current sub-ROI is.
    
    Attributes:
        size: Size in (z, y, x) order.
        overlap: Overlap size between sub-ROIs.
        coord: Coordinates of the sub-ROI, in (z, y, x) order.
        axis: The axis to calculate.
    
    Returns:
        Boundary of sides in (z, y, x) order as a (start, end) tuple.
    """
    pixels = max_pixels[axis]
    start = coord[axis] * pixels
    end = start + pixels + overlap[axis]
    if end > size[axis]:
        end = size[axis]
    return (start, end)

def stack_splitter(roi, max_pixels_factor, overlap_factor=overlap_factor):
    """Splits a stack into multiple sub regions.
    
    Params:
        roi: The region of interest, a stack in (z, y, x) dimensions.
    
    Return:
        sub_rois: Array of sub regions, in (z, y, x) dimensions.
        overlap: The overlap size, in pixels.
        sub_rois_offsets: Array of offsets for each sub_roi, in
            (z, y, x) dimensions.
    """
    size = roi.shape
    #overlap = calc_tolerance()
    scaling_factor = detector.calc_scaling_factor()
    overlap = np.multiply(scaling_factor, overlap_factor)
    max_pixels = np.multiply(scaling_factor, max_pixels_factor)
    num_units = _num_units(size, max_pixels)
    print("num_units: {}".format(num_units))
    sub_rois = np.zeros(num_units, dtype=object)
    sub_rois_offsets = np.zeros(np.append(num_units, 3))
    print("sub_rois_offsets shape: {}".format(sub_rois_offsets.shape))
    for z in range(num_units[0]):
        for y in range(num_units[1]):
            for x in range(num_units[2]):
                coord = (z, y, x)
                bounds = [_bounds_side(size, max_pixels, overlap, coord, axis) for axis in range(3)]
                print("bounds: {}".format(bounds))
                sub_rois[coord] = roi[slice(*bounds[0]), slice(*bounds[1]), slice(*bounds[2])]
                sub_rois_offsets[coord] = (bounds[0][0], bounds[1][0], bounds[2][0])
    return sub_rois, overlap, sub_rois_offsets

def merge_split_stack(sub_rois, overlap):
    """Merges sub regions back into a single stack.
    
    Params:
        sub_rois: Array of sub regions, in (z, y, x) dimensions.
    
    Return:
        The merged stack.
    """
    size = sub_rois.shape
    merged = None
    for z in range(size[0]):
        merged_y = None
        for y in range(size[1]):
            merged_x = None
            for x in range(size[2]):
                #sub_roi = _get_sub_roi(sub_rois, overlap, (z, y, x))
                coord = (z, y, x)
                sub_roi = sub_rois[coord]
                edges = list(sub_roi.shape)
                # remove overlap if not at last sub_roi or row or column
                for n in range(len(edges)):
                    if coord[n] != size[n] - 1:
                        edges[n] -= overlap[n]
                sub_roi = sub_roi[:edges[0], :edges[1], :edges[2]]
                if merged_x is None:
                    merged_x = sub_roi
                else:
                    merged_x = np.concatenate((merged_x, sub_roi), axis=2)
            if merged_y is None:
                merged_y = merged_x
            else:
                merged_y = np.concatenate((merged_y, merged_x), axis=1)
        if merged is None:
            merged = merged_y
        else:
            merged = np.concatenate((merged, merged_y), axis=0)
    return merged

def remove_duplicate_blobs(blobs, region):
    """Removes duplicate blobs.
    
    Params:
        blobs: The blobs, given as 2D array of [n, [z, row, column, radius]].
        region: Slice within each blob to check, such as slice(0, 2) to check
           for (z, row, column).
    
    Return:
        The blobs array with only unique elements.
    """
    # workaround while awaiting https://github.com/numpy/numpy/pull/7742
    # to become a reality, presumably in Numpy 1.13
    blobs_region = blobs[:, region]
    blobs_contig = np.ascontiguousarray(blobs_region)
    blobs_type = np.dtype((np.void, blobs_region.dtype.itemsize * blobs_region.shape[1]))
    blobs_contig = blobs_contig.view(blobs_type)
    _, unique_indices = np.unique(blobs_contig, return_index=True)
    print("removed {} duplicate blobs".format(blobs.shape[0] - unique_indices.size))
    return blobs[unique_indices]

def remove_close_blobs(blobs, blobs_master, region, tol):
    """Removes blobs that are close to one another.
    
    Params:
        blobs: The blobs to add, given as 2D array of [n, [z, row, column, 
            radius]].
        blobs_master: The list by which to check for close blobs, in the same
            format as blobs.
        region: Slice within each blob to check, such as slice(0, 2) to check
            for (z, row, column).
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
    
    Return:
        The blobs array without blobs falling inside the tolerance range.
    """
    # workaround while awaiting https://github.com/numpy/numpy/pull/7742
    # to become a reality, presumably in Numpy 1.13
    blobs_diffs = np.abs(blobs_master[:, region][:, None] - blobs[:, region])
    close_master, close = np.nonzero((blobs_diffs <= tol).all(2))
    pruned = np.delete(blobs, close, axis=0)
    print("removed {} close blobs".format(blobs.shape[0] - pruned.shape[0]))
    return pruned

def remove_close_blobs_within_array(blobs, region, tol):
    """Removes close blobs within a given array.
    
    Uses remove_close_blobs() to detect blobs close to one another inside
    the master array.
    
    Params:
        blobs: The blobs to add, given as 2D array of [n, [z, row, column, 
            radius]].
        region: Slice within each blob to check, such as slice(0, 2) to check
            for (z, row, column).
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
    
    Return:
        The blobs array without blobs falling inside the tolerance range.
    """
    if blobs is None:
        return
    blobs_all = None
    for blob in blobs:
        #print("blob: {}".format(blob))
        if blobs_all is None:
            blobs_all = np.array([blob])
        else:
            blobs_to_add = remove_close_blobs(np.array([blob]), blobs_all, region, tol)
            blobs_all = np.concatenate((blobs_all, blobs_to_add))
    return blobs_all

def _compare_last_roi(blobs, coord, axis, blob_rois, region, tol):
    if coord[axis] > 0:
        coord_last = list(coord)
        coord_last[axis] = coord_last[axis] - 1
        blobs_ref = blob_rois[tuple(coord_last)]
        print("checking against blobs_ref at {}:\n{}".format(coord_last, blobs_ref))
        if blobs_ref is not None:
            return remove_close_blobs(blobs, blobs_ref, region, tol)
    return blobs

def prune_overlapping_blobs(blob_rois, region, tol):
    blobs_all = None
    print("pruning overlapping blobs with tolerance {}".format(tol))
    for z in range(blob_rois.shape[0]):
        for y in range(blob_rois.shape[1]):
            for x in range(blob_rois.shape[2]):
                coord = (z, y, x)
                blobs = blob_rois[coord]
                print("checking blobs in {}:\n{}".format(coord, blobs))
                if blobs is None:
                    print("no blobs to add, skipping")
                elif blobs_all is None:
                    print("initializing master blobs list")
                    blobs_all = blobs
                else:
                    for axis in range(len(coord)):
                        blobs = _compare_last_roi(blobs, coord, axis, blob_rois, region, tol)
                    blobs_all = np.concatenate((blobs_all, blobs))
    return blobs_all

if __name__ == "__main__":
    print("Starting chunking...")
    
    # tests splitting and remerging
    #roi = np.arange(1920 * 1920 * 500)
    max_pixels = (2, 2, 3)
    overlap_base = 1
    detector.resolutions = [[6.6, 1.1, 1.1]]
    roi = np.arange(5 * 4 * 4)
    roi = roi.reshape((5, 4, 4))
    print("roi:\n{}".format(roi))
    sub_rois, overlap, sub_rois_offsets = stack_splitter(roi)
    print("sub_rois shape: {}".format(sub_rois.shape))
    print("sub_rois:\n{}".format(sub_rois))
    print("overlap: {}".format(overlap))
    print("sub_rois_offsets:\n{}".format(sub_rois_offsets))
    merged = merge_split_stack(sub_rois, overlap)
    print("merged:\n{}".format(merged))
    print("merged shape: {}".format(merged.shape))
    print("test roi == merged: {}".format(np.all(roi == merged)))
    
    # tests blob duplication removal
    blobs = np.array([[1, 3, 4, 2.2342], [1, 8, 5, 3.13452], [1, 3, 4, 5.1234],
                      [1, 3, 5, 2.2342], [3, 8, 5, 3.13452]])
    print("sample blobs:\n{}".format(blobs))
    end = 3
    blobs_unique = remove_duplicate_blobs(blobs, slice(0, end))
    print("blobs_unique through first {} elements:\n{}".format(end, blobs_unique))
    
    # tests removal of blobs within a given tolerance level
    tol = (1, 2, 2)
    blobs = remove_close_blobs_within_array(blobs, slice(0, end), tol)
    print("pruned sample blobs within tolerance {}:\n{}".format(tol, blobs))
    blobs_to_add = np.array([[1, 3, 5, 2.2342], [2, 10, 5, 3.13452], 
                             [2, 2, 4, 5.1234], [3, 3, 5, 2.2342]])
    print("blobs to add:\n{}".format(blobs_to_add))
    blobs_to_add = remove_close_blobs(blobs_to_add, blobs, slice(0, end), tol)
    print("pruned blobs to add:\n{}".format(blobs_to_add))
