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
    overlap = np.ceil(np.multiply(scaling_factor, overlap_factor)).astype(int)
    max_pixels = np.ceil(np.multiply(scaling_factor, max_pixels_factor)).astype(int)
    print("overlap: {}, max_pixels: {}".format(overlap, max_pixels))
    num_units = _num_units(size, max_pixels)
    #print("num_units: {}".format(num_units))
    sub_rois = np.zeros(num_units, dtype=object)
    sub_rois_offsets = np.zeros(np.append(num_units, 3))
    print("sub_rois_offsets shape: {}".format(sub_rois_offsets.shape))
    for z in range(num_units[0]):
        for y in range(num_units[1]):
            for x in range(num_units[2]):
                coord = (z, y, x)
                bounds = [_bounds_side(size, max_pixels, overlap, coord, axis) for axis in range(3)]
                #print("bounds: {}".format(bounds))
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

def _compare_last_roi(blobs, coord, axis, blob_rois, region, tol, sub_rois, sub_rois_offsets):
    if coord[axis] > 0:
        coord_last = list(coord)
        coord_last[axis] = coord_last[axis] - 1
        coord_last_tup = tuple(coord_last)
        blobs_ref = blob_rois[coord_last_tup]
        if blobs_ref is not None:
            #overlap_slices = [slice(None)] * roi_size
            #print("blobs_ref:\n{}".format(blobs_ref))
            size = sub_rois[coord_last_tup].shape[axis]
            offset_axis = sub_rois_offsets[coord_last_tup][axis]
            bound_start = offset_axis + size - tol[axis]
            #bound_end = sub_rois_offsets[coord][axis]
            bound_end = bound_start + tol[axis]
            '''
            print("overlap is from {} to {} at coord_last_tup {} in axis {}"
                  .format(bound_start, bound_end, coord_last_tup, axis))
            print("offset last: {}, current: {}"
                  .format(sub_rois_offsets[coord_last_tup], sub_rois_offsets[coord]))
            '''
            blobs_ref_ol = blobs_ref[blobs_ref[:, axis] >= bound_start]
            blobs_ol = blobs[blobs[:, axis] < bound_end]
            '''
            print("checking overlapping blobs_ol:\n{}\nagaginst blobs_ref_ol from {}:\n{}"
                  .format(blobs_ol, coord_last, blobs_ref_ol))
            '''
            blobs_ol_pruned = detector.remove_close_blobs(blobs_ol, blobs_ref_ol, region, tol)
            blobs_pruned = np.concatenate((blobs_ol_pruned, blobs[blobs[:, axis] >= tol[axis]]))
            #print("non-overlapping blobs to add:\n{}".format(blobs_pruned))
    return blobs

def prune_overlapping_blobs(blob_rois, region, tol, sub_rois, sub_rois_offsets):
    blobs_all = None
    print("pruning overlapping blobs with tolerance {}".format(tol))
    for z in range(blob_rois.shape[0]):
        for y in range(blob_rois.shape[1]):
            for x in range(blob_rois.shape[2]):
                coord = (z, y, x)
                blobs = blob_rois[coord]
                #print("checking blobs in {}:\n{}".format(coord, blobs))
                print("checking blobs in {}".format(coord))
                if blobs is None:
                    print("no blobs to add, skipping")
                elif blobs_all is None:
                    print("initializing master blobs list")
                    blobs_all = blobs
                else:
                    for axis in range(len(coord)):
                        blobs = _compare_last_roi(blobs, coord, axis, blob_rois, 
                                                  region, tol, sub_rois, sub_rois_offsets)
                        blob_rois[coord] = blobs
                    blobs_all = np.concatenate((blobs_all, blobs))
    return blobs_all

if __name__ == "__main__":
    print("Starting chunking...")
    
    # tests splitting and remerging
    overlap_base = 1
    detector.resolutions = [[6.6, 1.1, 1.1]]
    roi = np.arange(5 * 4 * 4)
    roi = roi.reshape((5, 4, 4))
    print("roi:\n{}".format(roi))
    sub_rois, overlap, sub_rois_offsets = stack_splitter(roi, max_pixels_factor_denoise)
    print("sub_rois shape: {}".format(sub_rois.shape))
    print("sub_rois:\n{}".format(sub_rois))
    print("overlap: {}".format(overlap))
    print("sub_rois_offsets:\n{}".format(sub_rois_offsets))
    merged = merge_split_stack(sub_rois, overlap)
    print("merged:\n{}".format(merged))
    print("merged shape: {}".format(merged.shape))
    print("test roi == merged: {}".format(np.all(roi == merged)))
    