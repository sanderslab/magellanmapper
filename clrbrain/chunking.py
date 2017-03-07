#!/bin/bash
# Chunking image stacks
# Author: David Young, 2017
"""Divides a region into smaller chunks and reassembles it.

Attributes:
    max_pixels_factor_denoise: Factor to multiply by scaling
        for maximum number of pixels per sub ROI when denoising.
        See detector.calc_scaling_factor() for scaling.
    max_pixels_factor_segment: Factor to multiply by scaling
        for maximum number of pixels per sub ROI when segmenting.
    overlap_factor: Factor to multiply by scaling
        for maximum number of pixels per sub ROI for overlap.
"""

import numpy as np
import math

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
    # overlap and max pixels per sub ROI are each dependent on scaling
    scaling_factor = detector.calc_scaling_factor()
    overlap = np.ceil(np.multiply(scaling_factor, overlap_factor)).astype(int)
    max_pixels = np.ceil(np.multiply(scaling_factor, max_pixels_factor)).astype(int)
    print("overlap: {}, max_pixels: {}".format(overlap, max_pixels))
    
    # prepare the array containing sub ROI slices with type object so that it
    # can contain an arbitrary object of any size, as well as offset for 
    # coordinates of bottom corner for each sub ROI for transposing later
    num_units = _num_units(size, max_pixels)
    #print("num_units: {}".format(num_units))
    sub_rois = np.zeros(num_units, dtype=object)
    sub_rois_offsets = np.zeros(np.append(num_units, 3))
    print("sub_rois_offsets shape: {}".format(sub_rois_offsets.shape))
    
    # fill with sub ROIs including overlap extending into next sub ROI except for 
    # the last one in each dimension
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
                
                # add back the non-overlapping region to build an x-direction
                # array, using concatenate to avoid copying the original array
                if merged_x is None:
                    merged_x = sub_roi
                else:
                    merged_x = np.concatenate((merged_x, sub_roi), axis=2)
            # add back non-overlapping regions from each x to build xy
            if merged_y is None:
                merged_y = merged_x
            else:
                merged_y = np.concatenate((merged_y, merged_x), axis=1)
        # add back non-overlapping regions from xy to build xyz
        if merged is None:
            merged = merged_y
        else:
            merged = np.concatenate((merged, merged_y), axis=0)
    return merged

def _compare_last_roi(blobs, coord, axis, blob_rois, region, tol, sub_rois, sub_rois_offsets):
    """Compares blobs in a sub ROI with the blobs in the immediately preceding sub ROI along
    the given axis
    
    Params:
        blobs: Numpy array of segments to display in the subplot, which 
            can be None. Segments are generally given as an (n, p)
            dimension array, where each segment is at least of (z, y, x, radius)
            elements.
        coord: Coordinate of the sub ROI, given as (z, y, x).
        axis: Axis along which to check.
        blob_rois: An array of blob arrays, where each element contains the blobs that
            were detected within the corresponding sub region within the image stack.
        region: Slice within each blob to check, such as slice(0, 2) to check
            for (z, row, column).
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
        sub_rois: Array of sub regions, in (z, y, x) dimensions.
        sub_rois_offsets: Array of offsets for each sub_roi, in
            (z, y, x) dimensions.
    
    Returns:
        Array of blobs without blobs that are close in the overlapping region to blobs
            already in the immediately preceding region.
    """
    #blobs = blob_rois[coord]
    #print("num of blobs to add: {}".format(blobs.shape[0]))
    #tol = np.ceil(np.multiply(overlap, 1.5)).astype(int)
    if coord[axis] > 0:
        # find the immediately preceding sub ROI
        coord_last = list(coord)
        coord_last[axis] = coord_last[axis] - 1
        coord_last_tup = tuple(coord_last)
        blobs_ref = blob_rois[coord_last_tup]
        if blobs_ref is not None:
            # find the boundaries for the overlapping region, giving extra padding to 
            # allow for slight differences in coordinates when the same blob was ID'ed
            # in different regions
            #print("blobs_ref:\n{}".format(blobs_ref))
            size = sub_rois[coord_last_tup].shape[axis]
            overlap = math.ceil(tol[axis] * 1.5)
            offset_axis = sub_rois_offsets[coord_last_tup][axis]
            bound_start = offset_axis + size - math.ceil(tol[axis] * 1.5)
            #bound_end = sub_rois_offsets[coord][axis]
            bound_end = bound_start + math.ceil(tol[axis] * 2) #tol[axis]
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
            
            # prune close blobs within the overlapping regions and add the remaining
            # blobs to the non-overlapping region
            blobs_ol_pruned = detector.remove_close_blobs(blobs_ol, blobs_ref_ol, region, tol)
            blobs_pruned = np.concatenate((blobs_ol_pruned, blobs[blobs[:, axis] >= bound_end]))
            #print("non-overlapping blobs to add:\n{}".format(blobs_pruned))
            #print("num of pruned blobs to add: {}".format(blobs_pruned.shape[0]))
            return blobs_pruned
            
    return blobs

def prune_overlapping_blobs(blob_rois, region, tol, sub_rois, sub_rois_offsets):
    """Removes overlapping blobs, which are blobs that are within a certain tolerance of
    one another.
    
    Params:
        blobs: Numpy array of segments to display in the subplot, which 
            can be None. Segments are generally given as an (n, p)
            dimension array, where each segment is at least of (z, y, x, radius)
            elements.
        blob_rois: An array of blob arrays, where each element contains the blobs that
            were detected within the corresponding sub region within the image stack.
        region: Slice within each blob to check, such as slice(0, 2) to check
            for (z, row, column).
        tol: Tolerance to check for closeness, given in the same format
            as region. Blobs that are equal to or less than the the absolute
            difference for all corresponding parameters will be pruned in
            the returned array.
        sub_rois: Array of sub regions, in (z, y, x) dimensions.
        sub_rois_offsets: Array of offsets for each sub_roi, in
            (z, y, x) dimensions.
    
    Returns:
        Array of all the blobs, minus any blobs that are close to one another in the
            overlapping regions.
    """
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
                    # checks immediately preceding sub ROI in each dimension; should
                    # not put back into blob_rois since other blob sets may need to
                    # detect pruned blobs when the same blob occurs in >2 sets that
                    # will not be diretly compared with one another
                    for axis in range(len(coord)):
                        blobs = _compare_last_roi(blobs, coord, axis, blob_rois, 
                                                  region, tol, sub_rois, sub_rois_offsets)
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
    