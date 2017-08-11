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

from clrbrain import config
from clrbrain import detector

OVERLAP_FACTOR = 5

def calc_overlap():
    return np.ceil(np.multiply(detector.calc_scaling_factor(), 
                               OVERLAP_FACTOR)).astype(int)

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

'''
def super_stack_splitter(roi, max_pixels, overlap):
    num_units = _num_units(roi.shape, max_pixels)
    dims = []
    for z in range(num_units[0]):
        for y in range(num_units[1]):
            for x in range(num_units[2]):
                bounds = [_bounds_side(size, max_pixels, overlap, (z, y, x), axis) for axis in range(3)]
                offset = (bounds[0][0], bounds[1][0], bounds[2][0])
                size = (bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], bounds[2][1] - bounds[2][0])
                #print("bounds: {}".format(bounds))
                dims.append((offset, size))
    return dims
''' 

def stack_splitter(roi, max_pixels, overlap):
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
    print("total stack size: {}".format(size))
    
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
    return sub_rois, sub_rois_offsets

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

def _compare_last_roi(blobs, coord, axis, blob_rois, region, tol, sub_rois, 
                      sub_rois_offsets):
    """Compares blobs in a sub ROI with the blobs in the immediately preceding 
    sub ROI along the given axis
    
    Params:
        blobs: Numpy array of segments to display in the subplot, which 
            can be None. Segments are generally given as an (n, p)
            dimension array, where each segment is at least of (z, y, x, radius)
            elements.
        coord: Coordinate of the sub ROI, given as (z, y, x).
        axis: Axis along which to check.
        blob_rois: An array of blob arrays, where each element contains the 
            blobs that were detected within the corresponding sub region within 
            the image stack.
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
        blobs_pruned: Array of blobs without blobs that are close in the 
            overlapping region to blobs already in the immediately preceding 
            region.
        blobs_ref_shifted: A copy of the blobs array from the previous ROI in 
            blob_rois with tail values shifted to the mean of any corresponding 
            duplicate blob from blobs.
        coord_next_tup: The coordinates of the next ROI, useful in case
            blobs_ref_shifted needs to be replace the current array in 
            blobs_roi.
    """
    # only compare if a next ROI in the given axis exists
    blobs_pruned = blobs
    blobs_ref_shifted = None
    coord_next_tup = None
    if coord[axis] + 1 < sub_rois.shape[axis]:
        # find the immediately preceding sub ROI
        coord_next = list(coord)
        coord_next[axis] += 1
        coord_next_tup = tuple(coord_next)
        blobs_ref = blob_rois[coord_next_tup] # the next ROI
        if blobs_ref is not None:
            # find the boundaries for the overlapping region, giving extra 
            # padding to allow for slight differences in coordinates when the 
            # same blob was ID'ed in different regions
            #print("blobs_ref:\n{}".format(blobs_ref))
            size = sub_rois[coord].shape[axis]
            #tol_expand = math.ceil(tol[axis] * 1.5)
            tol_expand = tol[axis] * 2
            offset_axis = sub_rois_offsets[coord][axis]
            bound_start = offset_axis + size - tol_expand
            bound_end = sub_rois_offsets[coord_next_tup][axis] + tol_expand
            # 0.5*tol_expand longer than the end of the next sub ROI
            #bound_end = bound_start + math.ceil(tol_expand * 1.5)
            
            # overlapping blobs from current and next ("ref") ROI
            blobs_ol = blobs[blobs[:, axis] >= bound_start]
            blobs_ref_ol = blobs_ref[blobs_ref[:, axis] < bound_end]
            
            if config.verbose:
                print("Comparing blobs in axis {}, ROI {}"
                      .format(axis, coord))
                overlap_start = np.copy(sub_rois_offsets[coord])
                overlap_end = np.add(overlap_start, sub_rois[coord].shape)
                overlap_start[axis] = bound_start
                overlap_end[axis] = bound_end
                print("overlap from {} to {}"
                      .format(overlap_start, overlap_end))
                print("checking overlapping blobs_ol:\n{}\n"
                      "against blobs_ref_ol with tol {} from ROI {}:\n{}"
                      .format(blobs_ol[:, 0:4], tol, coord_next, 
                              blobs_ref_ol[:, 0:4]))
            
            # prune close blobs within the overlapping regions and add the 
            # remaining or shifted blobs to the non-overlapping region
            blobs_ol_pruned, blobs_ref_ol_shifted = (
                detector.remove_close_blobs(blobs_ol, blobs_ref_ol, 
                                            region, tol))
            blobs_pruned = np.concatenate(
                (blobs_ol_pruned, blobs[blobs[:, axis] < bound_start]))
            blobs_ref_shifted = np.concatenate(
                (blobs_ref_ol_shifted, 
                 blobs_ref[blobs_ref[:, axis] >= bound_end]))
    # check overlapping blobs from same region as before but 1 z below;
    # TODO: not yet working, now seemingly with more overlapping blobs
    if coord[axis] > 0 and axis != 0 and coord[axis] + 1 < sub_rois.shape[axis]:
        blobs = blobs_pruned
        # find the immediately preceding sub ROI
        coord_last = list(coord)
        coord_last[axis] += 1
        coord_last[0] -= 1
        coord_last_tup = tuple(coord_last)
        blobs_ref = blob_rois[coord_last_tup] # the last ROI
        if blobs_ref is not None:
            # find the boundaries for the overlapping region, giving extra 
            # padding to allow for slight differences in coordinates when the 
            # same blob was ID'ed in different regions
            tol_expand = tol[axis] * 2
            offset_end = np.add(sub_rois_offsets[coord], sub_rois[coord].shape)
            print(offset_end)
            
            # overlapping blobs from current and last ("ref") ROI
            mask_blobs_ol = np.all([
                blobs[:, axis] >= offset_end[axis] - tol[axis], 
                blobs[:, 0] < sub_rois_offsets[coord][0] + tol[0]], axis=0)
            blobs_ol = blobs[mask_blobs_ol]
            blobs_ref_ol = blobs_ref[np.all([
                blobs_ref[:, axis] < offset_end[axis], 
                blobs_ref[:, 0] >= sub_rois_offsets[coord][0]], axis=0)]
            
            if config.verbose:
                print("Comparing 2nd degree blobs in axis {}, ROI {} vs {}"
                      .format(axis, coord, coord_last_tup))
                if len(blobs_ol) > 0 and len(blobs_ref_ol) > 0:
                    print("checking overlapping blobs_ol:\n{}\n"
                          "against blobs_ref_ol with tol {} from ROI {}:\n{}"
                          .format(blobs_ol[:, 0:4], tol, coord_last, 
                                  blobs_ref_ol[:, 0:4]))
            
            # prune close blobs within the overlapping regions and add the 
            # remaining or shifted blobs to the non-overlapping region
            blobs_ol_pruned, _ = (
                detector.remove_close_blobs(blobs_ol, blobs_ref_ol, 
                                            region, tol))
            blobs_pruned = np.concatenate(
                (blobs_ol_pruned, blobs[np.invert(mask_blobs_ol)]))
    return blobs_pruned, blobs_ref_shifted, coord_next_tup

def prune_overlapping_blobs(blob_rois, region, tol, sub_rois, sub_rois_offsets):
    """Removes overlapping blobs, which are blobs that are within a certain 
    tolerance of one another, by comparing a given sub-ROI with the 
    immediately preceding sub-ROI.
    
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
                print("** checking blobs in ROI {}".format(coord))
                if blobs is None:
                    print("no blobs to add, skipping")
                elif blobs_all is None:
                    print("initializing master blobs list")
                    blobs_all = blobs
                else:
                    # checks immediately preceding sub ROI in each dimension; 
                    # puts the updated reference blobs from the next ROI into 
                    # for the shifted components and adds pruned blob list from
                    # current ROI into final list
                    for axis in range(len(coord)):
                        blobs, blobs_next, coord_next = _compare_last_roi(
                            blobs, coord, axis, blob_rois, region, tol, 
                            sub_rois, sub_rois_offsets)
                        if blobs_next is not None:
                            blob_rois[coord_next] = blobs_next
                    blobs_all = np.concatenate((blobs_all, blobs))
    # copy shifted coordinates to final coordinates
    '''
    np.set_printoptions(linewidth=200, threshold=10000)
    print("blobs_all:\n{}".format(blobs_all[:, 0:4] == blobs_all[:, 5:9]))
    '''
    blobs_all[:, 0:4] = blobs_all[:, 5:9]
    return blobs_all[:, 0:5]

def prune_overlapping_blobs2(blob_rois, region, overlap, tol, sub_rois, sub_rois_offsets):
    """Removes overlapping blobs, which are blobs that are within a certain 
    tolerance of one another, by comparing a given sub-ROI with the 
    immediately following sub-ROI.
    
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
    # combine all blobs into a master list so that each overlapping region
    # will contain all blobs from all sub-ROIs that had blobs in those regions,
    # obviating the need to pair sub-ROIs with one another
    blobs_all = None
    print("pruning overlapping blobs with tolerance {}".format(tol))
    for z in range(blob_rois.shape[0]):
        for y in range(blob_rois.shape[1]):
            for x in range(blob_rois.shape[2]):
                coord = (z, y, x)
                blobs = blob_rois[coord]
                #print("checking blobs in {}:\n{}".format(coord, blobs))
                if blobs is None:
                    print("no blobs to add, skipping")
                elif blobs_all is None:
                    print("initializing master blobs list")
                    blobs_all = blobs
                else:
                    blobs_all = np.concatenate((blobs_all, blobs))
    if blobs_all is None:
        return None
    
    # find the overlapping regions and compare blobs within them
    for z in range(sub_rois_offsets.shape[0]):
        for y in range(sub_rois_offsets.shape[1]):
            for x in range(sub_rois_offsets.shape[2]):
                coord = [z, y, x]
                print("** checking blobs in ROI {}".format(coord))
                offset = sub_rois_offsets[tuple(coord)]
                size = sub_rois[tuple(coord)].shape
                print("offset: {}, size: {}, overlap: {}, tol: {}".format(offset, size, overlap, tol))
                for axis in range(3):
                    axes = np.arange(3)
                    if coord[axis] + 1 < sub_rois_offsets.shape[axis]:
                        axes = np.delete(axes, axis) # get remaining axes
                        bounds = [
                            offset[axis] + size[axis] - overlap[axis] - tol[axis],
                            offset[axis] + size[axis] + tol[axis],
                            offset[axes[0]] - tol[axes[0]],
                            offset[axes[0]] + size[axes[0]] + tol[axes[0]],
                            offset[axes[1]] - tol[axes[1]],
                            offset[axes[1]] + size[axes[1]] + tol[axes[1]]
                        ]
                        print("axis {}, boundaries: {}".format(axis, bounds))
                        mask_blobs_ol = np.all([
                            blobs_all[:, axis] >= bounds[0], 
                            blobs_all[:, axis] < bounds[1],
                            blobs_all[:, axes[0]] >= bounds[2],
                            blobs_all[:, axes[0]] < bounds[3],
                            blobs_all[:, axes[1]] >= bounds[4],
                            blobs_all[:, axes[1]] < bounds[5]], axis=0)
                        blobs_ol = blobs_all[mask_blobs_ol]
                        #print("len before before: {}".format(len(blobs_all)))
                        blobs_ol_pruned = detector.remove_close_blobs_within_sorted_array(blobs_ol, region, tol)
                        print("blobs without close duplicates:\n{}".format(blobs_ol_pruned))
                        
                        if blobs_ol_pruned is not None:
                            #print("len before: {}, len blobs_ol: {}".format(len(blobs_all), len(blobs_ol)))
                            blobs_all = blobs_all[np.invert(mask_blobs_ol)]
                            #print("len after: {}".format(len(blobs_all)))
                            blobs_all = np.concatenate((blobs_all, blobs_ol_pruned))
                            #print("len after after: {}".format(len(blobs_all)))
                        
    # copy shifted coordinates to final coordinates
    #print("blobs_all:\n{}".format(blobs_all[:, 0:4] == blobs_all[:, 5:9]))
    blobs_all[:, 0:4] = blobs_all[:, 6:]
    return blobs_all#[:, 0:6]

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
    