#!/bin/bash
# Chunking image stacks
# Author: David Young, 2017, 2018
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

from magmap import config
from magmap import detector
from magmap.io import lib_clrbrain

OVERLAP_FACTOR = 5

def calc_overlap():
    """Calculate overlap based on scaling factor and :const:``OVERLAP_FACTOR``.
    
    Returns:
        Overlap as an array in the same shape and order as in 
        :attr:``detector.resolutions``.
    """
    return np.ceil(np.multiply(detector.calc_scaling_factor(), 
                               OVERLAP_FACTOR)).astype(int)

def _num_units(size, max_pixels):
    """Calculates number of sub regions.
    
    Args:
        size: Size of the entire region
    
    Returns:
        The size of sub-ROIs array.
    """
    num = np.floor_divide(size, max_pixels)
    num[np.remainder(size, max_pixels) > 0] += 1
    return num.astype(np.int)

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
    end = start + pixels
    if overlap is not None:
        end += overlap[axis]
    if end > size[axis]:
        end = size[axis]
    return (int(start), int(end))

def stack_splitter(roi, max_pixels, overlap=None):
    """Splits a stack into multiple sub regions.
    
    Args:
        roi: The region of interest, a stack in (z, y, x, ...) dimensions.
        max_pixels: Max pixels for each side in (z, y, x) order.
        overlap: Overlap size between sub-ROIs. Defaults to None for no overlap.
    
    Return:
        Tuple of (sub_rois, sub_rois_offsets), where 
        ``sub_rois`` is a Numpy object array of smaller, potentially 
        overlapping sub regions split in (z, y, x) order, and 
        ``sub_rois_offsets`` is a Numpy array of offsets for each sub_roi 
        from the master ``roi`` array in (z, y, x) order coordinates.
    """
    size = roi.shape
    
    # prepare the array containing sub ROI slices with type object so that it
    # can contain an arbitrary object of any size and channels, accessible by
    # (z, y, x) coordinates of the chunk, as well as offset for 
    # coordinates of bottom corner for each sub ROI for transposing later
    num_units = _num_units(size[0:3], max_pixels)
    #print("num_units: {}".format(num_units))
    sub_rois = np.zeros(num_units, dtype=object)
    sub_rois_offsets = np.zeros(np.append(num_units, 3))
    
    # fill with sub ROIs including overlap extending into next sub ROI 
    # except for the last one in each dimension
    for z in range(num_units[0]):
        for y in range(num_units[1]):
            for x in range(num_units[2]):
                coord = (z, y, x)
                bounds = [_bounds_side(size, max_pixels, overlap, coord, axis) 
                          for axis in range(3)]
                #print("bounds: {}".format(bounds))
                sub_rois[coord] = roi[
                    slice(*bounds[0]), slice(*bounds[1]), slice(*bounds[2])]
                sub_rois_offsets[coord] = (
                    bounds[0][0], bounds[1][0], bounds[2][0])
    return sub_rois, sub_rois_offsets

def merge_split_stack(sub_rois, overlap):
    """Merges sub regions back into a single stack.
    
    See :func:``merge_split_stack2`` for a much faster implementation 
    if the final output array size is known beforehand.
    
    Args:
        sub_rois: Array of sub regions, in (z, y, x, ...) dimensions.
        overlap: Overlap size between sub-ROIs.
    
    Return:
        The merged stack.
    """
    size = sub_rois.shape
    merged = None
    if overlap.dtype != np.int:
        overlap = overlap.astype(np.int)
    for z in range(size[0]):
        merged_y = None
        for y in range(size[1]):
            merged_x = None
            for x in range(size[2]):
                coord = (z, y, x)
                sub_roi = sub_rois[coord]
                edges = list(sub_roi.shape[0:3])
                
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

def get_split_stack_total_shape(sub_rois, overlap=None):
    """Get the shape of a chunked stack.
    
    Useful for determining the final shape of a stack that has been 
    chunked and potentially scaled before merging the stack into 
    an output array of this shape.
    
    Attributes:
        sub_rois: Array of sub regions, in (z, y, x, ...) dimensions.
        overlap: Overlap size between sub-ROIs. Defaults to None for no overlap.
    
    Returns:
        The shape of the chunked stack after it would be merged.
    """
    size = sub_rois.shape
    shape_sub_roi = sub_rois[0, 0, 0].shape # for number of dimensions
    merged_shape = np.zeros(len(shape_sub_roi)).astype(np.int)
    final_shape = np.zeros(len(shape_sub_roi)).astype(np.int)
    edges = None
    for z in range(size[0]):
        for y in range(size[1]):
            for x in range(size[2]):
                coord = (z, y, x)
                sub_roi = sub_rois[coord]
                edges = list(sub_roi.shape[0:3])
                
                if overlap is not None:
                    # remove overlap if not at last sub_roi or row or column
                    for n in range(len(edges)):
                        if coord[n] != size[n] - 1:
                            edges[n] -= overlap[n]
                #print("edges: {}".format(edges))
                merged_shape[2] += edges[2]
            if final_shape[2] <= 0:
                final_shape[2] = merged_shape[2]
            merged_shape[1] += edges[1]
        if final_shape[1] <= 0:
            final_shape[1] = merged_shape[1]
        final_shape[0] += edges[0]
    channel_dim = 3
    if len(shape_sub_roi) > channel_dim:
        final_shape[channel_dim] = shape_sub_roi[channel_dim]
    lib_clrbrain.printv("final_shape: {}".format(final_shape))
    return final_shape

def merge_split_stack2(sub_rois, overlap, offset, output):
    """Merges sub regions back into a single stack, saving directly to 
    an output variable such as a memmapped array.
    
    Args:
        sub_rois: Array of sub regions, in (z, y, x, ...) dimensions.
        overlap: Overlap size between sub-ROIs given as an int seq in 
            z,y,x. Can be None for no overlap.
        offset: Axis offset for output array.
        output: Output array, such as a memmapped array to bypass 
            storing the merged array in RAM.
    
    Return:
        The merged stack.
    """
    size = sub_rois.shape
    merged_coord = np.zeros(3, dtype=np.int)
    sub_roi_shape = sub_rois[0, 0, 0].shape
    if offset > 0:
        # axis offset, such as skipping the time axis
        output = output[0]
    for z in range(size[0]):
        merged_coord[1] = 0
        for y in range(size[1]):
            merged_coord[2] = 0
            for x in range(size[2]):
                coord = (z, y, x)
                sub_roi = sub_rois[coord]
                edges = list(sub_roi.shape[0:3])
                
                if overlap is not None:
                    # remove overlap if not at last sub_roi or row or column
                    for n in range(len(edges)):
                        if coord[n] != size[n] - 1:
                            edges[n] -= overlap[n]
                sub_roi = sub_roi[:edges[0], :edges[1], :edges[2]]
                output[merged_coord[0]:merged_coord[0]+edges[0], 
                       merged_coord[1]:merged_coord[1]+edges[1], 
                       merged_coord[2]:merged_coord[2]+edges[2]] = sub_roi
                merged_coord[2] += sub_roi_shape[2]
            merged_coord[2] = 0
            merged_coord[1] += sub_roi_shape[1]
        merged_coord[1] = 0
        merged_coord[0] += sub_roi_shape[0]

def merge_blobs(blob_rois):
    # combine all blobs into a master list so that each overlapping region
    # will contain all blobs from all sub-ROIs that had blobs in those regions,
    # obviating the need to pair sub-ROIs with one another
    blobs_all = []
    for z in range(blob_rois.shape[0]):
        for y in range(blob_rois.shape[1]):
            for x in range(blob_rois.shape[2]):
                coord = (z, y, x)
                blobs = blob_rois[coord]
                #print("checking blobs in {}:\n{}".format(coord, blobs))
                if blobs is None:
                    lib_clrbrain.printv("no blobs to add, skipping")
                else:
                    # add temporary tag with sub-ROI coordinate
                    extras = np.zeros((blobs.shape[0], 3), dtype=int)
                    extras[:] = coord
                    blobs = np.concatenate((blobs, extras), axis=1)
                    blobs_all.append(blobs)
    if len(blobs_all) > 0:
        blobs_all = np.vstack(blobs_all)
    else:
        blobs_all = None
    return blobs_all

if __name__ == "__main__":
    print("Starting chunking...")
    
    # tests splitting and remerging
    overlap_base = 1
    config.resolutions = [[6.6, 1.1, 1.1]]
    roi = np.arange(5 * 4 * 4)
    roi = roi.reshape((5, 4, 4))
    print("roi:\n{}".format(roi))
    overlap = calc_overlap()
    sub_rois, sub_rois_offsets = stack_splitter(roi, [1, 2, 2])
    print("sub_rois shape: {}".format(sub_rois.shape))
    print("sub_rois:\n{}".format(sub_rois))
    print("overlap: {}".format(overlap))
    print("sub_rois_offsets:\n{}".format(sub_rois_offsets))
    merged = merge_split_stack(sub_rois, overlap)
    print("merged:\n{}".format(merged))
    print("merged shape: {}".format(merged.shape))
    print("test roi == merged: {}".format(np.all(roi == merged)))
    