#!/bin/bash
# Chunking image stacks
# Author: David Young, 2017
"""Divides a region into smaller chunks and reassembles it.

Attributes:
    max_pixels: Maximum number of pixels in (x, y, z) dimensions.
    overlap_base: Base number of pixels for overlap, which will
        be scaled up by detector.scaling_factor.
"""

import numpy as np

from clrbrain import detector

max_pixels = (50, 50, 0.5) # (x, y, z) order
overlap_base = 5

def _num_units(size):
    """Calculates number of sub regions.
    
    Params:
        size: Size of the entire region
    
    Returns:
        The size of sub-ROIs array.
    """
    pixels = max_pixels[::-1]
    num = np.floor_divide(size, pixels)
    num[np.remainder(size, pixels) > 0] += 1
    return num

def _bounds_side(size, overlap, coord, axis):
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
    # max_pixels is in opposite (human) order (x, y, z)
    pixels = max_pixels[len(coord) - axis - 1]
    start = coord[axis] * pixels
    end = start + pixels + overlap
    if end > size[axis]:
        end = size[axis]
    return (start, end)

def stack_splitter(roi):
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
    overlap = int(overlap_base * detector.scaling_factor)
    num_units = _num_units(size)
    print("num_units: {}".format(num_units))
    sub_rois = np.zeros(num_units, dtype=object)
    sub_rois_offsets = np.zeros(np.append(num_units, 3))
    print("sub_rois_offsets shape: {}".format(sub_rois_offsets.shape))
    for z in range(num_units[0]):
        for y in range(num_units[1]):
            for x in range(num_units[2]):
                coord = (z, y, x)
                bounds = [_bounds_side(size, overlap, coord, axis) for axis in range(3)]
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
                        edges[n] -= overlap
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

def scale_max_pixels():
    """Scales the max pixels by detector.scaling_factor.
    """
    global max_pixels
    max_pixels = [int(detector.scaling_factor * n) for n in max_pixels]
    print("scaled max pixels: {}".format(max_pixels))

if __name__ == "__main__":
    print("Starting chunking...")
    #roi = np.arange(1920 * 1920 * 500)
    max_pixels = (2, 2, 3)
    overlap_base = 1
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
    #blobs = np.random.randint(0, high=1920, size=(10, 3))
    blobs = np.array([[1, 3, 4, 2.2342], [1, 8, 5, 3.13452], [1, 3, 4, 5.1234]])
    print("sample blobs: {}".format(blobs))
    end = 3
    blobs_unique = remove_duplicate_blobs(blobs, slice(0, end))
    print("blobs_unique through first {} elements: {}".format(end, blobs_unique))
    