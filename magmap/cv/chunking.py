# Chunking image stacks
# Author: David Young, 2017, 2020
"""Divides a region into smaller chunks and reassembles it."""

import multiprocessing as mp

import numpy as np

from magmap.settings import config
from magmap.cv import detector
from magmap.io import libmag

#: int: Factor to multiply by scaling for maximum number of pixels per
# sub ROI for overlap.
OVERLAP_FACTOR = 5


def set_mp_start_method(val=None):
    """Set the multiprocessing start method.

    If the start method has already been applied, will skip.

    Args:
        val (str): Start method to set; defaults to None to use the default
            for the platform. If the given method is not available for the
            platform, the default method will be used instead.

    Returns:
        str: The applied start method.

    """
    if val is None:
        val = config.roi_profile["mp_start"]
    avail_start_methods = mp.get_all_start_methods()
    if val not in avail_start_methods:
        val = avail_start_methods[0]
    try:
        mp.set_start_method(val)
        print("set multiprocessing start method to", val)
    except RuntimeError:
        print("multiprocessing start method already set to {}, will skip"
              .format(mp.get_start_method(False)))
    return val


def is_fork():
    """Check if the multiprocessing start method is set to "fork".

    Returns:
        bool: True if the start method is "fork", False if otherwise.

    """
    return mp.get_start_method(False) == "fork"


def get_mp_pool():
    """Get a multiprocessing ``Pool`` object, configured based on ``config``
    settings.
    
    Returns:
        :obj:`multiprocessing.Pool`: Pool object with number of processes
        and max tasks per process determined by command-line and the main
        (first) ROI profile settings.

    """
    prof = config.get_roi_profile(0)
    max_tasks = None if not prof else prof["mp_max_tasks"]
    print("Setting up multiprocessing pool with {} processes (None uses all "
          "available)\nand max tasks of {} before replacing processes (None "
          "does not replace processes)".format(config.cpus, max_tasks))
    return mp.Pool(processes=config.cpus, maxtasksperchild=max_tasks)


def calc_overlap():
    """Calculate overlap based on scaling factor and :const:``OVERLAP_FACTOR``.
    
    Returns:
        Overlap as an array in the same shape and order as in 
        :attr:``detector.resolutions``.
    """
    return np.ceil(np.multiply(detector.calc_scaling_factor(),
                               OVERLAP_FACTOR)).astype(int)


def _num_units(size, max_pixels):
    """Calculates the shape of sub-regions that would comprise a total
    shape of ``size`` with ``max_pixels`` per dimension.
    
    Args:
        size (List[int]): Shape of the entire region.
        max_pixels (int): Max number of pixels per dimension.
    
    Returns:
        :obj:`np.ndarray`: Sequence of number of sub-regions for each
        dimension in ``size``.
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
        int, int: Boundary of sides for the given ``axis`` as ``start, end``.
    """
    pixels = max_pixels[axis]
    start = coord[axis] * pixels
    end = start + pixels
    if overlap is not None:
        end += overlap[axis]
    if end > size[axis]:
        end = size[axis]
    return int(start), int(end)


def stack_splitter(shape, max_pixels, overlap=None):
    """Split a stack into multiple sub regions.
    
    Args:
        shape (Tuple[int]): Shape of the stack to split.
        max_pixels (Tuple[int]): Max pixels for each side in (z, y, x) order.
        overlap (Tuple[int]): Overlap size between sub-ROIs. Defaults to None
            for no overlap.
    
    Return:
        :obj:`np.ndarray`, :obj:`np.ndarray`: Tuple of
        ``sub_roi_slices, sub_rois_offsets``, where
        ``sub_roi_slices`` is a Numpy object array where each element contains
        a tuple of slice objects defining the corresponding sub-region at
        that position, and ``sub_rois_offsets`` is a Numpy array of
        corresponding offsets for each sub-ROI in (z, y, x) order coordinates.
    """
    # prepare the array containing sub ROI slices with type object so that it
    # can contain an arbitrary object of any size and channels, accessible by
    # (z, y, x) coordinates of the chunk, as well as offset for 
    # coordinates of bottom corner for each sub ROI for transposing later
    num_units = _num_units(shape[:3], max_pixels)
    #print("num_units: {}".format(num_units))
    sub_rois_slices = np.zeros(num_units, dtype=object)
    sub_rois_offsets = np.zeros(np.append(num_units, 3))
    
    # fill with sub ROIs including overlap extending into next sub ROI 
    # except for the last one in each dimension
    for z in range(num_units[0]):
        for y in range(num_units[1]):
            for x in range(num_units[2]):
                coord = (z, y, x)
                bounds = [_bounds_side(shape, max_pixels, overlap, coord, axis)
                          for axis in range(3)]
                #print("bounds: {}".format(bounds))
                sub_rois_slices[coord] = (
                    slice(*bounds[0]), slice(*bounds[1]), slice(*bounds[2]))
                sub_rois_offsets[coord] = (
                    bounds[0][0], bounds[1][0], bounds[2][0])
    return sub_rois_slices, sub_rois_offsets


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
    libmag.printv("final_shape: {}".format(final_shape))
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
    """Combine all blobs into a master list so that each overlapping region
    will contain all blobs from all sub-ROIs that had blobs in those regions,
    obviating the need to pair sub-ROIs with one another.
    
    Args:
        blob_rois (:obj:`np.ndarray`): Blob from each sub-region defined by
            :meth:`stack_splitter`. Blobs are assumed to be a 2D array
            in the format ``[[z, y, x, ...], ...]``.

    Returns:
        :obj:`np.ndarray`: Merged blobs in 2D format of the format,
        ``[[z, y, x, ..., sub_roi_z, sub_roi_y, sub_roi_x], ...]``, where
        sub-ROI coordinates have been added as the final columns.

    """
    blobs_all = []
    for z in range(blob_rois.shape[0]):
        for y in range(blob_rois.shape[1]):
            for x in range(blob_rois.shape[2]):
                coord = (z, y, x)
                blobs = blob_rois[coord]
                #print("checking blobs in {}:\n{}".format(coord, blobs))
                if blobs is None:
                    libmag.printv("no blobs to add, skipping")
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


def main():
    """Test splitting and remerging."""
    config.resolutions = [[6.6, 1.1, 1.1]]
    roi = np.arange(5 * 4 * 4)
    roi = roi.reshape((5, 4, 4))
    print("roi:\n{}".format(roi))
    overlap = calc_overlap()
    sub_roi_slices, sub_rois_offsets = stack_splitter(roi.shape, [1, 2, 2])
    print("sub_rois shape: {}".format(sub_roi_slices.shape))
    print("sub_rois:\n{}".format(sub_roi_slices))
    print("overlap: {}".format(overlap))
    print("sub_rois_offsets:\n{}".format(sub_rois_offsets))
    for z in range(sub_roi_slices.shape[0]):
        for y in range(sub_roi_slices.shape[1]):
            for x in range(sub_roi_slices.shape[2]):
                coord = (z, y, x)
                sub_roi_slices[coord] = roi[sub_roi_slices[coord]]
    merged = merge_split_stack(sub_roi_slices, overlap)
    print("merged:\n{}".format(merged))
    print("merged shape: {}".format(merged.shape))
    print("test roi == merged: {}".format(np.all(roi == merged)))


if __name__ == "__main__":
    print("Starting chunking...")
    main()
