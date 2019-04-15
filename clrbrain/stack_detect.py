#!/bin/bash
# Detect blobs within a chunked stack through multiprocessing
# Author: David Young, 2019
"""Stack blob detector.

Detect blobs within a stack that has been chunked to allow parallel 
processing.
"""

from enum import Enum
import multiprocessing as mp
import os
from time import time

import numpy as np
import pandas as pd

from clrbrain import chunking
from clrbrain import config
from clrbrain import detector
from clrbrain import lib_clrbrain
from clrbrain import plot_3d
from clrbrain import sqlite
from clrbrain import stats

class StackTimes(Enum):
    """Stack processing durations."""
    DETECTION = "Detection"
    PRUNING = "Pruning"
    TOTAL = "Total_stack"

class StackDetector(object):
    """Detect blobs within a stack in a way that allows multiprocessing 
    without global variables.
    
    Attributes:
        sub_rois: Numpy object array containing chunked sub-ROIs within a 
            stack.
        sub_rois_offsets: Numpy object array of the same shape as 
            ``sub_rois`` with offsets in z,y,x corresponding to each 
            sub-ROI. Offets are used to transpose blobs into 
            absolute coordinates.
        denoise_max_shape: Maximum shape of each unit within each sub-ROI 
            for denoising.
    """
    sub_rois = None
    sub_rois_offsets = None
    denoise_max_shape = None
    
    @classmethod
    def set_data(cls, sub_rois, sub_rois_offsets, denoise_max_shape):
        """Set the class attributes to be shared during multiprocessing."""
        cls.sub_rois = sub_rois
        cls.sub_rois_offsets = sub_rois_offsets
        cls.denoise_max_shape = denoise_max_shape
    
    @classmethod
    def detect_sub_roi(cls, coord):
        """Segment the ROI within an array of ROIs.
        
        Args:
            coord: Coordinate of the sub-ROI in the order (z, y, x).
        
        Returns:
            Tuple of the coordinate given back again to identify the 
            sub-ROI and an array of detected blobs.
        """
        sub_roi = cls.sub_rois[coord]
        lib_clrbrain.printv_format(
            "detecting blobs in sub-ROI at {} of {}, with shape {}...", 
            (coord, np.add(cls.sub_rois.shape, -1), sub_roi.shape))
        
        if cls.denoise_max_shape is not None:
            # further split sub-ROI for preprocessing locally
            denoise_rois, _ = chunking.stack_splitter(
                sub_roi, cls.denoise_max_shape)
            for z in range(denoise_rois.shape[0]):
                for y in range(denoise_rois.shape[1]):
                    for x in range(denoise_rois.shape[2]):
                        denoise_coord = (z, y, x)
                        denoise_roi = plot_3d.saturate_roi(
                            denoise_rois[denoise_coord], channel=config.channel)
                        denoise_roi = plot_3d.denoise_roi(
                            denoise_roi, channel=config.channel)
                        denoise_rois[denoise_coord] = denoise_roi
            
            # re-merge into one large ROI (the image stack) in preparation for 
            # segmenting with differently sized chunks
            merged_shape = chunking.get_split_stack_total_shape(denoise_rois)
            merged = np.zeros(
                tuple(merged_shape), dtype=denoise_rois[0, 0, 0].dtype)
            chunking.merge_split_stack2(denoise_rois, None, 0, merged)
            sub_roi = merged
        
        segments = detector.detect_blobs(sub_roi, config.channel)
        offset = cls.sub_rois_offsets[coord]
        #print("segs before (offset: {}):\n{}".format(offset, segments))
        if segments is not None:
            # shift both coordinate sets (at beginning and end of array) to 
            # absolute positioning, using the latter set to store shifted 
            # coordinates based on duplicates and the former for initial 
            # positions to check for multiple duplicates
            detector.shift_blob_rel_coords(segments, offset)
            detector.shift_blob_abs_coords(segments, offset)
            #print("segs after:\n{}".format(segments))
        return coord, segments
    
def make_subimage_name(base, offset, shape):
    """Make name of subimage for a given offset and shape.
    
    Args:
        base: Start of name, which can include full parent path.
        offset: Offset, generally given as a tuple.
        shape: Shape, generally given as a tuple.
    
    Returns:
        Name (or path) to subimage.
    """
    roi_site = "{}x{}".format(offset, shape).replace(" ", "")
    series_fill = lib_clrbrain.series_as_str(config.series)
    name = lib_clrbrain.splice_before(base, series_fill, roi_site)
    print("subimage name: {}".format(name))
    return name

def detect_blobs_large_image(filename_base, image5d, offset, roi_size, 
                             verify=False):
    """Detect blobs within a large image through parallel processing of 
    smaller chunks.
    
    Args:
        filename_base: Base path to use file output.
        image5d: Large image to process as a Numpy array of t,z,y,x,[c]
        offset: Image offset given as coordinates in x,y,z.
        roi_size: ROI shape given in x,y,z.
        verify: True to verify detections against truth database; defaults 
            to False.
    """
    time_start = time()
    filename_image5d_proc = filename_base + config.SUFFIX_IMG_PROC
    filename_info_proc = filename_base + config.SUFFIX_INFO_PROC
    roi_offset = offset
    shape = roi_size
    if roi_size is None or offset is None:
        # uses the entire stack if no size or offset specified
        shape = image5d.shape[3:0:-1]
        roi_offset = (0, 0, 0)
    else:
        # sets up processing for partial stack
        filename_image5d_proc = make_subimage_name(
            filename_image5d_proc, offset, roi_size)
        filename_info_proc = make_subimage_name(
            filename_info_proc, offset, roi_size)
    
    # get ROI for given region, including all channels
    roi = plot_3d.prepare_roi(image5d, shape, roi_offset)
    _, channels = plot_3d.setup_channels(roi, config.channel, 3)
    
    # chunk into super-ROIs, which will each be further chunked into 
    # sub-ROIs for multi-processing
    overlap = chunking.calc_overlap()
    settings = config.process_settings # use default settings
    tol = (np.multiply(overlap, settings["prune_tol_factor"])
           .astype(int))
    max_pixels = config.sub_stack_max_pixels
    if max_pixels is None:
        # command-line set max takes precedence, but if not set, take 
        # from process settings
        max_pixels = settings["sub_stack_max_pixels"]
    print("overlap: {}, max_pixels: {}, tol: {}"
          .format(overlap, max_pixels, tol))
    super_rois, super_rois_offsets = chunking.stack_splitter(
        roi, max_pixels, overlap)
    seg_rois = np.zeros(super_rois.shape, dtype=object)
    dfs_times = []
    dfs_pruning = []
    for z in range(super_rois.shape[0]):
        for y in range(super_rois.shape[1]):
            for x in range(super_rois.shape[2]):
                coord = (z, y, x)
                roi = super_rois[coord]
                print("===============================================\n"
                      "Processing stack {} of {}"
                      .format(coord, np.add(super_rois.shape, -1)))
                merged, segs, df_times, df_pruning = detect_blobs_stack(
                    roi, overlap, tol, channels)
                del merged # TODO: check if helps reduce memory buildup
                dfs_times.append(df_times)
                if segs is not None:
                    # transpose seg coords since part of larger stack
                    off = super_rois_offsets[coord]
                    detector.shift_blob_rel_coords(segs, off)
                    detector.shift_blob_abs_coords(segs, off)
                    dfs_pruning.append(df_pruning)
                seg_rois[coord] = segs
    
    # prune segments in overlapping region between super-ROIs
    print("===============================================\n"
          "Pruning super-ROIs")
    time_pruning_start = time()
    segments_all, df_pruning = _prune_blobs_mp(
        seg_rois, overlap, tol, super_rois, super_rois_offsets, channels)
    pruning_time = time() - time_pruning_start
    print("super ROI pruning time (s)", pruning_time)
    #print("maxes:", np.amax(segments_all, axis=0))
    
    # combine pruning data frames and get weighted mean of ratios
    dfs_pruning.append(df_pruning)
    print("\nBlob pruning ratios:")
    df_pruning_all = stats.data_frames_to_csv(
        dfs_pruning, "blob_ratios.csv", show=True)
    cols = df_pruning_all.columns.tolist()
    blob_pruning_means = {}
    if "blobs" in cols:
        blobs_unpruned = df_pruning_all["blobs"]
        num_blobs_unpruned = np.sum(blobs_unpruned)
        for col in cols[1:]:
            blob_pruning_means["mean_{}".format(col)] = [
                np.sum(np.multiply(df_pruning_all[col], blobs_unpruned)) 
                / num_blobs_unpruned]
        df_pruning_means = stats.dict_to_data_frame(
            blob_pruning_means, "blob_ratios_means.csv", show=" ")
    else:
        print("no blob ratios found")
    
    '''# report any remaining duplicates
    np.set_printoptions(linewidth=500, threshold=10000000)
    print("all blobs (len {}):".format(len(segments_all)))
    sort = np.lexsort(
        (segments_all[:, 2], segments_all[:, 1], segments_all[:, 0]))
    blobs = segments_all[sort]
    print(blobs)
    print("checking for duplicates in all:")
    print(detector.remove_duplicate_blobs(blobs, slice(0, 3)))
    '''
    
    stats_detection = None
    fdbk = None
    if segments_all is not None:
        # remove the duplicated elements that were used for pruning
        detector.replace_rel_with_abs_blob_coords(segments_all)
        segments_all = detector.remove_abs_blob_coords(segments_all)
        
        # compared detected blobs with truth blobs
        if verify:
            db_path_base = None
            try:
                if config.truth_db_name and config.truth_db:
                    # use explicitly given truth DB if given, which can 
                    # containg multiple experiments for different subimages
                    print("using truth DB from {}"
                          .format(config.truth_db_name))
                    exp_name = importer.deconstruct_np_filename(
                        config.truth_db_name)[0]
                else:
                    # find truth DB based on filename and subimage
                    db_path_base = os.path.basename(
                        make_subimage_name(filename_base, offset, roi_size))
                    print("about to verify with truth db from {}"
                          .format(db_path_base))
                    sqlite.load_truth_db(db_path_base)
                    exp_name = make_subimage_name(
                        os.path.basename(config.filename), roi_offset, 
                        shape)
                print("exp name: {}".format(exp_name))
                if config.truth_db is not None:
                    # series not included in exp name since in ROI
                    exp_id = sqlite.insert_experiment(
                        config.verified_db.conn, config.verified_db.cur, 
                        exp_name, None)
                    rois = config.truth_db.get_rois(exp_name)
                    verify_tol = np.multiply(
                        overlap, settings["verify_tol_factor"]).astype(int)
                    stats_detection, fdbk = detector.verify_rois(
                        rois, segments_all, config.truth_db.blobs_truth, 
                        verify_tol, config.verified_db, exp_id, config.channel)
            except FileNotFoundError as e:
                print("Could not load truth DB from {}; "
                      "will not verify ROIs".format(db_path_base))
    
    file_time_start = time()
    if config.saveroi:
        '''
        # write the merged, denoised file (old behavior, <0.4.3)
        # TODO: write files to memmap array to release RAM?
        outfile_image5d_proc = open(filename_image5d_proc, "wb")
        np.save(outfile_image5d_proc, merged)
        outfile_image5d_proc.close()
        '''
        # write the original, raw ROI
        outfile_image5d_proc = open(filename_image5d_proc, "wb")
        np.save(outfile_image5d_proc, roi)
        outfile_image5d_proc.close()
    
    outfile_info_proc = open(filename_info_proc, "wb")
    np.savez(outfile_info_proc, segments=segments_all, 
             resolutions=detector.resolutions, 
             basename=os.path.basename(config.filename), # only save name
             offset=offset, roi_size=roi_size) # None unless explicitly set
    outfile_info_proc.close()
    file_save_time = time() - file_time_start
    
    # whole image benchmarking time
    df_times_all = stats.data_frames_to_csv(
        dfs_times, "stack_detection_times.csv")
    times = (
        [np.sum(df_times_all[StackTimes.DETECTION.value])], 
        [np.sum(df_times_all[StackTimes.PRUNING.value]) + pruning_time], 
        time() - time_start)
    times_dict = {}
    for key, val in zip(StackTimes, times):
        times_dict[key] = val
    if segments_all is None:
        print("\nNo blobs detected")
    else:
        print("\nTotal blobs found:", len(segments_all))
        detector.show_blobs_per_channel(segments_all)
    print("file save time:", file_save_time)
    print("\nTotal detection processing times (s):")
    df_times_sum = stats.dict_to_data_frame(
        times_dict, "stacks_detection_times.csv", show=" ")
    
    return stats_detection, fdbk, segments_all

def detect_blobs_stack(roi, overlap, tol, channels):
    """Process a stack, whcih can be a sub-region within an ROI.
    
    Args:
        roi: The ROI to process.
        overlap: The amount of overlap to use between chunks within the stack.
        tol: Tolerance as (z, y, x), within which a segment will be 
            considered a duplicate of a segment in the master array and
            removed.
    
    Returns:
        Tuple of the merged, processed image stack; all the blobs 
        found within the stack, given as a Numpy array in the format, 
        ``[n, [z, row, column, radius, ...]]``, including additional 
        elements as given in :meth:``StackDetect.detect_sub_roi``; a 
        Pandas data frame of durations for each component of blob 
        detections; and a Pandas data frame of blob pruning ratios.
    """
    time_start = time()
    scaling_factor = detector.calc_scaling_factor()
    segments_all = None
    merged = roi
    
    # detect blobs, using larger sub-ROI size to minimize the number 
    # of sub-ROIs and thus the number of edge overlaps
    time_segmenting_start = time()
    denoise_size = config.process_settings["denoise_size"]
    denoise_max_shape = None
    if denoise_size:
        denoise_max_shape = np.ceil(
            np.multiply(scaling_factor, denoise_size)).astype(int)
    max_pixels = np.ceil(np.multiply(
        scaling_factor, 
        config.process_settings["segment_size"])).astype(int)
    #print("max_factor: {}".format(max_factor))
    sub_rois, sub_rois_offsets = chunking.stack_splitter(
        merged, max_pixels, overlap)
    StackDetector.set_data(sub_rois, sub_rois_offsets, denoise_max_shape)
    pool = mp.Pool()
    pool_results = []
    for z in range(sub_rois.shape[0]):
        for y in range(sub_rois.shape[1]):
            for x in range(sub_rois.shape[2]):
                coord = (z, y, x)
                pool_results.append(
                    pool.apply_async(StackDetector.detect_sub_roi, 
                                     args=(coord, )))
    
    seg_rois = np.zeros(sub_rois.shape, dtype=object)
    for result in pool_results:
        coord, segments = result.get()
        lib_clrbrain.printv(
            "adding segments from sub_roi at {} of {}"
            .format(coord, np.add(sub_rois.shape, -1)))
        seg_rois[coord] = segments
    
    pool.close()
    pool.join()
    time_segmenting_end = time()
    
    # prune segments
    time_pruning_start = time()
    segments_all, df_pruning = _prune_blobs_mp(
        seg_rois, overlap, tol, sub_rois, sub_rois_offsets, channels)
    # copy shifted coordinates to final coordinates
    #print("blobs_all:\n{}".format(blobs_all[:, 0:4] == blobs_all[:, 5:9]))
    if segments_all is not None:
        detector.replace_rel_with_abs_blob_coords(segments_all)
    pruning_time = time() - time_pruning_start
    
    
    # benchmarking time
    times = (
        [time_segmenting_end - time_segmenting_start], 
        [pruning_time], 
        [time() - time_start])
    times_dict = {}
    for key, val in zip(StackTimes, times):
        times_dict[key] = val
    print("Stack processing times (s):")
    df_times = stats.dict_to_data_frame(times_dict, show=" ")
    
    return merged, segments_all, df_times, df_pruning

class StackPruner(object):
    """Prune blobs within a stack in a way that allows multiprocessing 
    without global variables.
    
    Attributes:
        blobs_to_prune: List of tuples to be passed to 
            :meth:``detector.remove_close_blobs_within_sorted_array``.
    """
    blobs_to_prune = None
    
    @classmethod
    def set_data(cls, blobs_to_prune):
        """Set the data to be shared during multiprocessing.
        
        Args:
            blobs_to_prune: List of tuples as specified for 
                :attr:``blobs_to_prune``.
        """
        cls.blobs_to_prune = blobs_to_prune
    
    @classmethod
    def prune_overlap(cls, i):
        """Prune an overlapping region.
        
        Args:
            i: Index in :attr:``blobs_to_prune``.
        
        Returns:
            The results from 
            :meth:``detector.remove_close_blobs_within_sorted_array``.
        """
        return detector.remove_close_blobs_within_sorted_array(
            *cls.blobs_to_prune[i])
        
def _prune_blobs_mp(seg_rois, overlap, tol, sub_rois, sub_rois_offsets, 
                    channels):
    """Prune close blobs within overlapping regions by checking within
    entire planes across the ROI in parallel with multiprocessing.
    
    Args:
        segs_roi: Segments from each sub-region.
        overlap: 1D array of size 3 with the number of overlapping pixels 
            for each image axis.
        tol: Tolerance as (z, y, x), within which a segment will be 
            considered a duplicate of a segment in the master array and
            removed.
        sub_rois: Sub-regions, used to check size.
        sub_rois_offset: Offsets of each sub-region.
    
    Returns:
        Tuple of all blobs as a Numpy array and a data frame of 
        pruning stats, or None for both if no blobs are in the ``seg_rois``.
    """
    # collects all blobs in master array to group all overlapping regions
    blobs_merged = chunking.merge_blobs(seg_rois)
    if blobs_merged is None:
        return None, None
    
    blobs_all = []
    blob_ratios = {}
    cols = ("blobs", "ratio_pruning", "ratio_adjacent")
    for i in channels:
        # prune blobs from each channel separately to avoid pruning based on 
        # co-localized channel signals
        blobs = detector.blobs_in_channel(blobs_merged, i)
        for axis in range(3):
            # prune planes with all the overlapping regions within a given axis,
            # skipping if this axis has no overlapping sub-regions
            num_sections = sub_rois_offsets.shape[axis]
            if num_sections <= 1:
                continue
            
            # multiprocess pruning by overlapping planes
            blobs_all_non_ol = None # all blobs from non-overlapping regions
            blobs_to_prune = []
            coord_last = tuple(np.subtract(sub_rois.shape, 1))
            for i in range(num_sections):
                # build overlapping region dimensions based on size of 
                # sub-region in the given axis
                coord = np.zeros(3, dtype=np.int)
                coord[axis] = i
                lib_clrbrain.printv("** checking blobs in ROI {}".format(coord))
                offset = sub_rois_offsets[tuple(coord)]
                size = sub_rois[tuple(coord)].shape
                lib_clrbrain.printv("offset: {}, size: {}, overlap: {}, tol: {}"
                                    .format(offset, size, overlap, tol))
                
                # overlapping region: each region but the last extends 
                # into the next region, with the overlapping volume from 
                # the end of the region, minus the overlap and a tolerance 
                # space, to the region's end plus this tolerance space; 
                # non-overlapping region: the region before the overlap, 
                # after any overlap with the prior region (n > 1) 
                # to the start of the overlap (n < last region)
                blobs_ol = None
                blobs_ol_next = None
                blobs_in_non_ol = []
                shift = overlap[axis] + tol[axis]
                offset_axis = offset[axis]
                if i < num_sections - 1:
                    bounds = [offset_axis + size[axis] - shift,
                              offset_axis + size[axis] + tol[axis]]
                    lib_clrbrain.printv(
                        "axis {}, boundaries: {}".format(axis, bounds))
                    blobs_ol = blobs[np.all([
                        blobs[:, axis] >= bounds[0], 
                        blobs[:, axis] < bounds[1]], axis=0)]
                    
                    # get blobs from immediatley adjacent region of the same 
                    # size as that of the overlapping region
                    bounds_next = [
                        offset_axis + size[axis] + tol[axis],
                        offset_axis + size[axis] + overlap[axis] + 3 * tol[axis]
                    ]
                    shape = np.add(
                        sub_rois_offsets[coord_last], 
                        sub_rois[coord_last].shape[:3])
                    lib_clrbrain.printv(
                        "axis {}, boundaries (next): {}, max bounds: {}"
                        .format(axis, bounds_next, shape[axis]))
                    if np.all(np.less(bounds_next, shape[axis])):
                        # ensure that next overlapping region is within ROI
                        blobs_ol_next = blobs[np.all([
                            blobs[:, axis] >= bounds_next[0], 
                            blobs[:, axis] < bounds_next[1]], axis=0)]
                    # non-overlapping region extends up this overlap
                    blobs_in_non_ol.append(blobs[:, axis] < bounds[0])
                else:
                    # last non-overlapping region extends to end of region
                    blobs_in_non_ol.append(
                        blobs[:, axis] < offset_axis + size[axis])
                
                # get non-overlapping area
                start = offset_axis
                if i > 0:
                    # shift past overlapping part at start of region
                    start += shift
                blobs_in_non_ol.append(blobs[:, axis] >= start)
                blobs_non_ol = blobs[np.all(blobs_in_non_ol, axis=0)]
                # collect all non-overlapping region blobs
                if blobs_all_non_ol is None:
                    blobs_all_non_ol = blobs_non_ol
                elif blobs_non_ol is not None:
                    blobs_all_non_ol = np.concatenate(
                        (blobs_all_non_ol, blobs_non_ol))
                
                blobs_to_prune.append((blobs_ol, tol, blobs_ol_next))
            
            StackPruner.set_data(blobs_to_prune)
            pool = mp.Pool()
            pool_results = []
            for i in range(len(blobs_to_prune)):
                # prune blobs from overlapping regions via multiprocessing, 
                # using a separate class to avoid pickling input blobs
                pool_results.append(pool.apply_async(
                    StackPruner.prune_overlap, args=(i, )))
            
            # collect all the pruned blob lists
            blobs_all_ol = None
            for result in pool_results:
                blobs_ol_pruned, ratios = result.get()
                if blobs_all_ol is None:
                    blobs_all_ol = blobs_ol_pruned
                elif blobs_ol_pruned is not None:
                    blobs_all_ol = np.concatenate(
                        (blobs_all_ol, blobs_ol_pruned))
                if ratios:
                    for col, val in zip(cols, ratios):
                        blob_ratios.setdefault(col, []).append(val)
            
            # recombine blobs from the non-overlapping with the pruned  
            # overlapping regions from the entire stack to re-prune along 
            # any remaining axes
            pool.close()
            pool.join()
            if blobs_all_ol is None:
                blobs = blobs_all_non_ol
            elif blobs_all_non_ol is None:
                blobs = blobs_all_ol
            else:
                blobs = np.concatenate((blobs_all_non_ol, blobs_all_ol))
        # build up list from each channel
        blobs_all.append(blobs)
    blobs_all = np.vstack(blobs_all)
    print("total blobs after pruning:", len(blobs_all))
    
    # export blob ratios as data frame
    df = pd.DataFrame(blob_ratios)
    
    return blobs_all, df
