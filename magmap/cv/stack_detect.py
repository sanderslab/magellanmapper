#!/bin/bash
# Detect blobs within a chunked stack through multiprocessing
# Author: David Young, 2019, 2020
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

from magmap.cv import chunking
from magmap.settings import config
from magmap.cv import detector
from magmap.io import importer
from magmap.io import libmag
from magmap.plot import plot_3d
from magmap.io import sqlite
from magmap.io import df_io

# Numpy archive for blobs versions:
# 0: initial version
# 1: added resolutions, basene, offset, roi_size fields
# 2: added archive version number
BLOBS_NP_VER = 2


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
        exclude_border: Sequence of border pixels in x,y,z to exclude;
            defaults to None.
    """
    sub_rois = None
    sub_rois_offsets = None
    denoise_max_shape = None
    exclude_border = None
    
    @classmethod
    def set_data(cls, sub_rois, sub_rois_offsets, denoise_max_shape, 
                 exclude_border):
        """Set the class attributes to be shared during multiprocessing."""
        cls.sub_rois = sub_rois
        cls.sub_rois_offsets = sub_rois_offsets
        cls.denoise_max_shape = denoise_max_shape
        cls.exclude_border = exclude_border
    
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
        offset = cls.sub_rois_offsets[coord]
        last_coord = np.subtract(cls.sub_rois.shape, 1)
        print("detecting blobs in sub-ROI at {} of {}, offset {}, shape {}..."
              .format(coord, last_coord, tuple(offset.astype(int)), 
                      sub_roi.shape))
        
        if cls.denoise_max_shape is not None:
            # further split sub-ROI for preprocessing locally
            denoise_rois, _ = chunking.stack_splitter(
                sub_roi, cls.denoise_max_shape)
            for z in range(denoise_rois.shape[0]):
                for y in range(denoise_rois.shape[1]):
                    for x in range(denoise_rois.shape[2]):
                        denoise_coord = (z, y, x)
                        libmag.printv_format(
                            "preprocessing sub-sub-ROI {} of {} (shape {}"
                            " within sub-ROI shape {})", 
                            (denoise_coord, np.subtract(denoise_rois.shape, 1), 
                             denoise_rois[denoise_coord].shape, 
                             sub_roi.shape))
                        denoise_roi = plot_3d.saturate_roi(
                            denoise_rois[denoise_coord], channel=config.channel)
                        denoise_roi = plot_3d.denoise_roi(
                            denoise_roi, channel=config.channel)
                        denoise_rois[denoise_coord] = denoise_roi
            
            # re-merge into one large ROI (the image stack) in preparation for 
            # segmenting with differently sized chunks, typically larger 
            # to minimize the number of sub-ROIs and edge overlaps
            merged_shape = chunking.get_split_stack_total_shape(denoise_rois)
            merged = np.zeros(
                tuple(merged_shape), dtype=denoise_rois[0, 0, 0].dtype)
            chunking.merge_split_stack2(denoise_rois, None, 0, merged)
            sub_roi = merged
        
        if cls.exclude_border is None:
            exclude = None
        else:
            exclude = np.array([cls.exclude_border, cls.exclude_border])
            exclude[0, np.equal(coord, 0)] = 0
            exclude[1, np.equal(coord, last_coord)] = 0
        segments = detector.detect_blobs(sub_roi, config.channel, exclude)
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
    series_fill = libmag.series_as_str(config.series)
    name = libmag.splice_before(base, series_fill, roi_site)
    print("subimage name: {}".format(name))
    return name


def detect_blobs_large_image(filename_base, image5d, offset, roi_size, 
                             verify=False, save_dfs=True, full_roi=False):
    """Detect blobs within a large image through parallel processing of 
    smaller chunks.
    
    Args:
        filename_base: Base path to use file output.
        image5d: Large image to process as a Numpy array of t,z,y,x,[c]
        offset: Image offset given as coordinates in x,y,z.
        roi_size: ROI shape given in x,y,z.
        verify: True to verify detections against truth database; defaults 
            to False.
        save_dfs: True to save data frames to file; defaults to True.
        full_roi (bool): True to treat ``image5d`` as the full ROI; defaults
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
    if full_roi:
        # treat the full image as the ROI
        roi = image5d[0]
    else:
        roi = plot_3d.prepare_roi(image5d, shape, roi_offset)
    _, channels = plot_3d.setup_channels(roi, config.channel, 3)
    
    # prep chunking ROI into sub-ROIs with size based on segment_size, scaling
    # by physical units to make more independent of resolution
    time_detection_start = time()
    settings = config.process_settings  # use default settings
    scaling_factor = detector.calc_scaling_factor()
    print("microsope scaling factor based on resolutions: {}"
          .format(scaling_factor))
    denoise_size = config.process_settings["denoise_size"]
    denoise_max_shape = None
    if denoise_size:
        # further subdivide each sub-ROI for local preprocessing
        denoise_max_shape = np.ceil(
            np.multiply(scaling_factor, denoise_size)).astype(int)

    # overlap sub-ROIs to minimize edge effects
    overlap_base = chunking.calc_overlap()
    tol = np.multiply(overlap_base, settings["prune_tol_factor"]).astype(int)
    overlap_padding = np.copy(tol)
    overlap = np.copy(overlap_base)
    exclude_border = config.process_settings["exclude_border"]
    if exclude_border is not None:
        # exclude border to avoid blob detector edge effects, where blobs
        # often collect at the faces of the sub-ROI;
        # ensure that overlap is greater than twice the border exclusion per
        # axis so that no plane will be excluded from both overlapping sub-ROIs
        exclude_border_thresh = np.multiply(2, exclude_border)
        overlap_less = np.less(overlap, exclude_border_thresh)
        overlap[overlap_less] = exclude_border_thresh[overlap_less]
        excluded = np.greater(exclude_border, 0)
        overlap[excluded] += 1  # additional padding
        overlap_padding[excluded] = 0  # no need to prune past excluded border
    print("sub-ROI overlap: {}, pruning tolerance: {}, padding beyond "
          "overlap for pruning: {}, exclude borders: {}"
          .format(overlap, tol, overlap_padding, exclude_border))
    max_pixels = np.ceil(np.multiply(
        scaling_factor, 
        config.process_settings["segment_size"])).astype(int)
    print("preprocessing max shape: {}, detection max pixels: {}"
          .format(denoise_max_shape, max_pixels))
    sub_rois, sub_rois_offsets = chunking.stack_splitter(
        roi, max_pixels, overlap)
    # TODO: option to distribute groups of sub-ROIs to different servers 
    # for blob detection
    seg_rois = detect_blobs_sub_rois(
        sub_rois, sub_rois_offsets, denoise_max_shape, exclude_border)
    detection_time = time() - time_detection_start
    print("blob detection time (s):", detection_time)
    
    # prune blobs in overlapping portions of sub-ROIs
    time_pruning_start = time()
    segments_all, df_pruning = _prune_blobs_mp(
        seg_rois, overlap, tol, sub_rois, sub_rois_offsets, channels, 
        overlap_padding)
    pruning_time = time() - time_pruning_start
    print("blob pruning time (s):", pruning_time)
    #print("maxes:", np.amax(segments_all, axis=0))
    
    # get weighted mean of ratios
    if df_pruning is not None:
        print("\nBlob pruning ratios:")
        path_pruning = "blob_ratios.csv" if save_dfs else None
        df_pruning_all = df_io.data_frames_to_csv(
            df_pruning, path_pruning, show=" ")
        cols = df_pruning_all.columns.tolist()
        blob_pruning_means = {}
        if "blobs" in cols:
            blobs_unpruned = df_pruning_all["blobs"]
            num_blobs_unpruned = np.sum(blobs_unpruned)
            for col in cols[1:]:
                blob_pruning_means["mean_{}".format(col)] = [
                    np.sum(np.multiply(df_pruning_all[col], blobs_unpruned)) 
                    / num_blobs_unpruned]
            path_pruning_means = "blob_ratios_means.csv" if save_dfs else None
            df_pruning_means = df_io.dict_to_data_frame(
                blob_pruning_means, path_pruning_means, show=" ")
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
                    # contain multiple experiments for different subimages
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
                        overlap_base, settings["verify_tol_factor"])
                    stats_detection, fdbk = detector.verify_rois(
                        rois, segments_all, config.truth_db.blobs_truth, 
                        verify_tol, config.verified_db, exp_id, config.channel)
            except FileNotFoundError as e:
                print("Could not load truth DB from {}; "
                      "will not verify ROIs".format(db_path_base))
    
    file_time_start = time()
    if config.saveroi:
        # write the original, raw ROI
        outfile_image5d_proc = open(filename_image5d_proc, "wb")
        np.save(outfile_image5d_proc, roi)
        outfile_image5d_proc.close()
    
    outfile_info_proc = open(filename_info_proc, "wb")
    np.savez(outfile_info_proc, ver=BLOBS_NP_VER, segments=segments_all,
             resolutions=config.resolutions,
             basename=os.path.basename(config.filename),  # only save name
             offset=offset, roi_size=roi_size) # None unless explicitly set
    outfile_info_proc.close()
    file_save_time = time() - file_time_start
    
    # whole image benchmarking time
    times = (
        [detection_time], 
        [pruning_time], 
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
    path_times = "stack_detection_times.csv" if save_dfs else None
    df_times_sum = df_io.dict_to_data_frame(
        times_dict, path_times, show=" ")
    
    return stats_detection, fdbk, segments_all

def detect_blobs_sub_rois(sub_rois, sub_rois_offsets, denoise_max_shape, 
                          exclude_border):
    """Process blobs in an ROI chunked into multiple sub-ROIs via 
    multiprocessing.
    
    Args:
        sub_rois: Numpy object array containing chunked sub-ROIs within a 
            stack.
        sub_rois_offsets: Numpy object array of the same shape as 
            ``sub_rois`` with offsets in z,y,x corresponding to each 
            sub-ROI. Offets are used to transpose blobs into 
            absolute coordinates.
        denoise_max_shape: Maximum shape of each unit within each sub-ROI 
            for denoising.
        exclude_border: Sequence of border pixels in x,y,z to exclude;
            defaults to None.
    
    Returns:
        Numpy object array of blobs corresponding to ``sub_rois``, with 
        each set of blobs given as a Numpy array in the format, 
        ``[n, [z, row, column, radius, ...]]``, including additional 
        elements as given in :meth:``StackDetect.detect_sub_roi``.
    """
    # detect nuclei in each sub-ROI, passing an index to access each 
    # sub-ROI to minimize pickling
    StackDetector.set_data(
        sub_rois, sub_rois_offsets, denoise_max_shape, exclude_border)
    pool = mp.Pool()
    pool_results = []
    for z in range(sub_rois.shape[0]):
        for y in range(sub_rois.shape[1]):
            for x in range(sub_rois.shape[2]):
                coord = (z, y, x)
                pool_results.append(
                    pool.apply_async(StackDetector.detect_sub_roi, 
                                     args=(coord, )))
    
    # retrieve blobs and assign to object array corresponding to sub_rois
    seg_rois = np.zeros(sub_rois.shape, dtype=object)
    for result in pool_results:
        coord, segments = result.get()
        num_blobs = 0 if segments is None else len(segments)
        print("adding {} blobs from sub_roi at {} of {}"
              .format(num_blobs, coord, np.add(sub_rois.shape, -1)))
        seg_rois[coord] = segments
    
    pool.close()
    pool.join()
    return seg_rois


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
        pruner = cls.blobs_to_prune[i]
        blobs, axis, tol, blobs_next = pruner
        axis_col = 10 + axis
        #print("orig blobs in axis {}, i {}\n{}".format(axis, i, blobs))
        if blobs is None: return None, None
        num_blobs_orig = len(blobs)
        print("num_blobs_orig in axis {}, {}: {}"
              .format(axis, i, num_blobs_orig))
        blobs_master = blobs[blobs[:, axis_col] == i]
        blobs = blobs[blobs[:, axis_col] == i + 1]
        #print("blobs_master in axis {}, i {}\n{}".format(axis, i, blobs_master))
        #print("blobs to check in axis {}, next i ({})\n{}".format(axis, i + 1, blobs))
        pruned, blobs_master = detector.remove_close_blobs(
            blobs, blobs_master, tol)
        blobs_after_pruning = np.concatenate((blobs_master, pruned))
        #blobs_after_pruning = detector.remove_close_blobs_within_sorted_array(blobs, tol)
        pruning_ratios = None
        if blobs_next is not None:
            pruning_ratios = detector.meas_pruning_ratio(
                num_blobs_orig, len(blobs_after_pruning), len(blobs_next))
        return blobs_after_pruning, pruning_ratios


def _prune_blobs_mp(seg_rois, overlap, tol, sub_rois, sub_rois_offsets,
                    channels, overlap_padding=None):
    """Prune close blobs within overlapping regions by checking within
    entire planes across the ROI in parallel with multiprocessing.
    
    Args:
        seg_rois: Segments from each sub-region.
        overlap: 1D array of size 3 with the number of overlapping pixels 
            for each image axis.
        tol: Tolerance as (z, y, x), within which a segment will be 
            considered a duplicate of a segment in the master array and
            removed.
        sub_rois: Sub-regions, used to check size.
        sub_rois_offsets: Offsets of each sub-region.
        overlap_padding: Sequence of z,y,x for additional padding beyond
            ``overlap``. Defaults to None to use ``tol`` as padding.
    
    Returns:
        Tuple of all blobs as a Numpy array and a data frame of 
        pruning stats, or None for both if no blobs are in the ``seg_rois``.
    """
    # collects all blobs in master array to group all overlapping regions
    blobs_merged = chunking.merge_blobs(seg_rois)
    if blobs_merged is None:
        return None, None
    print("total blobs before pruning:", len(blobs_merged))
    
    print("pruning with overlap: {}, overlap tol: {}, pruning tol: {}"
          .format(overlap, overlap_padding, tol))
    blobs_all = []
    blob_ratios = {}
    cols = ("blobs", "ratio_pruning", "ratio_adjacent")
    if overlap_padding is None: overlap_padding = tol
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
                print("** setting up blob pruning in axis {}, section {} of {}"
                      .format(axis, i, num_sections - 1))
                offset = sub_rois_offsets[tuple(coord)]
                size = sub_rois[tuple(coord)].shape
                libmag.printv_format(
                    "offset: {}, size: {}", (offset, size))
                
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
                shift = overlap[axis] + overlap_padding[axis]
                offset_axis = offset[axis]
                if i < num_sections - 1:
                    bounds = [offset_axis + size[axis] - shift,
                              offset_axis + size[axis] + overlap_padding[axis]]
                    libmag.printv(
                        "axis {}, boundaries: {}".format(axis, bounds))
                    blobs_ol = blobs[np.all([
                        blobs[:, axis] >= bounds[0], 
                        blobs[:, axis] < bounds[1]], axis=0)]
                    
                    # get blobs from immediatley adjacent region of the same 
                    # size as that of the overlapping region; keep same 
                    # starting point with or without overlap_tol
                    start = offset_axis + size[axis] + tol[axis]
                    bounds_next = [
                        start, start + overlap[axis] + 2 * overlap_padding[axis]]
                    shape = np.add(
                        sub_rois_offsets[coord_last], 
                        sub_rois[coord_last].shape[:3])
                    libmag.printv(
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
                
                #blobs_to_prune.append((blobs_ol, tol, blobs_ol_next))
                blobs_to_prune.append((blobs_ol, axis, tol, blobs_ol_next))
            
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
