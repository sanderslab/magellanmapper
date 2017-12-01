#!/bin/bash
# Command line parsing and setup
# Author: David Young, 2017
"""Command line parser and and environment setup for Clrbrain.

This module can be run either as a script to work in headless mode or 
loaded and initialized by calling main(). 

Note on dimensions order: User-defined dimension 
variables are generally given in (x, y, z) order as per normal
convention, but otherwise dimensions are generally in (z, y, x) for
consistency with microscopy order and ease of processing stacks by z.

Examples:
    Launch in headless mode with the given file at a particular size and 
    offset:
        
        $ python -m clrbrain.cli --img /path/to/file.czi --offset 30,50,205 \
            --size 150,150,10

Command-line arguments in addition to those from attributes listed below:
    * load_labels: Path to labels reference file, which also serves as a flag 
        to load the labels image as well 
        (see :attr:`config.load_labels`).
    * mlab_3d: 3D visualization mode (see plot_3d.py).
    * padding_2d: Padding around the ROI given as (x, y, z) from which to 
        include segments and and show further 2D planes.
    * plane: Plane type (see plot_2d.py PLANE).
    * res: Resolution given as (x, y, z) in floating point (see
        cli.py, though order is natural here as command-line argument).
    * saveroi: Save ROI from original image to file during stack processing.

Attributes:
    filename: The filename of the source images. A corresponding file with
        the subset as a 5 digit number (eg 00003) with .npz appended to 
        the end will be checked first based on this filename. Set with
        "img=path/to/file" argument.
    series: The series for multi-stack files, using 0-based indexing. Set
        with "series=n" argument.
    channel: The channel to view. Set with "channel=n" argument.
    roi_size: The size in pixels of the region of interest. Set with
        "size=x,y,z" argument, where x, y, and z are integers.
    offset: The bottom corner in pixels of the region of interest. Set 
        with "offset=x,y,z" argument, where x, y, and z are integers.
    PROC_TYPES: Processing modes. ``importonly`` imports an image stack and 
        exits non-interactively. ``processing`` processes and segments the 
        entire image stack and exits non-interactively. ``load`` loads already 
        processed images and segments.
    proc: The chosen processing mode
"""

import os
import sys
import argparse
from time import time
import multiprocessing as mp
import numpy as np
#from memory_profiler import profile

from clrbrain import config
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import sqlite
from clrbrain import plot_3d
from clrbrain import detector
from clrbrain import chunking
from clrbrain import mlearn

filename = None # current image file path
filenames = None # list of multiple image paths
series = 0 # series for multi-stack files
series_list = [series] # list of series
channel = 0 # channel of interest
roi_size = None # current region of interest
roi_sizes = None # list of regions of interest
offset = None # current offset
offsets = None # list of offsets

image5d = None # numpy image array
image5d_proc = None
segments_proc = None
sub_rois = None
_blobs_all = None # share blobs among multiple processes

PROC_TYPES = ("importonly", "processing", "processing_mp", "load", "extract")
proc_type = None

TRUTH_DB_TYPES = ("view", "verify", "verified")
truth_db_type = None

BLOB_COORD_SLICE = slice(0, 3)

def denoise_sub_roi(coord):
    """Denoises the ROI within an array of ROIs.
    
    The array of ROIs is assumed to be cli.sub_rois.
    
    Args:
        coord: Coordinate of the sub-ROI in the order (z, y, x).
    
    Returns:
        Tuple of coord, which is the coordinate given back again to 
            identify the sub-ROI, and the denoised sub-ROI.
    """
    sub_roi = sub_rois[coord]
    print("denoising sub_roi at {} of {}, with shape {}..."
          .format(coord, np.add(sub_rois.shape, -1), sub_roi.shape))
    sub_roi = plot_3d.saturate_roi(sub_roi)
    sub_roi = plot_3d.denoise_roi(sub_roi)
    #sub_roi = plot_3d.deconvolve(sub_roi)
    if config.process_settings["thresholding"]:
        sub_roi = plot_3d.threshold(sub_roi)
    return (coord, sub_roi)

def segment_sub_roi(sub_rois_offsets, coord):
    """Segments the ROI within an array of ROIs.
    
    The array of ROIs is assumed to be cli.sub_rois.
    
    Args:
        sub_rois_offsets: Array of offsets for each sub_roi in
            the larger array, used to give transpose the segments
            into absolute coordinates.
        coord: Coordinate of the sub-ROI in the order (z, y, x).
    
    Returns:
        Tuple of coord, which is the coordinate given back again to 
            identify the sub-ROI, and the denoised sub-ROI.
    """
    sub_roi = sub_rois[coord]
    print("segmenting sub_roi at {} of {}, with shape {}..."
          .format(coord, np.add(sub_rois.shape, -1), sub_roi.shape))
    segments = detector.segment_blob(sub_roi)
    offset = sub_rois_offsets[coord]
    if segments is not None:
        # duplicate positions, appending to end of each blob, for further
        # adjustments such as shifting the blob based on close duplicates
        segments = np.concatenate((segments, segments[:, 0:4]), axis=1)
        # transpose segments
        segments = np.add(segments, (offset[0], offset[1], offset[2], 0, 0, 0,
                                     offset[0], offset[1], offset[2], 0))
    return (coord, segments)

def collect_segments(segments_all, segments, region, tol):
    """Adds an array of segments into a master array, removing any close
    segments before adding them.
    
    Args:
        segments_all: Master array of segments against which the array to
            add will be compared for close segments.
        segments: Array of segments to add.
        region: The region of each segment array to compare for closeness,
            given as a slice.
        tol: Tolerance as (z, y, x), within which a segment will be 
            considered a duplicate of a segment in the master array and
            removed.
    
    Returns:
        The master array.
    """
    # join segments
    if segments_all is None:
        # TODO: don't really need since should already be checking for 
        # closeness in the blob detection algorithm
        segments_all = chunking.remove_close_blobs_within_array(segments,
                                                                region, tol)
    elif segments is not None:
        segments = chunking.remove_close_blobs(segments, segments_all, 
                                               region, tol)
        segments_all = np.concatenate((segments_all, segments))
    return segments_all

def _splice_before(base, search, splice):
    i = base.rfind(search)
    if i == -1:
        return base
    return base[0:i] + splice + base[i:]

def _load_db(filename_base, suffix):
    path = os.path.basename(filename_base + suffix)
    if not os.path.exists(path):
        raise FileNotFoundError("{} not found for DB".format(path))
    print("Set to load DB from {}".format(path))
    db = sqlite.ClrDB()
    db.load_db(path, False)
    return db

def _load_truth_db(filename_base):
    truth_db = _load_db(filename_base, "_truth.db")
    truth_db.load_truth_blobs()
    config.truth_db = truth_db
    return truth_db

def _parse_coords(arg):
    coords = list(arg) # copy list to avoid altering the arg itself
    n = 0
    for coord in coords:
        coord_split = coord.split(",")
        if len(coord_split) >= 3:
            coord = tuple(int(i) for i in coord_split)
        else:
            print("Coordinates ({}) should be given as 3 values (x, y, z)"
              .format(coord))
        coords[n] = coord
        n += 1
    return coords

def _check_np_none(val):
    """Checks if a value is either NoneType or a Numpy None object such as
    that returned from a Numpy archive that saved an undefined variable.
    
    Args:
        val: Value to check.
    
    Returns:
        The value if not a type of None, or a NoneType.
    """
    return None if val is None or np.all(np.equal(val, None)) else val

def _prune_blobs(seg_rois, region, overlap, tol, sub_rois, sub_rois_offsets):
    """Prune close blobs within overlapping regions.
    
    Args:
        segs_roi: Segments from each sub-region.
        region: The region of each segment array to compare for closeness,
            given as a slice.
        tol: Tolerance as (z, y, x), within which a segment will be 
            considered a duplicate of a segment in the master array and
            removed.
        sub_rois: Sub-regions, used to check size.
        sub_rois_offset: Offsets of each sub-region.
    
    Returns:
        segments_all: All segments as a Numpy array.
        duration: Time to prune, in seconds.
    """
    time_pruning_start = time()
    segments_all = chunking.prune_overlapping_blobs2(
        seg_rois, region, overlap, tol, sub_rois, sub_rois_offsets)
    if segments_all is not None:
        print("total segments found: {}".format(segments_all.shape[0]))
    time_pruning_end = time()
    duration = time_pruning_end - time_pruning_start
    return segments_all, duration

def _prune_blobs_mp(seg_rois, overlap, tol, sub_rois, sub_rois_offsets):
    """Prune close blobs within overlapping regions by checking within
    entire planes across the ROI in parallel with multiprocessing.
    
    Args:
        segs_roi: Segments from each sub-region.
        tol: Tolerance as (z, y, x), within which a segment will be 
            considered a duplicate of a segment in the master array and
            removed.
        sub_rois: Sub-regions, used to check size.
        sub_rois_offset: Offsets of each sub-region.
    
    Returns:
        All segments as a Numpy array, or None if no segments.
    """
    # collects all blobs in master array to group all overlapping regions
    _blobs_all = chunking.merge_blobs(seg_rois)
    if _blobs_all is None:
        return None
    
    for axis in range(3):
        # prune planes with all the overlapping regions within a given axis,
        # skipping if this axis has no overlapping sub-regions
        num_sections = sub_rois_offsets.shape[axis]
        if num_sections <= 1:
            continue
        
        # multiprocess pruning by overlapping planes
        pool = mp.Pool()
        pool_results = []
        blobs_all_non_ol = None # all blobs from non-overlapping regions
        for i in range(num_sections):
            # build overlapping region dimensions based on size of sub-region
            # in the given axis
            coord = np.zeros(3)
            coord[axis] = i
            lib_clrbrain.printv("** checking blobs in ROI {}".format(coord))
            offset = sub_rois_offsets[tuple(coord)]
            size = sub_rois[tuple(coord)].shape
            lib_clrbrain.printv("offset: {}, size: {}, overlap: {}, tol: {}"
                                .format(offset, size, overlap, tol))
            # each region extends into the next region, so the overlap is
            # the end of the region minus its overlap and a tolerance space,
            # extending back out to the end plus the tolerance
            shift = overlap[axis] + tol[axis]
            bounds = [offset[axis] + size[axis] - shift,
                      offset[axis] + size[axis] + tol[axis]]
            lib_clrbrain.printv("axis {}, boundaries: {}".format(axis, bounds))
            blobs_ol = _blobs_all[np.all([
                _blobs_all[:, axis] >= bounds[0], 
                _blobs_all[:, axis] < bounds[1]], axis=0)]
            
            # non-overlapping area is the rest of the region, subtracting the
            # tolerance unless the region is first and not overlapped
            start = offset[axis]
            if i > 0:
                start += shift
            blobs_non_ol = _blobs_all[np.all([
                _blobs_all[:, axis] >= start, 
                _blobs_all[:, axis] < bounds[0]], axis=0)]
            # collect all these non-overlapping region blobs
            if blobs_all_non_ol is None:
                blobs_all_non_ol = blobs_non_ol
            elif blobs_non_ol is not None:
                blobs_all_non_ol = np.concatenate(
                    (blobs_all_non_ol, blobs_non_ol))
            
            # prune blobs from overlapping regions vis multiprocessing
            pool_results.append(pool.apply_async(
                detector.remove_close_blobs_within_sorted_array, 
                args=(blobs_ol, BLOB_COORD_SLICE, tol)))
        
        # collect all the pruned blob lists
        blobs_all_ol = None
        for result in pool_results:
            blobs_ol_pruned = result.get()
            if blobs_all_ol is None:
                blobs_all_ol = blobs_ol_pruned
            elif blobs_ol_pruned is not None:
                blobs_all_ol = np.concatenate((blobs_all_ol, blobs_ol_pruned))
        
        # re-combine blobs from the non-overlapping with the pruned overlapping 
        # regions
        pool.close()
        pool.join()
        if blobs_all_ol is None:
            _blobs_all = blobs_all_non_ol
        elif blobs_all_non_ol is None:
            _blobs_all = blobs_all_ol
        else:
            _blobs_all = np.concatenate((blobs_all_non_ol, blobs_all_ol))
    return _blobs_all

def main(process_args_only=False):
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    
    Args:
        process_args_only: If True, processes command-line arguments and exits.
    """
    parser = argparse.ArgumentParser(description="Setup environment for Clrbrain")
    global filename, filenames, series, series_list, channel, roi_size, \
            rois_sizes, offset, offsets, proc_type, mlab_3d, truth_db_type
    parser.add_argument("--img", nargs="*")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--series")
    parser.add_argument("--savefig")
    parser.add_argument("--padding_2d")
    #parser.add_argument("--verify", action="store_true")
    parser.add_argument("--offset", nargs="*")
    parser.add_argument("--size", nargs="*")
    parser.add_argument("--proc")
    parser.add_argument("--mlab_3d")
    parser.add_argument("--res")
    parser.add_argument("--mag")
    parser.add_argument("--zoom")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--microscope")
    parser.add_argument("--truth_db")
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--plane")
    parser.add_argument("--saveroi", action="store_true")
    parser.add_argument("--labels")
    parser.add_argument("--flip_horiz", action="store_true")
    args = parser.parse_args()
    
    # set image file path and convert to basis for additional paths
    if args.img is not None:
        filenames = args.img
        filename = filenames[0]
        print("Set filenames to {}, current filename {}"
              .format(filenames, filename))
    
    if args.channel is not None:
        channel = args.channel
        print("Set channel to {}".format(channel))
    if args.series is not None:
        series_split = args.series.split(",")
        series_list = []
        for ser in series_split:
            ser_split = ser.split("-")
            if len(ser_split) > 1:
                ser_range = np.arange(int(ser_split[0]), int(ser_split[1]) + 1)
                series_list.extend(ser_range.tolist())
            else:
                series_list.append(int(ser_split[0]))
        series = series_list[0]
        print("Set to series_list to {}, current series {}".format(
              series_list, series))
    if args.savefig is not None:
        from clrbrain import plot_2d
        plot_2d.savefig = args.savefig
        print("Set savefig extension to {}".format(plot_2d.savefig))
    '''
    if args.verify:
        from clrbrain import plot_2d
        plot_2d.verify = args.verify
        print("Set verify to {}".format(plot_2d.verify))
    '''
    if args.verbose:
        config.verbose = args.verbose
        print("Set verbose to {}".format(config.verbose))
    if args.roc:
        config.roc = args.roc
        print("Set ROC to {}".format(config.roc))
    if args.offset is not None:
        offsets = _parse_coords(args.offset)
        offset = offsets[0]
        print("Set offsets to {}, current offset {}".format(offsets, offset))
    if args.size is not None:
        roi_sizes = _parse_coords(args.size)
        roi_size = roi_sizes[0]
        print("Set ROI sizes to {}, current size {}".format(roi_sizes, roi_size))
    if args.padding_2d is not None:
        padding_split = args.padding_2d.split(",")
        if len(padding_split) >= 3:
            from clrbrain import plot_2d
            plot_2d.padding = tuple(int(i) for i in padding_split)
            print("Set plot_2d.padding to {}".format(plot_2d.padding))
        else:
            print("padding_2d ({}) should be given as 3 values (x, y, z)"
                  .format(args.padding_2d))
    if args.proc is not None:
        if args.proc in PROC_TYPES:
            proc_type = args.proc
            print("processing type set to {}".format(proc_type))
        else:
            print("Did not recognize processing type: {}"
                  .format(args.proc))
    if args.mlab_3d is not None:
        if args.mlab_3d in plot_3d.MLAB_3D_TYPES:
            plot_3d.mlab_3d = args.mlab_3d
            print("3D rendering set to {}".format(plot_3d.mlab_3d))
        else:
            print("Did not recognize 3D rendering type: {}"
                  .format(args.mlab_3d))
    if args.res is not None:
        res_split = args.res.split(",")
        if len(res_split) >= 3:
            detector.resolutions = [tuple(float(i) for i in res_split)[::-1]]
            print("Set resolutions to {}".format(detector.resolutions))
        else:
            print("Resolution ({}) should be given as 3 values (x, y, z)"
                  .format(args.res))
    if args.mag:
        detector.magnification = args.mag
        print("Set magnification to {}".format(detector.magnification))
    if args.zoom:
        detector.zoom = args.zoom
        print("Set zoom to {}".format(detector.zoom))
    # microscope settings default to lightsheet 5x but can be updated
    if args.microscope is not None:
        config.update_process_settings(config.process_settings, args.microscope)
    print("Set microscope processing settings to {}"
          .format(config.process_settings["microscope_type"]))
    if args.plane is not None:
        from clrbrain import plot_2d
        plot_2d.plane = args.plane
        print("Set plane to {}".format(plot_2d.plane))
    if args.saveroi:
        config.saveroi = args.saveroi
        print("Set save ROI to file to ".format(config.saveroi))
    if args.labels:
        config.load_labels = args.labels
        print("Set load labels path to {}".format(config.load_labels))
    if args.flip_horiz:
        config.flip_horiz = args.flip_horiz
        print("Set flip horizontal to {}".format(config.flip_horiz))
    
    # load "truth blobs" from separate database for viewing
    filename_base = importer.filename_to_base(filename, series)
    if args.truth_db is not None:
        truth_db_type = args.truth_db
        print("Set truth_db type to {}".format(truth_db_type))
    if args.truth_db == TRUTH_DB_TYPES[0]:
        # loads truth DB
        try:
            _load_truth_db(filename_base)
        except FileNotFoundError as e:
            print(e)
            print("Could not load truth DB from current image path")
    elif args.truth_db == TRUTH_DB_TYPES[1]:
        # creates a new verified DB to store all ROC results
        config.verified_db = sqlite.ClrDB()
        config.verified_db.load_db(sqlite.DB_NAME_VERIFIED, True)
    elif args.truth_db == TRUTH_DB_TYPES[2]:
        # loads verified DB as the main DB, which includes copies of truth 
        # values with flags for whether they were detected
        try:
            config.db = _load_db(sqlite.DB_NAME_VERIFIED, "")
            config.verified_db = config.db
        except FileNotFoundError as e:
            print(e)
            print("Could not load verified DB from {}"
                  .format(sqlite.DB_NAME_VERIFIED))
    if config.db is None:
        config.db = sqlite.ClrDB()
        config.db.load_db(None, False)
    
    if process_args_only:
        return
    
    # process the image stack for each series
    for series in series_list:
        filename_base = importer.filename_to_base(filename, series)
        if config.roc:
            # grid search with ROC curve
            stats_dict = mlearn.grid_search(
                _iterate_file_processing, filename_base, offsets, roi_sizes)
            parsed_dict = mlearn.parse_grid_stats(stats_dict)
            # plot ROC curve
            from clrbrain import plot_2d
            plot_2d.plot_roc(parsed_dict, filename)
        else:
            # processes file with default settings
            process_file(filename_base, offset, roi_size)
    
    # unless loading images for GUI, exit directly since otherwise application 
    #hangs if launched from module with GUI
    if proc_type != None and proc_type != PROC_TYPES[3]:
        os._exit(os.EX_OK)

def _iterate_file_processing(filename_base, offsets, roi_sizes):
    """Processes files iteratively based on offsets.
    
    Args:
        filename_base: Base filename.
        offsets: 2D array of multiple offsets.
        roi_sizes: 2D array of multiple ROI sizes corresponding to offsets.
    
    Returns:
        stats: Summed stats.
        summaries: Concatenated summaries.
    """
    stat = np.zeros(3)
    roi_sizes_len = len(roi_sizes)
    summaries = []
    for i in range(len(offsets)):
        size = (roi_sizes[i] if roi_sizes_len > 1 
                else roi_sizes[0])
        stat_roi, fdbk = process_file(
            filename_base, offsets[i], size)
        if stat_roi is not None:
            stat = np.add(stat, stat_roi)
        summaries.append(
            "Offset {}:\n{}".format(offsets[i], fdbk))
    return stat, summaries

#@profile
def process_file(filename_base, offset, roi_size):
    """Processes a single image file non-interactively.
    
    Args:
        filename: Base filename.
        offset: Offset as (x, y, z) to start processing.
        roi_size: Size of region to process, given as (x, y, z).
    
    Returns:
        stats: Stats from processing, or None if no stats.
        fdbk: Text feedback from the processing, or None if no feedback.
    """
    # print longer Numpy arrays to assist debugging
    np.set_printoptions(linewidth=200, threshold=10000)
    
    # prepares the filenames
    global image5d
    filename_image5d_proc = filename_base + "_image5d_proc.npz"
    filename_info_proc = filename_base + "_info_proc.npz"
    filename_roi = None
    #print(filename_image5d_proc)
    
    # LOAD MAIN IMAGE
    
    if proc_type == PROC_TYPES[3]:
        # loads from processed files
        global image5d_proc, segments_proc
        try:
            # processed image file, which < v.0.4.3 was the saved 
            # filtered image, but >= v.0.4.3 is the ROI chunk of the orig image
            image5d_proc = np.load(filename_image5d_proc, mmap_mode="r")
        except IOError:
            print("Unable to load processed image file from {}, will ignore"
                  .format(filename_image5d_proc))
        try:
            # processed segments and other image information
            output_info = np.load(filename_info_proc)
            '''
            # converts old monolithic format to new format with separate files
            # to allow loading file with only image file as memory-backed array;
            # switch commented area from here to above to convert formats
            print("converting proc file to new format...")
            filename_proc = filename + str(series).zfill(5) + "_proc.npz" # old
            output = np.load(filename_proc)
            outfile_image5d_proc = open(filename_image5d_proc, "wb")
            outfile_info_proc = open(filename_info_proc, "wb")
            np.save(outfile_image5d_proc, output["roi"])
            np.savez(outfile_info_proc, segments=output["segments"], 
                     resolutions=output["resolutions"])
            outfile_image5d_proc.close()
            outfile_info_proc.close()
            return
            '''
            segments_proc = output_info["segments"]
            print("{} segments loaded".format(len(segments_proc)))
            detector.resolutions = output_info["resolutions"]
            roi_offset = None
            shape = None
            path = filename
            try:
                basename = output_info["basename"]
                roi_offset = _check_np_none(output_info["offset"])
                shape = _check_np_none(output_info["roi_size"])
                print("loaded processed offset: {}, roi_size: {}"
                      .format(roi_offset, shape))
                # raw image file assumed to be in same dir as processed file
                path = os.path.join(os.path.dirname(filename_base), 
                                    str(basename))
            except KeyError as e:
                print(e)
                print("No information on portion of stack to load")
            image5d = importer.read_file(
                path, series, offset=roi_offset, size=shape, channel=channel,
                import_if_absent=False)
            if image5d is None:
                # if unable to load original image, attempts to use ROI file
                image5d = image5d_proc
                if image5d is None:
                    raise IOError("Neither original nor ROI image file found")
        except IOError as e:
            print("Unable to load processed info file at {}, will exit"
                  .format(filename_info_proc))
            raise e
    
    # attempts to load the main image stack
    if image5d is None:
        if os.path.isdir(filename):
            image5d = importer.import_dir(os.path.join(filename, "*"))
        else:
            image5d = importer.read_file(filename, series)
    
    if config.load_labels is not None:
        # load labels image and set up scaling
        from clrbrain import register
        config.labels_img = register.load_labels(filename)
        config.labels_scaling = register.reg_scaling(image5d, config.labels_img)
        config.labels_ref = register.load_labels_ref(config.load_labels)
        config.labels_ref_lookup = register.create_aba_reverse_lookup(
            config.labels_ref)
    
    
    # PROCESS BY TYPE
    
    if proc_type == PROC_TYPES[3]:
        # loading completed
        return None, None
        
    elif proc_type == PROC_TYPES[0]:
        # already imported so does nothing
        print("imported {}, will exit".format(filename))
    
    elif proc_type == PROC_TYPES[4]:
        # extracts plane
        print("extracting plane at {} and exiting".format(offset[2]))
        name = ("{}-(series{})-z{}").format(
            os.path.basename(filename).replace(".czi", ""), 
            series, str(offset[2]).zfill(5))
        from clrbrain import plot_2d
        plot_2d.extract_plane(
            image5d, offset[2], plot_2d.plane, channel, plot_2d.savefig, name)
    
    elif proc_type == PROC_TYPES[1] or proc_type == PROC_TYPES[2]:
        # denoises and segments the region, saving processed image
        # and segments to file
        time_start = time()
        if roi_size is None or offset is None:
            # uses the entire stack if no size or offset specified
            shape = image5d.shape[3:0:-1]
            roi_offset = (0, 0, 0)
        else:
            # sets up processing for partial stack
            shape = roi_size
            roi_offset = offset
            splice = "{}x{}".format(roi_offset, shape).replace(" ", "")
            print("using {}".format(splice))
            series_fill = str(series).zfill(5)
            filename_roi = filename + splice
            filename_image5d_proc = _splice_before(filename_image5d_proc, 
                                                   series_fill, splice)
            filename_info_proc = _splice_before(filename_info_proc, 
                                                series_fill, splice)
            
        roi = plot_3d.prepare_roi(image5d, channel, shape, roi_offset)
        
        # chunk into super-ROIs, which will each be further chunked into 
        # sub-ROIs for multi-processing
        overlap = chunking.calc_overlap()
        tol = (np.multiply(overlap, config.process_settings["prune_tol_factor"])
               .astype(int))
        max_pixels = (roi.shape[0], config.sub_stack_max_pixels, 
                      config.sub_stack_max_pixels)
        print("overlap: {}, max_pixels: {}".format(overlap, max_pixels))
        super_rois, super_rois_offsets = chunking.stack_splitter(
            roi, max_pixels, overlap)
        seg_rois = np.zeros(super_rois.shape, dtype=object)
        for z in range(super_rois.shape[0]):
            for y in range(super_rois.shape[1]):
                for x in range(super_rois.shape[2]):
                    coord = (z, y, x)
                    roi = super_rois[coord]
                    print("===============================================\n"
                          "Processing stack {} of {}"
                          .format(coord, np.add(super_rois.shape, -1)))
                    merged, segs = process_stack(roi, overlap, tol)
                    del merged # TODO: check if helps reduce memory buildup
                    if segs is not None:
                        # transpose seg coords since part of larger stack
                        off = super_rois_offsets[coord]
                        segs = np.add(segs, (*off, 0, 0, 0, *off, 0))
                    seg_rois[coord] = segs
        
        # prune segments in overlapping region between super-ROIs
        time_pruning_start = time()
        segments_all = _prune_blobs_mp(
            seg_rois, overlap, tol, super_rois, super_rois_offsets)
        pruning_time = time() - time_pruning_start
        '''# report any remaining duplicates
        np.set_printoptions(linewidth=500, threshold=10000000)
        print("all blobs (len {}):".format(len(segments_all)))
        sort = np.lexsort((segments_all[:, 2], segments_all[:, 1], segments_all[:, 0]))
        blobs = segments_all[sort]
        print(blobs)
        print("checking for duplicates in all:")
        print(detector.remove_duplicate_blobs(blobs, BLOB_COORD_SLICE))
        '''
        
        stats = None
        fdbk = None
        if segments_all is not None:
            # remove the duplicated elements that were used for pruning
            segments_all[:, 0:4] = segments_all[:, 6:]
            segments_all = segments_all[:, 0:6]
            
            # compared detected blobs with truth blobs
            if truth_db_type == TRUTH_DB_TYPES[1]:
                db_path_base = _splice_before(filename_base, series_fill, splice)
                try:
                    _load_truth_db(db_path_base)
                    if config.truth_db is not None:
                        '''
                        verified_db = sqlite.ClrDB()
                        verified_db.load_db(
                            os.path.basename(db_path_base) + "_verified.db", True)
                        '''
                        exp_name = os.path.basename(filename_roi)
                        exp_id = sqlite.insert_experiment(
                            config.verified_db.conn, config.verified_db.cur, 
                            exp_name, None)
                        rois = config.truth_db.get_rois(exp_name)
                        stats, fdbk = detector.verify_rois(
                            rois, segments_all, config.truth_db.blobs_truth, 
                            BLOB_COORD_SLICE, tol, config.verified_db, exp_id)
                except FileNotFoundError as e:
                    print("Could not load truth DB from {}; will not verify ROIs"
                          .format(db_path_base))
        
        # save denoised stack, segments, and scaling info to file
        file_time_start = time()
        if config.saveroi:
            '''
            # write the merged file
            # TODO: write files to memmap array to release RAM?
            outfile_image5d_proc = open(filename_image5d_proc, "wb")
            np.save(outfile_image5d_proc, merged)
            outfile_image5d_proc.close()
            '''
            # write the ROI
            outfile_image5d_proc = open(filename_image5d_proc, "wb")
            np.save(outfile_image5d_proc, roi)
            outfile_image5d_proc.close()
        
        outfile_info_proc = open(filename_info_proc, "wb")
        #print("merged shape: {}".format(merged.shape))
        np.savez(outfile_info_proc, segments=segments_all, 
                 resolutions=detector.resolutions, 
                 basename=os.path.basename(filename), # only save filename
                 offset=offset, roi_size=roi_size) # None unless explicitly set
        outfile_info_proc.close()
        
        segs_len = 0 if segments_all is None else len(segments_all)
        print("super ROI pruning time (s): {}".format(pruning_time))
        print("total segments found: {}".format(segs_len))
        print("file save time: {}".format(time() - file_time_start))
        print("total file processing time (s): {}".format(time() - time_start))
        return stats, fdbk
    return None, None
    
def process_stack(roi, overlap, tol):
    """Processes a stack, whcih can be a sub-region within an ROI.
    
    Args:
        roi: The ROI to process.
        overlap: The amount of overlap to use between chunks within the stack.
        tol: Tolerance as (z, y, x), within which a segment will be 
            considered a duplicate of a segment in the master array and
            removed.
    
    Returns:
        merged: The merged, processed image stack.
        segments_all: All the segments found within the stack, given as a
            [n, [z, row, column, radius, ...]], including additional elements 
            as given in :meth:`segment_sub_roi`.
    """
    time_start = time()
    # prepare ROI for processing;
    # need to make module-level to allow shared memory of this large array
    global sub_rois
    scaling_factor = detector.calc_scaling_factor()
    max_pixels = np.ceil(np.multiply(
        scaling_factor, config.process_settings["denoise_size"])).astype(int)
    # no overlap for denoising
    sub_rois, _ = chunking.stack_splitter(roi, max_pixels, np.zeros(3))
    segments_all = None
    merged = None
    
    # process ROI
    time_denoising_start = time()
    if proc_type == PROC_TYPES[2]:
        # Multiprocessing
        
        # denoise all sub-ROIs and re-merge
        pool = mp.Pool()
        pool_results = []
        # asynchronously denoise since denoising is independent of adjacent
        # sub-ROIs
        for z in range(sub_rois.shape[0]):
            for y in range(sub_rois.shape[1]):
                for x in range(sub_rois.shape[2]):
                    coord = (z, y, x)
                    pool_results.append(pool.apply_async(denoise_sub_roi, 
                                     args=(coord, )))
        for result in pool_results:
            coord, sub_roi = result.get()
            print("replacing sub_roi at {} of {}"
                  .format(coord, np.add(sub_rois.shape, -1)))
            sub_rois[coord] = sub_roi
        
        pool.close()
        pool.join()
        # re-merge into one large ROI (the image stack) in preparation for 
        # segmenting with differently sized chunks
        merged = chunking.merge_split_stack(sub_rois, np.zeros(3))
        time_denoising_end = time()
        
        # segment objects through blob detection, using larger sub-ROI size
        # to minimize the number of sub-ROIs and thus the number of edge 
        # overlaps to account for
        time_segmenting_start = time()
        max_pixels = np.ceil(np.multiply(
            scaling_factor, 
            config.process_settings["segment_size"])).astype(int)
        #print("max_factor: {}".format(max_factor))
        sub_rois, sub_rois_offsets = chunking.stack_splitter(
            merged, max_pixels, overlap)
        pool = mp.Pool()
        pool_results = []
        for z in range(sub_rois.shape[0]):
            for y in range(sub_rois.shape[1]):
                for x in range(sub_rois.shape[2]):
                    coord = (z, y, x)
                    pool_results.append(pool.apply_async(segment_sub_roi, 
                                     args=(sub_rois_offsets, coord)))
        
        seg_rois = np.zeros(sub_rois.shape, dtype=object)
        for result in pool_results:
            coord, segments = result.get()
            print("adding segments from sub_roi at {} of {}"
                  .format(coord, np.add(sub_rois.shape, -1)))
            seg_rois[coord] = segments
        
        pool.close()
        pool.join()
        time_segmenting_end = time()
        
        # prune segments
        time_pruning_start = time()
        segments_all = _prune_blobs_mp(
            seg_rois, overlap, tol, sub_rois, sub_rois_offsets)
        # copy shifted coordinates to final coordinates
        #print("blobs_all:\n{}".format(blobs_all[:, 0:4] == blobs_all[:, 5:9]))
        if segments_all is not None:
            segments_all[:, 0:4] = segments_all[:, 6:]
        pruning_time = time() - time_pruning_start
        
    else:
        # Non-multiprocessing
        
        for z in range(sub_rois.shape[0]):
            for y in range(sub_rois.shape[1]):
                for x in range(sub_rois.shape[2]):
                    coord = (z, y, x)
                    coord, sub_roi = denoise_sub_roi(coord)
                    sub_rois[coord] = sub_roi
        merged = chunking.merge_split_stack(sub_rois, overlap)
        time_denoising_end = time()
        
        time_segmenting_start = time()
        max_factor = config.process_settings["segment_size"]
        sub_rois, overlap, sub_rois_offsets = chunking.stack_splitter(
            merged, max_factor)
        seg_rois = np.zeros(sub_rois.shape, dtype=object)
        for z in range(sub_rois.shape[0]):
            for y in range(sub_rois.shape[1]):
                for x in range(sub_rois.shape[2]):
                    coord = (z, y, x)
                    coord, segments = segment_sub_roi(sub_rois_offsets, coord)
                    seg_rois[coord] = segments
        time_segmenting_end = time()
        
        # older pruning method than multiprocessing version
        segments_all, pruning_time = _prune_blobs(
            seg_rois, BLOB_COORD_SLICE, overlap, tol, sub_rois, sub_rois_offsets)
    
    # benchmarking time
    print("total denoising time (s): {}"
          .format(time_denoising_end - time_denoising_start))
    print("total segmenting time (s): {}"
          .format(time_segmenting_end - time_segmenting_start))
    print("total pruning time (s): {}".format(pruning_time))
    print("total stack processing time (s): {}".format(time() - time_start))
    
    return merged, segments_all
    
if __name__ == "__main__":
    print("Starting clrbrain command-line interface...")
    main()
    
