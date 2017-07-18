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
    * resolution: Resolution given as (x, y, z) in floating point (see
        cli.py, though order is natural here as command-line argument).
    * padding_2d: Padding around the ROI given as (x, y, z) from which to 
        include segments and and show further 2D planes.
    * mlab_3d: 3D visualization mode (see plot_3d.py).

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
    PROC_TYPES: Processing modes.
        * "importonly": Imports an image stack and exists non-
          interactively.
        * "processing": Processes and segments the entire image
          stack and exits non-interactively.
        * "load": Loads already processed images and segments.
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
from clrbrain import sqlite
from clrbrain import plot_3d
from clrbrain import detector
from clrbrain import chunking

filename = None # current image file path
filenames = None # list of multiple image paths
series = 0 # series for multi-stack files
channel = 0 # channel of interest
roi_size = None # current region of interest
roi_sizes = None # list of regions of interest
offset = None # current offset
offsets = None # list of offsets

image5d = None # numpy image array
image5d_proc = None
segments_proc = None
sub_rois = None

PROC_TYPES = ("importonly", "processing", "processing_mp", "load", "extract")
proc_type = None

TRUTH_DB_TYPES = ("view", "verified")
truth_db_type = None

BLOB_COORD_SLICE = slice(0, 3)

def denoise_sub_roi(coord):
    """Denoises the ROI within an array of ROIs.
    
    The array of ROIs is assumed to be cli.sub_rois.
    
    Params:
        coord: Coordinate of the sub-ROI in the order (z, y, x).
    
    Returns:
        Tuple of coord, which is the coordinate given back again to 
            identify the sub-ROI, and the denoised sub-ROI.
    """
    sub_roi = sub_rois[coord]
    print("denoising sub_roi at {} of {}, with shape {}..."
          .format(coord, np.add(sub_rois.shape, -1), sub_roi.shape))
    sub_roi = plot_3d.denoise(sub_roi)
    #sub_roi = plot_3d.deconvolve(sub_roi)
    if config.process_settings["thresholding"]:
        sub_roi = plot_3d.threshold(sub_roi)
    return (coord, sub_roi)

def segment_sub_roi(sub_rois_offsets, coord):
    """Segments the ROI within an array of ROIs.
    
    The array of ROIs is assumed to be cli.sub_rois.
    
    Params:
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
    
    Params:
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
    
    Params:
        val: Value to check.
    
    Returns:
        The value if not a type of None, or a NoneType.
    """
    return None if val is None or np.all(np.equal(val, None)) else val

def _prune_blobs(seg_rois, region, overlap, tol, sub_rois, sub_rois_offsets):
    # prune close blobs within overlapping regions
    time_pruning_start = time()
    segments_all = chunking.prune_overlapping_blobs2(
        seg_rois, region, overlap, tol, sub_rois, sub_rois_offsets)
    if segments_all is not None:
        print("total segments found: {}".format(segments_all.shape[0]))
    time_pruning_end = time()
    duration = time_pruning_end - time_pruning_start
    return segments_all, duration

def main(process_args_only=False):
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Setup environment for Clrbrain")
    global filename, series, channel, roi_size, offset, proc_type, mlab_3d, truth_db_type
    parser.add_argument("--img", nargs="*")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--series", type=int)
    parser.add_argument("--savefig")
    parser.add_argument("--padding_2d")
    #parser.add_argument("--verify", action="store_true")
    parser.add_argument("--offset", nargs="*")
    parser.add_argument("--size", nargs="*")
    parser.add_argument("--proc")
    parser.add_argument("--mlab_3d")
    parser.add_argument("--res")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--microscope")
    parser.add_argument("--truth_db")
    parser.add_argument("--roc", action="store_true")
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
        series = args.series
        print("Set to series {}".format(series))
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
    # microscope settings default to lightsheet 5x but can be updated
    if args.microscope is not None:
        config.update_process_settings(config.process_settings, args.microscope)
    print("Set microscope processing settings to {}"
          .format(config.process_settings["microscope_type"]))
    
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
        # loads verified DB, which includes copies of truth values with 
        # flags for whether they were detected
        try:
            config.db = _load_db(filename_base, "_verified.db")
        except FileNotFoundError as e:
            print(e)
            print("Could not load verified DB from {}".format(filename_base))
    if config.db is None:
        config.db = sqlite.ClrDB()
        config.db.load_db(None, False)
    
    if process_args_only:
        return
    
    # process the image stack
    if config.roc:
        settings = config.process_settings
        stats_dict = {}
        file_summaries = []
        for key, value in config.roc_dict.items():
            # group of settings, where key is the name of the group, and 
            # value is another dictionary with the group's settings
            for key2, value2 in value.items():
                if np.isscalar(value2):
                    # set scalar values rather than iterating and processing
                    settings[key2] = value2
                    print("changed {} to {}".format(key2, value2))
                else:
                    # process each value in parameter array
                    stats = []
                    for n in value2:
                        print("Processing with settings {}, {}, {}"
                              .format(key, key2, n))
                        settings[key2] = n
                        stat = np.zeros(3)
                        roi_sizes_len = len(roi_sizes)
                        for i in range(len(offsets)):
                            size = (roi_sizes[i] if roi_sizes_len > 1 
                                    else roi_sizes[0])
                            stat_roi, fdbk = process_file(
                                filename_base, offsets[i], size)
                            stat = np.add(stat, stat_roi)
                            file_summaries.append(
                                "Offset {}:\n{}".format(offsets[i], fdbk))
                        stats.append(stat)
                    stats_dict[key + "-" + key2] = (stats, value2)
        # summary of each file collected together
        for summary in file_summaries:
            print(summary)
        # plot ROC curve
        from clrbrain import plot_2d
        plot_2d.plot_roc(stats_dict, filename)
    else:
        process_file(filename_base, offset, roi_size)
    
    # unless loading images for GUI, exit directly since otherwise application 
    #hangs if launched from module with GUI
    if proc_type != None and proc_type != PROC_TYPES[3]:
        os._exit(os.EX_OK)

#@profile
def process_file(filename_base, offset, roi_size):
    # print longer Numpy arrays to assist debugging
    np.set_printoptions(linewidth=200, threshold=10000)
    
    # prepares the filenames
    global image5d
    filename_image5d_proc = filename_base + "_image5d_proc.npz"
    filename_info_proc = filename_base + "_info_proc.npz"
    filename_roi = None
    #print(filename_image5d_proc)
    
    if proc_type == PROC_TYPES[3]:
        # loads from processed files
        global image5d_proc, segments_proc
        '''# deprecated processed image loading
        try:
            # processed image file, using mem-mapped accessed for the 
            # image file to minimize memory requirement, only loading on-the-fly
            image5d_proc = np.load(filename_image5d_proc, mmap_mode="r")
        except IOError:
            print("Unable to load processed image file from {}, will ignore"
                  .format(filename_image5d_proc))
        '''
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
                print("loaded processed offset: {}, roi_size: {}".format(roi_offset, shape))
                # raw image file assumed to be in same dir as processed file
                path = os.path.join(os.path.dirname(filename_base), 
                                    str(basename))
            except KeyError as e:
                print(e)
                print("No information on portion of stack to load")
            image5d = importer.read_file(
                path, series, offset=roi_offset, size=shape, channel=channel)
            return
        except IOError as e:
            print("Unable to load processed info file at {}, will exit"
                  .format(filename_info_proc))
            raise e
    
    # attempts to load the main image stack
    image5d = importer.read_file(filename, series)
    
    if proc_type == PROC_TYPES[0]:
        # already imported so does nothing
        print("imported {}, will exit".format(filename))
    
    elif proc_type == PROC_TYPES[4]:
        # extracts plane
        print("extracting plane at {} and exiting".format(offset[2]))
        name = ("{}-(series{})-z{}").format(
            os.path.basename(filename).replace(".czi", ""), 
            series, str(offset[2]).zfill(5))
        plot_2d.extract_plane(image5d, channel, offset, name)
    
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
                    merged, segs = process_stack(roi, overlap, tol)
                    del merged # TODO: check if helps reduce memory buildup
                    if segs is not None:
                        # transpose seg coords since part of larger stack
                        off = super_rois_offsets[coord]
                        segs = np.add(segs, (*off, 0, 0, 0, *off, 0))
                    seg_rois[coord] = segs
        segments_all, pruning_time = _prune_blobs(
            seg_rois, BLOB_COORD_SLICE, overlap, tol, super_rois, 
            super_rois_offsets)
        #merged, segments_all = process_stack(roi, overlap, tol)
        
        stats = None
        fdbk = None
        if segments_all is not None:
            # remove the duplicated elements that were used for pruning
            segments_all = segments_all[:, 0:6]
            
            # compared detected blobs with truth blobs
            if truth_db_type == TRUTH_DB_TYPES[1]:
                db_path_base = _splice_before(filename_base, series_fill, splice)
                try:
                    _load_truth_db(db_path_base)
                except FileNotFoundError as e:
                    print("Could not load truth DB from {}; will not verify ROIs"
                          .format(db_path_base))
                if config.truth_db is not None:
                    verified_db = sqlite.ClrDB()
                    verified_db.load_db(
                        os.path.basename(db_path_base) + "_verified.db", True)
                    exp_name = os.path.basename(filename_roi)
                    exp_id = sqlite.insert_experiment(
                        verified_db.conn, verified_db.cur, exp_name, None)
                    rois = config.truth_db.get_rois(exp_name)
                    stats, fdbk = detector.verify_rois(
                        rois, segments_all, config.truth_db.blobs_truth, 
                        BLOB_COORD_SLICE, tol, verified_db, exp_id)
        
        # save denoised stack, segments, and scaling info to file
        time_start = time()
        '''
        # TODO: write files to memmap array to release RAM?
        outfile_image5d_proc = open(filename_image5d_proc, "wb")
        np.save(outfile_image5d_proc, merged)
        outfile_image5d_proc.close()
        '''
        outfile_info_proc = open(filename_info_proc, "wb")
        #print("merged shape: {}".format(merged.shape))
        np.savez(outfile_info_proc, segments=segments_all, 
                 resolutions=detector.resolutions, 
                 basename=os.path.basename(filename), # only save filename
                 offset=offset, roi_size=roi_size) # None unless explicitly set
        outfile_info_proc.close()
        
        segs_len = 0 if segments_all is None else len(segments_all)
        print("total segments found: {}".format(segs_len))
        print("file save time: {}".format(time() - time_start))
        print("total file processing time (s): {}".format(time() - time_start))
        return stats, fdbk
    return None, None
    
def process_stack(roi, overlap, tol):
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
    
    segments_all, pruning_time = _prune_blobs(
        seg_rois, BLOB_COORD_SLICE, overlap, tol, sub_rois, sub_rois_offsets)
    
    # benchmarking time
    print("total denoising time (s): {}".format(time_denoising_end - time_denoising_start))
    print("total segmenting time (s): {}".format(time_segmenting_end - time_segmenting_start))
    print("total pruning time (s): {}".format(pruning_time))
    print("total stack processing time (s): {}".format(time() - time_start))
    
    return merged, segments_all
    
if __name__ == "__main__":
    print("Starting clrbrain command-line interface...")
    main()
    
