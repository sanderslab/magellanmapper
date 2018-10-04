#!/bin/bash
# Command line parsing and setup
# Author: David Young, 2017, 2018
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
    * plane: Plane type (see :const:``config.PLANE``).
    * res: Resolution given as (x, y, z) in floating point (see
        cli.py, though order is natural here as command-line argument).
    * saveroi: Save ROI from original image to file during stack processing.
    * register: Registration type. See :attr:``config.REGISTER_TYPES`` for 
        types of registration and :mod:``register`` for how to use these 
        types.
    * labels: Load annotation JSON file. The first argument is the path 
        to the JSON file. If a 2nd arguments is given, it is taken as an int of 
        the ontology level for grouping volumes.
    * flip: Flags for flipping images horizontally for registration. 
        "0" or "false" (case-insensivite) are taken as False, and 
        "1" or "true" are taken as True. The number of flags should 
        correspond to the number of images to register, such as several for 
        groupwise registration.
    * rescale: Rescaling factor as a float value.
    * interval: Interval as an int, such as for animated GIF stack planes.
    * chunk_shape: Stack processing chunk shape given as integeres in z,y,x 
        order. This value will take precedence over the 
        ``sub_stack_max_pixels`` entry in the :class:``ProcessSettings`` 
        profile entry.
    * ec2_start: EC2 start instances parameters, used in 
        :function:``aws.start_instances``.
    * notify: Notification with up to three parameters for URL, message, and 
        attachment file path, stored respectively as 
        :attr:``config.notify_url``, :attr:``config.notify_msg``, and 
        :attr:``config.notify_attach``.

Attributes:
    roi_size: The size in pixels of the region of interest. Set with
        "size=x,y,z" argument, where x, y, and z are integers.
    offset: The bottom corner in pixels of the region of interest. Set 
        with "offset=x,y,z" argument, where x, y, and z are integers.
    PROC_TYPES: Processing modes. ``importonly`` imports an image stack and 
        exits non-interactively. ``processing`` processes and segments the 
        entire image stack and exits non-interactively. ``load`` loads already 
        processed images and segments. ``extract`` extracts a single plane 
        using the z-value from the offset and exits. ``export_rois`` 
        exports ROIs from the current database to serial 2D plots. 
        ``transpose`` transposes the Numpy image file associated with 
        ``filename`` with the ``--rescale`` option. ``animated`` generates 
        an animated GIF with the ``--interval`` and ``--rescale`` options. 
        ``export_blobs`` exports blob coordinates/radii to compressed CSV file.
    proc: The chosen processing mode; defaults to None.
    TRUTH_DB_TYPES: Truth database modes. ``view`` loads the truth 
        database corresponding to the filename and any offset/size to show 
        alongside the current database. ``verify`` creates a new database 
        to store results from ROC curve building. ``verified`` loads the  
        verified database generated from the prior mode.
    truth_db_type: The chosen truth database type; defaults to None. The first 
        argument will compared with ``TRUTH_DB_TYPES``. If a second argument 
        is given, it will be used as the path to the truth database for 
        ``view`` and ``verify``, the main and verified databases for 
        ``verified``, and the main database for ``edit``.
"""

import os
import sys
import argparse
from time import time
import multiprocessing as mp
import numpy as np

from clrbrain import chunking
from clrbrain import config
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import sqlite
from clrbrain import plot_3d
from clrbrain import detector
from clrbrain import chunking
from clrbrain import mlearn

roi_size = None # current region of interest
roi_sizes = None # list of regions of interest
offset = None # current offset
offsets = None # list of offsets

image5d = None # numpy image array
image5d_proc = None
segments_proc = None
sub_rois = None
_blobs_all = None # share blobs among multiple processes

PROC_TYPES = (
    "importonly", "processing", "processing_mp", "load", "extract", 
    "export_rois", "transpose", "animated", "export_blobs"
)
proc_type = None

TRUTH_DB_TYPES = ("view", "verify", "verified", "edit")
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
    lib_clrbrain.printv("denoising sub_roi at {} of {}, with shape {}..."
          .format(coord, np.add(sub_rois.shape, -1), sub_roi.shape))
    sub_roi = plot_3d.saturate_roi(sub_roi, channel=config.channel)
    sub_roi = plot_3d.denoise_roi(sub_roi, channel=config.channel)
    #sub_roi = plot_3d.deconvolve(sub_roi)
    if config.process_settings["thresholding"]:
        sub_roi = plot_3d.threshold(sub_roi)
    return (coord, sub_roi)

def segment_sub_roi(sub_rois_offsets, coord):
    """Segments the ROI within an array of ROIs.
    
    The array of ROIs is assumed to be cli.sub_rois.
    
    Args:
        sub_rois_offsets: Array of offsets each given as (z, y, x) 
            for each sub_roi in the larger array, used to transpose the 
            segments into absolute coordinates.
        coord: Coordinate of the sub-ROI in the order (z, y, x).
    
    Returns:
        Tuple of coord, which is the coordinate given back again to 
            identify the sub-ROI, and the denoised sub-ROI.
    """
    sub_roi = sub_rois[coord]
    lib_clrbrain.printv("segmenting sub_roi at {} of {}, with shape {}..."
          .format(coord, np.add(sub_rois.shape, -1), sub_roi.shape))
    segments = detector.detect_blobs(sub_roi, config.channel)
    offset = sub_rois_offsets[coord]
    #print("segs before (offset: {}):\n{}".format(offset, segments))
    if segments is not None:
        # shift both coordinate sets (at beginning and end of array) to 
        # absolute positioning, using the latter set to store shifted 
        # coordinates based on duplicates and the former for initial 
        # positions to check for multiple duplicates
        detector.shift_blob_rel_coords(segments, offset)
        detector.shift_blob_abs_coords(segments, offset)
        #print("segs after:\n{}".format(segments))
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

def _splice_before(base, search, splice, post_splice="_"):
    i = base.rfind(search)
    if i == -1:
        # fallback to splicing before extension
        i = base.rfind(".")
        if i == -1:
            return base
        else:
            # turn post-splice into pre-splice delimiter, assuming that the 
            # absence of search string means delimiter is not before the ext
            splice = post_splice + splice
            post_splice = ""
    return base[0:i] + splice + post_splice + base[i:]

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
    series_fill = importer.series_as_str(config.series)
    name = _splice_before(base, series_fill, roi_site)
    print("subimage name: {}".format(name))
    return name

def _load_db(path):
    if not os.path.exists(path):
        raise FileNotFoundError("{} not found for DB".format(path))
    print("loading DB from {}".format(path))
    db = sqlite.ClrDB()
    db.load_db(path, False)
    return db

def _load_truth_db(filename_base):
    path = filename_base
    if not filename_base.endswith(sqlite.DB_SUFFIX_TRUTH):
        path = os.path.basename(filename_base + sqlite.DB_SUFFIX_TRUTH)
    truth_db = _load_db(path)
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

def _prune_blobs_mp(seg_rois, overlap, tol, sub_rois, sub_rois_offsets, 
                    channels):
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
    blobs_merged = chunking.merge_blobs(seg_rois)
    if blobs_merged is None:
        return None
    
    blobs_all = []
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
            pool = mp.Pool()
            pool_results = []
            blobs_all_non_ol = None # all blobs from non-overlapping regions
            for i in range(num_sections):
                # build overlapping region dimensions based on size of 
                # sub-region in the given axis
                coord = np.zeros(3).astype(np.int)
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
                lib_clrbrain.printv(
                    "axis {}, boundaries: {}".format(axis, bounds))
                blobs_ol = blobs[np.all([
                    blobs[:, axis] >= bounds[0], 
                    blobs[:, axis] < bounds[1]], axis=0)]
                
                # non-overlapping area is the rest of the region, subtracting 
                # the tolerance unless the region is first and not overlapped
                start = offset[axis]
                if i > 0:
                    start += shift
                blobs_non_ol = blobs[np.all([
                    blobs[:, axis] >= start, 
                    blobs[:, axis] < bounds[0]], axis=0)]
                # collect all these non-overlapping region blobs
                if blobs_all_non_ol is None:
                    blobs_all_non_ol = blobs_non_ol
                elif blobs_non_ol is not None:
                    blobs_all_non_ol = np.concatenate(
                        (blobs_all_non_ol, blobs_non_ol))
                
                # prune blobs from overlapping regions via multiprocessing
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
                    blobs_all_ol = np.concatenate(
                        (blobs_all_ol, blobs_ol_pruned))
            
            # recombine blobs from the non-overlapping with the pruned  
            # overlapping regions from the entire stack
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
    return blobs_all

def _is_arg_true(arg):
    return arg.lower() == "true" or arg == "1"

def args_with_dict(args):
    """Parse arguments list with optional arguments given as dictionary-like 
    elements.
    
    Args:
        args: List of arguments, which can be single values or "=" delimited 
           values. Single values will be stored in the same order, while 
           delimited entries will be entered sequentially into a dictionary. 
           Entries can also be comma-delimited to specify lists.
    
    Returns:
        List of arguments ordered first with single-value entries in the 
        same order in which they were entered, followed by a dictionary 
        with all equals-delimited entries, also in the same order as entered. 
        Entries that contain commas will be split into comma-delimited 
        lists. All values will be converted to ints if possible.
    """
    parsed = []
    args_dict = {}
    for arg in args:
        arg_split = arg.split("=")
        for_dict = len(arg_split) > 1
        vals = arg_split[1] if for_dict else arg
        vals_split = vals.split(",")
        if len(vals_split) > 1: vals = vals_split
        vals = lib_clrbrain.get_int(vals)
        if for_dict:
            args_dict[arg_split[0]] = vals
        else:
            parsed.append(vals)
    parsed.append(args_dict)
    return parsed

def main(process_args_only=False):
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    
    Args:
        process_args_only: If True, processes command-line arguments and exits.
    """
    parser = argparse.ArgumentParser(
        description="Setup environment for Clrbrain")
    global roi_size, \
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
    parser.add_argument("--microscope", nargs="*")
    parser.add_argument("--truth_db", nargs="*")
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--plane")
    parser.add_argument("--saveroi", action="store_true")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--flip", nargs="*")
    parser.add_argument("--register")
    parser.add_argument("--reg_profile")
    parser.add_argument("--rescale")
    parser.add_argument("--interval")
    parser.add_argument("--delay")
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--border", nargs="*")
    parser.add_argument("--db")
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--chunk_shape", nargs="*")
    parser.add_argument("--ec2_start", nargs="*")
    parser.add_argument("--ec2_list", nargs="*")
    parser.add_argument("--ec2_terminate", nargs="*")
    parser.add_argument("--notify", nargs="*")
    parser.add_argument("--prefix")
    parser.add_argument("--suffix")
    args = parser.parse_args()
    
    # set image file path and convert to basis for additional paths
    if args.img is not None:
        config.filenames = args.img
        config.filename = config.filenames[0]
        print("Set filenames to {}, current filename {}"
              .format(config.filenames, config.filename))
    
    if args.channel is not None:
        config.channel = args.channel
        if config.channel == -1:
            config.channel = None
        print("Set channel to {}".format(config.channel))
    series_list = [config.series] # list of series
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
        config.series = series_list[0]
        print("Set to series_list to {}, current series {}".format(
              series_list, config.series))
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
        print("Set ROI sizes to {}, current size {}"
              .format(roi_sizes, roi_size))
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
        for i in range(len(args.microscope)):
            settings = config.process_settings if i == 0 else config.ProcessSettings()
            config.update_process_settings(settings, args.microscope[i])
            if i > 0:
                config.process_settings_list.append(settings)
                print("Added {} settings for channel {}".format(
                      config.process_settings_list[i]["settings_name"], i))
    print("Set default microscope processing settings to {}"
          .format(config.process_settings["settings_name"]))
    # registration profile settings
    if args.reg_profile is not None:
        config.update_register_settings(
            config.register_settings, args.reg_profile)
    print("Set register settings to {}"
          .format(config.register_settings["settings_name"]))
    if args.plane is not None:
        from clrbrain import plot_2d
        config.plane = args.plane
        print("Set plane to {}".format(config.plane))
    if args.saveroi:
        config.saveroi = args.saveroi
        print("Set save ROI to file to ".format(config.saveroi))
    if args.labels:
        # parse labels with optional args transferred directly to config
        labels_parsed = args_with_dict(args.labels)
        parsed_len = len(labels_parsed)
        parsed_dict = labels_parsed[-1]
        config.load_labels = (
            labels_parsed[0] if parsed_len > 1 
            else parsed_dict.get("labels_path"))
        print("Set load labels path to {}".format(config.load_labels))
        config.labels_level = (
            labels_parsed[1] if parsed_len > 2 else parsed_dict.get("level"))
        if config.labels_level is not None:
            config.labels_level = int(config.labels_level)
        print("Set labels level to {}".format(config.labels_level))
    if args.flip:
        config.flip = []
        for flip in args.flip:
            config.flip.append(_is_arg_true(flip))
        print("Set flip to {}".format(config.flip))
    if args.register:
        config.register_type = args.register
        print("Set register type to {}".format(config.register_type))
    if args.rescale:
        config.rescale = float(args.rescale)
        print("Set rescale to {}".format(config.rescale))
    if args.interval:
        config.interval = int(args.interval)
        print("Set interval to {}".format(config.interval))
    if args.delay:
        config.delay = int(args.delay)
        print("Set delay to {}".format(config.delay))
    if args.no_show:
        config.no_show = args.no_show
        print("Set no show to {}".format(config.no_show))
    if args.border:
        borders = _parse_coords(args.border)
        config.border = borders[0]
        print("Set ROI export to clip to border: {}".format(config.border))
    if args.groups:
        config.groups = args.groups
        print("Set groups to {}".format(config.groups))
    if args.chunk_shape is not None:
        # TODO: given as z,y,x for overall project order consistency; need 
        # to consider whether to shift to x,y,z for user-input consistency or 
        # to change user-input to z,y,x
        chunk_shapes = _parse_coords(args.chunk_shape)
        if len(chunk_shapes) > 0:
            config.sub_stack_max_pixels = chunk_shapes[0]
            print("Set chunk shape to {}".format(config.sub_stack_max_pixels))
    if args.ec2_start is not None:
        # start EC2 instances
        config.ec2_start = args_with_dict(args.ec2_start)
        print("Set ec2 start to {}".format(config.ec2_start))
    if args.ec2_list:
        # list EC2 instances
        config.ec2_list = args_with_dict(args.ec2_list)
        print("Set ec2 list to {}".format(config.ec2_list))
    if args.ec2_terminate:
        config.ec2_terminate = args.ec2_terminate
        print("Set ec2 terminate to {}".format(config.ec2_terminate))
    if args.notify:
        notify_len = len(args.notify)
        if notify_len > 0:
            config.notify_url = args.notify[0]
            print("Set notification URL to {}".format(config.notify_url))
        if notify_len > 1:
            config.notify_msg = args.notify[1]
            print("Set notification message to {}".format(config.notify_msg))
        if notify_len > 2:
            config.notify_attach = args.notify[2]
            print("Set notification attachment path to {}"
                  .format(config.notify_attach))
    if args.prefix:
        config.prefix = args.prefix
        print("Set path prefix to {}".format(config.prefix))
    if args.suffix:
        config.suffix = args.suffix
        print("Set path suffix to {}".format(config.suffix))
    
    # prep filename
    if not config.filename:
        # unable to parse anymore args without filename
        print("filename not specified, stopping argparsing")
        return
    ext = lib_clrbrain.get_filename_ext(config.filename)
    filename_base = importer.filename_to_base(
        config.filename, config.series, ext=ext)
    
    
    # Database prep
    
    if args.db:
        config.db_name = args.db
        print("Set database name to {}".format(config.db_name))
    # load "truth blobs" from separate database for viewing
    truth_db_type = None
    if args.truth_db is not None:
        truth_db_type = args.truth_db[0]
        print("Set truth_db type to {}".format(truth_db_type))
        if len(args.truth_db) > 1:
            config.truth_db_name = args.truth_db[1]
            print("Set truth_db name to {}".format(config.truth_db_name))
    if truth_db_type == TRUTH_DB_TYPES[0]:
        # loads truth DB as a separate database in parallel with the given 
        # editable database, with name based on filename by default unless 
        # truth DB name explicitly given
        path = config.truth_db_name if config.truth_db_name else filename_base
        try:
            _load_truth_db(path)
        except FileNotFoundError as e:
            print(e)
            print("Could not load truth DB from current image path")
    elif truth_db_type == TRUTH_DB_TYPES[1]:
        # creates a new verified DB to store all ROC results
        config.verified_db = sqlite.ClrDB()
        config.verified_db.load_db(sqlite.DB_NAME_VERIFIED, True)
        if config.truth_db_name:
            # load truth DB path to verify against if explicitly given
            try:
                _load_truth_db(config.truth_db_name)
            except FileNotFoundError as e:
                print(e)
                print("Could not load truth DB from {}"
                      .format(config.truth_db_name))
    elif truth_db_type == TRUTH_DB_TYPES[2]:
        # loads verified DB as the main DB, which includes copies of truth 
        # values with flags for whether they were detected
        path = sqlite.DB_NAME_VERIFIED
        if config.truth_db_name: path = config.truth_db_name
        try:
            config.db = _load_db(path)
            config.verified_db = config.db
        except FileNotFoundError as e:
            print(e)
            print("Could not load verified DB from {}"
                  .format(sqlite.DB_NAME_VERIFIED))
    elif truth_db_type == TRUTH_DB_TYPES[3]:
        # loads truth DB as the main database for editing rather than 
        # loading as a truth database
        config.db_name = config.truth_db_name
        if not config.db_name: 
            config.db_name = "{}{}".format(
                os.path.basename(filename_base), sqlite.DB_SUFFIX_TRUTH)
        print("Editing truth database at {}".format(config.db_name))
    
    if config.db is None:
        config.db = sqlite.ClrDB()
        config.db.load_db(None, False)
    
    
    
    # done with arg parsing
    if process_args_only:
        return
    
    
    
    # process the image stack for each series
    for series in series_list:
        filename_base = importer.filename_to_base(
            config.filename, series, ext=ext)
        if config.roc:
            # grid search with ROC curve
            stats_dict = mlearn.grid_search(
                _iterate_file_processing, filename_base, offsets, roi_sizes)
            parsed_dict = mlearn.parse_grid_stats(stats_dict)
            # plot ROC curve
            from clrbrain import plot_2d
            plot_2d.setup_style()
            plot_2d.plot_roc(parsed_dict, config.filename)
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
        stat_roi, fdbk, _ = process_file(filename_base, offsets[i], size)
        if stat_roi is not None:
            stat = np.add(stat, stat_roi)
        summaries.append(
            "Offset {}:\n{}".format(offsets[i], fdbk))
    return stat, summaries

#@profile
def process_file(filename_base, offset, roi_size):
    """Processes a single image file non-interactively.
    
    Args:
        filename_base: Base filename.
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
    filename_image5d_proc = filename_base + config.SUFFIX_IMG_PROC
    filename_info_proc = filename_base + config.SUFFIX_INFO_PROC
    filename_roi = None
    #print(filename_image5d_proc)
    
    # LOAD MAIN IMAGE
    
    if proc_type in (PROC_TYPES[3], PROC_TYPES[5], PROC_TYPES[8]):
        # load a processed image, typically a chunk of a larger image
        print("Loading processed image files")
        global image5d_proc, segments_proc
        try:
            # processed image file, which < v.0.4.3 was the saved 
            # filtered image, but >= v.0.4.3 is the ROI chunk of the orig image
            image5d_proc = np.load(filename_image5d_proc, mmap_mode="r")
            image5d_proc = importer.roi_to_image5d(image5d_proc)
            print("Loading processed/ROI image from {} with shape {}"
                  .format(filename_image5d_proc, image5d_proc.shape))
        except IOError:
            print("Ignoring processed/ROI image file from {} as unable to load"
                  .format(filename_image5d_proc))
        try:
            # processed segments and other image information
            output_info = np.load(filename_info_proc)
            segments_proc = output_info["segments"]
            print("{} segments loaded".format(len(segments_proc)))
            #print("segments range:\n{}".format(np.max(segments_proc, axis=0)))
            #print("segments:\n{}".format(segments_proc))
            detector.resolutions = output_info["resolutions"]
            roi_offset = None
            shape = None
            path = config.filename
            try:
                # find original image path and prepare to extract ROI from it 
                # based on offset/roi_size saved in processed file
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
                path, config.series, offset=roi_offset, size=shape, 
                channel=config.channel, import_if_absent=False)
            if image5d is None:
                # if unable to load original image, attempts to use ROI file
                image5d = image5d_proc
                if image5d is None:
                    raise IOError("Neither original nor ROI image file found")
        except IOError as e:
            print("Unable to load processed info file at {}, will exit"
                  .format(filename_info_proc))
            raise e
    
    if image5d is None:
        # load or import the main image stack
        print("Loading main image")
        if os.path.isdir(config.filename):
            # import directory of TIFF images
            image5d = importer.import_dir(os.path.join(config.filename, "*"))
        elif (config.filename.endswith(".nii.gz") 
              or config.filename.endswith(".mha")
              or config.filename.endswith(".mhd")):
            # load formats supported by SimpleITK, using metadata from 
            # Numpy archive
            rotate = config.flip is not None and config.flip[0]
            filename_np = config.filename # default to same basic name
            if len(config.filenames) > 1:
                # load metadata from 2nd filename argument if given
                filename_np = config.filenames[1]
            image5d = importer.read_file_sitk(
                config.filename, filename_np, config.series, rotate)
        else:
            # load or import from Clrbrain Numpy format
            load = proc_type != PROC_TYPES[0] # explicitly re/import
            image5d = importer.read_file(
                config.filename, config.series, channel=config.channel, 
                load=load)
    
    if config.load_labels is not None:
        # load labels image and set up scaling
        from clrbrain import register
        config.labels_img = register.load_registered_img(
            config.filename, reg_name=register.IMG_LABELS)
        config.labels_scaling = importer.calc_scaling(
            image5d, config.labels_img)
        config.labels_ref = register.load_labels_ref(config.load_labels)
        config.labels_ref_lookup = register.create_aba_reverse_lookup(
            config.labels_ref)
        try:
            config.borders_img = register.load_registered_img(
                config.filename, reg_name=register.IMG_BORDERS)
        except FileNotFoundError as e:
            print(e)
    
    
    # PROCESS BY TYPE
    
    if proc_type == PROC_TYPES[3]:
        # loading completed
        return None, None
        
    elif proc_type == PROC_TYPES[0]:
        # already imported so does nothing
        print("imported {}, will exit".format(config.filename))
    
    elif proc_type == PROC_TYPES[4]:
        # extract and save plane
        print("extracting plane at {} and exiting".format(offset[2]))
        name = ("{}-(series{})-z{}").format(
            os.path.basename(config.filename).replace(".czi", ""), 
            config.series, str(offset[2]).zfill(5))
        from clrbrain import stack
        stack.save_plane(image5d, offset, roi_size, name)
    
    elif proc_type == PROC_TYPES[5]:
        # export ROIs; assumes that info_proc was already loaded to 
        # give smaller region from which smaller ROIs from the truth DB 
        # will be extracted
        from clrbrain import exporter
        db = config.db if config.truth_db is None else config.truth_db
        exporter.export_rois(
            db, image5d, config.channel, filename_base, config.border)
        
    elif proc_type == PROC_TYPES[6]:
        # transpose Numpy array
        importer.transpose_npy(
            config.filename, config.series, plane=config.plane, 
            rescale=config.rescale)
        
    elif proc_type == PROC_TYPES[7]:
        # generate animated GIF
        from clrbrain import stack
        stack.animated_gif(
            config.filename, series=config.series, interval=config.interval, 
            rescale=config.rescale, delay=config.delay)
    
    elif proc_type == PROC_TYPES[8]:
        # export blobs to CSV file
        from clrbrain import exporter
        exporter.blobs_to_csv(segments_proc, filename_info_proc)
        
    elif proc_type == PROC_TYPES[1] or proc_type == PROC_TYPES[2]:
        # denoises and segments the region, saving processed image
        # and segments to file
        time_start = time()
        roi_offset = offset
        shape = roi_size
        if roi_size is None or offset is None:
            # uses the entire stack if no size or offset specified
            shape = image5d.shape[3:0:-1]
            roi_offset = (0, 0, 0)
        else:
            # sets up processing for partial stack
            filename_image5d_proc = make_subimage_name(filename_image5d_proc, offset, roi_size)
            filename_info_proc = make_subimage_name(filename_info_proc, offset, roi_size)
        
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
                    merged, segs = process_stack(roi, overlap, tol, channels)
                    del merged # TODO: check if helps reduce memory buildup
                    if segs is not None:
                        # transpose seg coords since part of larger stack
                        off = super_rois_offsets[coord]
                        detector.shift_blob_rel_coords(segs, off)
                        detector.shift_blob_abs_coords(segs, off)
                    seg_rois[coord] = segs
        
        # prune segments in overlapping region between super-ROIs
        time_pruning_start = time()
        segments_all = _prune_blobs_mp(
            seg_rois, overlap, tol, super_rois, super_rois_offsets, channels)
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
            detector.replace_rel_with_abs_blob_coords(segments_all)
            segments_all = detector.remove_abs_blob_coords(segments_all)
            
            # compared detected blobs with truth blobs
            if truth_db_type == TRUTH_DB_TYPES[1]:
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
                        _load_truth_db(db_path_base)
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
                        stats, fdbk = detector.verify_rois(
                            rois, segments_all, config.truth_db.blobs_truth, 
                            BLOB_COORD_SLICE, tol, config.verified_db, exp_id,
                            config.channel)
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
        #print("merged shape: {}".format(merged.shape))
        np.savez(outfile_info_proc, segments=segments_all, 
                 resolutions=detector.resolutions, 
                 basename=os.path.basename(config.filename), # only save name
                 offset=offset, roi_size=roi_size) # None unless explicitly set
        outfile_info_proc.close()
        
        segs_len = 0 if segments_all is None else len(segments_all)
        print("super ROI pruning time (s): {}".format(pruning_time))
        print("total segments found: {}".format(segs_len))
        print("file save time: {}".format(time() - file_time_start))
        print("total file processing time (s): {}".format(time() - time_start))
        return stats, fdbk, segments_all
    return None, None, None
    
def process_stack(roi, overlap, tol, channels):
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
    denoise_size = config.process_settings["denoise_size"]
    # no overlap for denoising
    overlap_denoise = np.zeros(3).astype(np.int)
    if denoise_size:
        max_pixels = np.ceil(
            np.multiply(scaling_factor, denoise_size)).astype(int)
        sub_rois, _ = chunking.stack_splitter(roi, max_pixels, overlap_denoise)
    segments_all = None
    merged = roi
    
    # process ROI
    time_denoising_start = time()
    if proc_type == PROC_TYPES[2]:
        # Multiprocessing
        
        # denoise all sub-ROIs and re-merge
        if denoise_size:
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
                lib_clrbrain.printv(
                    "replacing sub_roi at {} of {}"
                    .format(coord, np.add(sub_rois.shape, -1)))
                sub_rois[coord] = sub_roi
            
            pool.close()
            pool.join()
            
            # re-merge into one large ROI (the image stack) in preparation for 
            # segmenting with differently sized chunks
            merged_shape = chunking.get_split_stack_total_shape(
                sub_rois, overlap_denoise)
            merged = np.zeros(tuple(merged_shape), dtype=sub_rois[0, 0, 0].dtype)
            chunking.merge_split_stack2(sub_rois, overlap_denoise, 0, merged)
        else:
            lib_clrbrain.printv(
                "no denoise size specified, will not preprocess")
        
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
            lib_clrbrain.printv(
                "adding segments from sub_roi at {} of {}"
                .format(coord, np.add(sub_rois.shape, -1)))
            seg_rois[coord] = segments
        
        pool.close()
        pool.join()
        time_segmenting_end = time()
        
        # prune segments
        time_pruning_start = time()
        segments_all = _prune_blobs_mp(
            seg_rois, overlap, tol, sub_rois, sub_rois_offsets, channels)
        # copy shifted coordinates to final coordinates
        #print("blobs_all:\n{}".format(blobs_all[:, 0:4] == blobs_all[:, 5:9]))
        if segments_all is not None:
            detector.replace_rel_with_abs_blob_coords(segments_all)
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
            seg_rois, BLOB_COORD_SLICE, overlap, tol, sub_rois, 
            sub_rois_offsets)
    
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
    
