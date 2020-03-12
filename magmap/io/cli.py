#!/bin/bash
# Command line parsing and setup
# Author: David Young, 2017, 2019
"""Command line parser and and environment setup for MagellanMapper.

This module can be run either as a script to work in headless mode or 
loaded and initialized by calling main(). 

Note on dimensions order: User-defined dimension 
variables are generally given in (x, y, z) order as per normal
convention, but otherwise dimensions are generally in (z, y, x) for
consistency with microscopy order and ease of processing stacks by z.

Examples:
    Launch in headless mode with the given file at a particular size and 
    offset:
        
        $ python -m magmap.cli --img /path/to/file.czi --offset 30,50,205 \
            --size 150,150,10

Command-line arguments in addition to those from attributes listed below:
    * load_labels: Path to labels reference file, which also serves as a flag 
        to load the labels image as well 
        (see :attr:`config.load_labels`).
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
    * slice: ``stop`` or ``start,stop[,step]`` values to create a slice
        object, such as for animated GIF stack planes.
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
    * proc: The chosen processing mode; defaults to None.
    * truth_db: Specify truth database parameters. First arg specifies the mode.
        Second arg (opt) specifies a path to the truth database for 
       ``view`` and ``verify``, the main and verified databases for 
       ``verified``, and the main database for ``edit``.
    * roi_size: The size in pixels of the region of interest. Set with
        "size=x,y,z" argument, where x, y, and z are integers.
    * offset: The bottom corner in pixels of the region of interest. Set 
        with "offset=x,y,z" argument, where x, y, and z are integers.

"""

import argparse
import os
import sys

import numpy as np

from magmap.gui import roi_editor
from magmap.settings import config
from magmap.io import importer
from magmap.io import libmag
from magmap.io import np_io
from magmap.io import sqlite
from magmap.stats import mlearn
from magmap.settings import profiles
from magmap.cv import stack_detect
from magmap.atlas import transformer


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
        vals = libmag.get_int(vals)
        if for_dict:
            args_dict[arg_split[0]] = vals
        else:
            parsed.append(vals)
    parsed.append(args_dict)
    return parsed


def args_to_dict(args, keys_enum, args_dict={}):
    """Parse arguments list with positional and keyword-based arguments 
    into an enum-keyed dictionary.
    
    Args:
        args: List of arguments with positional values followed by 
            "=" delimited values. Positional values will be entered 
            in the existing order of ``keys_enum`` based on member values, 
            while keyword-based values will be entered if an enum 
            member corresponding to the keyword exists.
            Entries can also be comma-delimited to specify lists.
        keys_enum: Enum to use as keys for dictionary. Values are 
            assumed to range from 1 to number of members as output 
            by the default Enum functional API.
        args_dict: Dictionary to be filled or updated with keys from 
            ``keys_enum``; defaults to empty dict.
    
    Returns:
        Dictionary filled with arguments. Values that contain commas 
        will be split into comma-delimited lists. All values will be 
        converted to ints if possible.
    """
    by_position = True
    num_enums = len(keys_enum)
    for i, arg in enumerate(args):
        arg_split = arg.split("=")
        len_arg_split = len(arg_split)
        # assume by position until any keyword given
        by_position = by_position and len_arg_split < 2
        key = None
        vals = arg
        if by_position:
            # positions are based on enum vals, assumed to range from 
            # 1 to num of members
            n = i + 1
            if n > num_enums:
                print("no further parameters in {} to assign \"{}\" by "
                      "position, skipping".format(keys_enum, arg))
                continue
            key = keys_enum(n)
        elif len_arg_split < 2:
            print("parameter {} does not contain a keyword, skipping"
                  .format(arg))
        else:
            # assign based on keyword if its equivalent enum exists
            vals = arg_split[1]
            key_str = arg_split[0].upper()
            try:
                key = keys_enum[key_str]
            except KeyError:
                print("unable to find {} in {}".format(key_str, keys_enum))
                continue
        if key:
            vals_split = vals.split(",")
            if len(vals_split) > 1:
                # use split value if comma-delimited
                vals = vals_split
            # cast to numeric types if possible and assign to found enum
            args_dict[key] = libmag.get_int(vals)
    return args_dict


def main(process_args_only=False):
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    
    Args:
        process_args_only: If True, processes command-line arguments and exits.
    """
    parser = argparse.ArgumentParser(
        description="Setup environment for MagellanMapper")
    parser.add_argument("--img", nargs="*")
    parser.add_argument("--meta", nargs="*")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--series")
    parser.add_argument("--savefig")
    parser.add_argument("--padding_2d")
    parser.add_argument("--offset", nargs="*")
    parser.add_argument("--size", nargs="*")
    parser.add_argument("--proc")
    parser.add_argument("--res")
    parser.add_argument("--mag")
    parser.add_argument("--zoom")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--microscope", nargs="*")
    parser.add_argument("--truth_db", nargs="*")
    parser.add_argument("--roc", nargs="*")
    parser.add_argument("--plane")
    parser.add_argument("--saveroi", action="store_true")
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--flip", nargs="*")
    parser.add_argument("--transform", nargs="*")
    parser.add_argument("--register")
    parser.add_argument("--stats")
    parser.add_argument("--plot_2d")
    parser.add_argument("--reg_profile")
    parser.add_argument("--rescale")
    parser.add_argument("--slice")
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
    parser.add_argument("--alphas")
    parser.add_argument("--vmin")
    parser.add_argument("--vmax")
    parser.add_argument("--seed")
    parser.add_argument("--reg_suffixes", nargs="*")
    parser.add_argument("--no_scale_bar", action="store_true")
    parser.add_argument("--plot_labels", nargs="*")
    parser.add_argument("--theme", nargs="*")
    args = parser.parse_args()
    
    if args.img is not None:
        # set image file path and convert to basis for additional paths
        config.filenames = args.img
        config.filename = config.filenames[0]
        print("Set filenames to {}, current filename {}"
              .format(config.filenames, config.filename))

    if args.meta is not None:
        # set metadata paths
        config.metadata_paths = args.meta
        print("Set metadata paths to", config.metadata_paths)
        config.metadatas = []
        for path in config.metadata_paths:
            # load metadata to dictionary
            md, _ = importer.load_metadata(path, assign=False)
            config.metadatas.append(md)

    if args.channel is not None:
        # set the channel; currently supports a single channel or -1 for all
        # TODO: consider allowing array to support multiple but not 
        # necessarily all channels; would need to match num of profiles and 
        # index based on order channels
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
        config.savefig = args.savefig
        print("Set savefig extension to {}".format(config.savefig))
    if args.verbose:
        # verbose mode, including printing longer Numpy arrays for debugging
        config.verbose = args.verbose
        np.set_printoptions(linewidth=200, threshold=10000)
        print("Set verbose to {}".format(config.verbose))
    if args.roc:
        config.roc = args.roc
        print("Set ROC to {}".format(config.roc))
    if args.offset is not None:
        config.roi_offsets = _parse_coords(args.offset)
        config.roi_offset = config.roi_offsets[0]
        print("Set offsets to {}, current offset {}"
              .format(config.roi_offsets, config.roi_offset))
    if args.size is not None:
        config.roi_sizes = _parse_coords(args.size)
        config.roi_size = config.roi_sizes[0]
        print("Set ROI sizes to {}, current size {}"
              .format(config.roi_sizes, config.roi_size))
    if args.padding_2d is not None:
        padding_split = args.padding_2d.split(",")
        if len(padding_split) >= 3:
            from magmap.plot import plot_2d
            roi_editor.padding = tuple(int(i) for i in padding_split)
            print("Set plot_2d.padding to {}".format(
                roi_editor.padding))
        else:
            print("padding_2d ({}) should be given as 3 values (x, y, z)"
                  .format(args.padding_2d))
    
    # set up main processing mode
    if args.proc is not None:
        config.proc_type = args.proc
        print("processing type set to {}".format(config.proc_type))
    proc_type = libmag.get_enum(config.proc_type, config.ProcessTypes)
    if config.proc_type and proc_type not in config.ProcessTypes:
        libmag.warn(
            "\"{}\" processing type not found".format(config.proc_type))
    
    if args.res is not None:
        res_split = args.res.split(",")
        if len(res_split) >= 3:
            config.resolutions = [tuple(float(i) for i in res_split)[::-1]]
            print("Set resolutions to {}".format(config.resolutions))
        else:
            print("Resolution ({}) should be given as 3 values (x, y, z)"
                  .format(args.res))
    if args.mag:
        config.magnification = args.mag
        print("Set magnification to {}".format(config.magnification))
    if args.zoom:
        config.zoom = args.zoom
        print("Set zoom to {}".format(config.zoom))
    
    # initialize microscope profile settings and update with modifiers
    config.process_settings = profiles.ProcessSettings()
    config.process_settings_list.append(config.process_settings)
    if args.microscope is not None:
        for i in range(len(args.microscope)):
            settings = (config.process_settings if i == 0 
                        else profiles.ProcessSettings())
            profiles.update_process_settings(settings, args.microscope[i])
            if i > 0:
                config.process_settings_list.append(settings)
                print("Added {} settings for channel {}".format(
                      config.process_settings_list[i]["settings_name"], i))
    print("Set default microscope processing settings to {}"
          .format(config.process_settings["settings_name"]))
    
    # initialize registration profile settings and update with modifiers
    config.register_settings = profiles.RegisterSettings()
    if args.reg_profile is not None:
        profiles.update_register_settings(
            config.register_settings, args.reg_profile)
    print("Set register settings to {}"
          .format(config.register_settings["settings_name"]))
    
    if args.plane is not None:
        from magmap.plot import plot_2d
        config.plane = args.plane
        print("Set plane to {}".format(config.plane))
    if args.saveroi:
        config.saveroi = args.saveroi
        print("Set save ROI to file to ".format(config.saveroi))
    if args.labels:
        # atlas labels as positional or dictionary-like args
        config.atlas_labels = args_to_dict(
            args.labels, config.AtlasLabels, config.atlas_labels)
        config.load_labels = config.atlas_labels[config.AtlasLabels.PATH_REF]
        config.labels_level = config.atlas_labels[config.AtlasLabels.LEVEL]
        print("Set labels to {}".format(config.atlas_labels))

    if args.flip:
        config.flip = []
        for flip in args.flip:
            config.flip.append(_is_arg_true(flip))
        print("Set flip to {}".format(config.flip))

    if args.transform is not None:
        # image transformations such as flipping, rotation;
        # TODO: consider superseding the flip arg by incorporation here
        config.transform = args_to_dict(
            args.transform, config.Transforms, config.transform)
        print("Set transformations to {}".format(config.transform))

    if args.register:
        # register type to process in register module
        config.register_type = args.register
        print("Set register type to {}".format(config.register_type))
    
    if args.stats:
        # stats type to process in stats module
        config.stats_type = args.stats
        print("Set stats type to {}".format(config.stats_type))
    
    if args.plot_2d:
        # 2D plot type to process in plot_2d module
        config.plot_2d_type = args.plot_2d
        print("Set plot_2d type to {}".format(config.plot_2d_type))
    
    if args.rescale:
        config.rescale = float(args.rescale)
        print("Set rescale to {}".format(config.rescale))
    if args.slice:
        # specify a generic slice by command-line, assuming same order 
        # of arguments as for slice built-in function and interpreting 
        # "none" string as None
        config.slice_vals = args.slice.split(",")
        config.slice_vals = [None if val.lower() == "none" else int(val)
                             for val in config.slice_vals]
        print("Set slice values to {}".format(config.slice_vals))
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
    
    if args.alphas:
        # specify alpha levels
        config.alphas = [float(val) for val in args.alphas.split(",")]
        print("Set alphas to", config.alphas)
    
    if args.vmin:
        # specify vmin levels
        config.vmins = [
            libmag.get_int(val) for val in args.vmin.split(",")]
        print("Set vmins to", config.vmins)
    
    if args.vmax:
        # specify vmax levels and copy to vmax overview used for plotting 
        # and updated for normalization
        config.vmaxs = [
            libmag.get_int(val) for val in args.vmax.split(",")]
        config.vmax_overview = list(config.vmaxs)
        print("Set vmaxs to", config.vmaxs)
    
    if args.reg_suffixes is not None:
        # specify suffixes of registered images to load
        config.reg_suffixes = args_to_dict(
            args.reg_suffixes, config.RegSuffixes, config.reg_suffixes)
        print("Set registered image suffixes to {}".format(config.reg_suffixes))
    
    if args.seed:
        # specify random number generator seed
        config.seed = int(args.seed)
        print("Set random number generator seed to", config.seed)
    
    if args.no_scale_bar:
        # turn off scale bar display
        config.scale_bar = False
        print("Set scale bar display to {}".format(config.scale_bar))
    
    if args.plot_labels is not None:
        # specify general plot labels
        config.plot_labels = args_to_dict(
            args.plot_labels, config.PlotLabels, config.plot_labels)
        print("Set plot labels to {}".format(config.plot_labels))

    if args.theme is not None:
        # specify themes, currently applied to Matplotlib elements
        theme_names = []
        for theme in args.theme:
            # add theme enum if found
            theme_enum = libmag.get_enum(theme, config.Themes)
            if theme_enum:
                config.rc_params.append(theme_enum)
                theme_names.append(theme_enum.name)
        print("Set to use themes to {}".format(theme_names))

    # prep filename
    if not config.filename:
        # unable to parse anymore args without filename
        print("filename not specified, stopping argparsing")
        return
    filename_base = importer.filename_to_base(
        config.filename, config.series)

    # Database prep
    
    if args.db:
        config.db_name = args.db
        print("Set database name to {}".format(config.db_name))
    
    # load "truth blobs" from separate database for viewing
    if args.truth_db is not None:
        # set the truth database mode
        config.truth_db_params = args_to_dict(
            args.truth_db, config.TruthDB, config.truth_db_params)
        mode = config.truth_db_params[config.TruthDB.MODE]
        config.truth_db_mode = libmag.get_enum(mode, config.TruthDBModes)
        print("Mapped \"{}\" truth_db mode to {}"
              .format(mode, config.truth_db_mode))
    truth_db_path = config.truth_db_params[config.TruthDB.PATH]
    if config.truth_db_mode is config.TruthDBModes.VIEW:
        # loads truth DB as a separate database in parallel with the given 
        # editable database, with name based on filename by default unless 
        # truth DB name explicitly given
        path = truth_db_path if truth_db_path else filename_base
        try:
            sqlite.load_truth_db(path)
        except FileNotFoundError as e:
            print(e)
            print("Could not load truth DB from current image path")
    elif config.truth_db_mode is config.TruthDBModes.VERIFY:
        # creates a new verified DB to store all ROC results
        config.verified_db = sqlite.ClrDB()
        config.verified_db.load_db(sqlite.DB_NAME_VERIFIED, True)
        if truth_db_path:
            # load truth DB path to verify against if explicitly given
            try:
                sqlite.load_truth_db(truth_db_path)
            except FileNotFoundError as e:
                print(e)
                print("Could not load truth DB from {}"
                      .format(truth_db_path))
    elif config.truth_db_mode is config.TruthDBModes.VERIFIED:
        # loads verified DB as the main DB, which includes copies of truth 
        # values with flags for whether they were detected
        path = sqlite.DB_NAME_VERIFIED
        if truth_db_path: path = truth_db_path
        try:
            config.db = sqlite.ClrDB()
            config.db.load_db(path)
            config.verified_db = config.db
        except FileNotFoundError as e:
            print(e)
            print("Could not load verified DB from {}"
                  .format(sqlite.DB_NAME_VERIFIED))
    elif config.truth_db_mode is config.TruthDBModes.EDIT:
        # loads truth DB as the main database for editing rather than 
        # loading as a truth database
        config.db_name = truth_db_path
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
        if config.roc:
            # grid search(es) for the specified hyperparameter groups
            stats_dict = mlearn.grid_search(
                profiles.roc_dict, config.roc, _iterate_file_processing,
                config.filename, series, config.roi_offsets, config.roi_sizes)
            parsed_dict, stats_dfs = mlearn.parse_grid_stats(stats_dict)
            from magmap.plot import plot_2d
            plot_2d.setup_style()
            for stats_df in stats_dfs:
                # plot ROC curve
                plot_2d.plot_roc(stats_df, not config.no_show)
        else:
            # processes file with default settings
            np_io.setup_images(
                config.filename, series, config.roi_offset,
                config.roi_size, config.proc_type)
            process_file(
                config.filename, series, config.roi_offset,
                config.roi_size, config.proc_type)
    
    # unless loading images for GUI, exit directly since otherwise application 
    # hangs if launched from module with GUI
    if proc_type is not None and proc_type is not config.ProcessTypes.LOAD:
        shutdown()


def _iterate_file_processing(path, series, offsets, roi_sizes):
    """Processes files iteratively based on offsets.
    
    Args:
        path (str): Path to image from which MagellanMapper-style paths will 
            be generated.
        series (int): Image series number.
        offsets: 2D array of multiple offsets.
        roi_sizes: 2D array of multiple ROI sizes corresponding to offsets.
    
    Returns:
        :obj:`np.ndarray`, str: Summed stats array and concatenated summaries.
    """
    stat = np.zeros(3)
    roi_sizes_len = len(roi_sizes)
    summaries = []
    for i in range(len(offsets)):
        size = (roi_sizes[i] if roi_sizes_len > 1 
                else roi_sizes[0])
        np_io.setup_images(path, series, offsets[i], size, config.proc_type)
        stat_roi, fdbk = process_file(
            path, series, offsets[i], size, config.proc_type)
        if stat_roi is not None:
            stat = np.add(stat, stat_roi)
        summaries.append(
            "Offset {}:\n{}".format(offsets[i], fdbk))
    return stat, summaries


def process_file(path, series, offset, roi_size, proc_mode):
    """Processes a single image file non-interactively.
    
    Args:
        path (str): Path to image from which MagellanMapper-style paths will 
            be generated.
        series (int): Image series number.
        offset: Offset as (x, y, z) to start processing.
        roi_size: Size of region to process, given as (x, y, z).
        proc_mode (str): Processing mode, which should be a key in 
            :class:`config.ProcessTypes`, case-insensitive.
    
    Returns:
        Tuple of stats from processing, or None if no stats, and 
        text feedback from the processing, or None if no feedback.
    """
    # PROCESS BY TYPE
    stats = None
    fdbk = None
    filename_base = importer.filename_to_base(path, series)
    proc_type = libmag.get_enum(proc_mode, config.ProcessTypes)
    if proc_type is config.ProcessTypes.LOAD:
        # loading completed
        return None, None

    elif proc_type is config.ProcessTypes.LOAD:
        # already imported so does nothing
        print("imported {}, will exit".format(path))
    
    elif proc_type is config.ProcessTypes.EXPORT_ROIS:
        # export ROIs; assumes that info_proc was already loaded to 
        # give smaller region from which smaller ROIs from the truth DB 
        # will be extracted
        from magmap.io import export_rois
        db = config.db if config.truth_db is None else config.truth_db
        export_rois.export_rois(
            db, config.image5d, config.channel, filename_base, config.border, 
            config.unit_factor, config.truth_db_mode,
            os.path.basename(config.filename))
        
    elif proc_type is config.ProcessTypes.TRANSFORM:
        # transpose and/or rescale whole large image
        transformer.transpose_img(
            path, series, plane=config.plane, 
            rescale=config.rescale)
        
    elif proc_type in (
            config.ProcessTypes.EXTRACT, config.ProcessTypes.ANIMATED):
        # generate animated GIF or extract single plane
        from magmap.io import export_stack
        export_stack.stack_to_img(
            config.filenames, series, offset, roi_size,
            proc_type is config.ProcessTypes.ANIMATED, config.suffix)
    
    elif proc_type is config.ProcessTypes.EXPORT_BLOBS:
        # export blobs to CSV file
        from magmap.io import export_rois
        export_rois.blobs_to_csv(config.blobs, filename_base)
        
    elif proc_type in (
            config.ProcessTypes.PROCESSING, config.ProcessTypes.PROCESSING_MP):
        # detect blobs in the full image
        stats, fdbk, segments_all = stack_detect.detect_blobs_large_image(
            filename_base, config.image5d, offset, roi_size,
            config.truth_db_mode is config.TruthDBModes.VERIFY, 
            not config.roc, config.image5d_is_roi)
    
    return stats, fdbk


def shutdown():
    """Clean up and shutdown MagellanMapper.

    Stops any running Java virtual machine and closes any main database.
    """
    importer.stop_jvm()
    if config.db is not None:
        config.db.conn.close()
    sys.exit()

    
if __name__ == "__main__":
    print("Starting MagellanMapper command-line interface...")
    main()
