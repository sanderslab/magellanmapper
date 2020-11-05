#!/usr/bin/env python
# Command line parsing and setup
# Author: David Young, 2017, 2020
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
    * plane: Plane type (see :const:``config.PLANE``).
    * save_subimg: Save sub-image during stack processing.
    * register: Registration type. See :attr:``config.REGISTER_TYPES`` for 
        types of registration and :mod:``register`` for how to use these 
        types.
    * labels: Load annotation JSON file. The first argument is the path 
        to the JSON file. If a 2nd arguments is given, it is taken as an int of 
        the ontology level for grouping volumes.
    * slice: ``stop`` or ``start,stop[,step]`` values to create a slice
        object, such as for animated GIF stack planes.
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

from magmap.atlas import register, transformer
from magmap.cloud import notify
from magmap.cv import chunking, colocalizer, stack_detect
from magmap.io import df_io, export_stack, importer, libmag, np_io, sqlite
from magmap.plot import colormaps, plot_2d
from magmap.settings import atlas_prof, config, grid_search_prof, roi_prof
from magmap.stats import mlearn


def _parse_coords(arg, rev=False):
    # parse a list of strings into 3D coordinates
    coords = []  # copy list to avoid altering the arg itself
    for coord in arg:
        if not coord: continue
        coord_split = coord.split(",")
        if len(coord_split) >= 3:
            coord = tuple(int(i) for i in coord_split)
            if rev:
                coord = coord[::-1]
            coords.append(coord)
        else:
            print("Coordinates ({}) should be given as 3 values (x, y, z)"
                  .format(coord))
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


def args_to_dict(args, keys_enum, args_dict=None, sep_args="=", sep_vals=",",
                 default=None):
    """Parse arguments list with positional and keyword-based arguments 
    into an enum-keyed dictionary.
    
    Args:
        args (List[str]): List of arguments with positional values followed by
            ``sep_args``-delimited values. Positional values will be entered
            in the existing order of ``keys_enum`` based on member values, 
            while keyword-based values will be entered if an enum 
            member corresponding to the keyword exists.
            Entries can also be ``sep_vals``-delimited to specify lists.
        keys_enum (Enum): Enum to use as keys for dictionary. Values are
            assumed to range from 1 to number of members as output 
            by the default Enum functional API.
        args_dict (dict): Dictionary to be filled or updated with keys from
            ``keys_enum``; defaults to None, which will assign an empty dict.
        sep_args (str): Separator between arguments and values; defaults to "=".
        sep_vals (str): Separator within values; defaults to ",".
        default (str): Default value for each argument. Effectively turns off
            positional argument assignments since all args become
            ``<keyword>=<default>``. Defaults to None.
    
    Returns:
        dict: Dictionary filled with arguments. Values that contain commas
        will be split into comma-delimited lists. All values will be 
        converted to ints if possible.
    
    """
    if args_dict is None:
        args_dict = {}
    by_position = True
    num_enums = len(keys_enum)
    for i, arg in enumerate(args):
        arg_split = arg.split(sep_args)
        if default and len(arg_split) < 2:
            # add default value unless another value is given
            arg_split.append(default)
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
            vals_split = vals.split(sep_vals)
            if len(vals_split) > 1:
                # use split value if comma-delimited
                vals = vals_split
            # cast to numeric types if possible and assign to found enum
            args_dict[key] = libmag.get_int(vals)
    return args_dict


def _get_args_dict_help(msg, keys):
    """Get the help message for command-line arguments that are converted
    to a dictionary.
    
    Args:
        msg (str): Message to prepend to the help message.
        keys (Enum): Keys as an Enumeration.

    Returns:
        str: Help message with available keys.

    """
    return ("{} Available keys follow this order until the first "
            "key=value pair is given: {}".format(
                msg, libmag.enum_names_aslist(keys)))


def main(process_args_only=False, skip_dbs=False):
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    
    Args:
        process_args_only (bool): Processes command-line arguments and
            returns; defaults to False.
        skip_dbs (bool): True to skip loading databases; defaults to False.
    """
    parser = argparse.ArgumentParser(
        description="Setup environment for MagellanMapper")

    # image specification arguments
    parser.add_argument(
        "--img", nargs="*",
        help="Main image path(s); after import, the filename is often "
             "given as the original name without its extension")
    parser.add_argument(
        "--meta", nargs="*",
        help="Metadata path(s), which can be given as multiple files "
             "corresponding to each image")
    parser.add_argument("--prefix", help="Path prefix")
    parser.add_argument("--suffix", help="Filename suffix")
    parser.add_argument("--channel", nargs="*", type=int, help="Channel index")
    parser.add_argument("--series", help="Series index")
    parser.add_argument(
        "--subimg_offset", nargs="*", help="Sub-image offset in x,y,z")
    parser.add_argument(
        "--subimg_size", nargs="*", help="Sub-image size in x,y,z")
    parser.add_argument("--offset", nargs="*", help="ROI offset in x,y,z")
    parser.add_argument("--size", nargs="*", help="ROI size in x,y,z")
    parser.add_argument("--db", help="Database path")
    parser.add_argument(
        "--cpus",
        help="Maximum number of CPUs/processes to use for multiprocessing "
             "tasks. Use \"none\" or 0 to auto-detect this number (default).")
    parser.add_argument(
        "--load", nargs="*",
        help="Load associated data files; see config.LoadData for settings")

    # task arguments
    parser.add_argument(
        "--proc", type=str.lower,
        choices=libmag.enum_names_aslist(config.ProcessTypes),
        help="Image processing mode")
    parser.add_argument(
        "--register", type=str.lower,
        choices=libmag.enum_names_aslist(config.RegisterTypes),
        help="Image registration task")
    parser.add_argument(
        "--df", type=str.lower,
        choices=libmag.enum_names_aslist(config.DFTasks),
        help="Data frame task")
    parser.add_argument(
        "--plot_2d", type=str.lower,
        choices=libmag.enum_names_aslist(config.Plot2DTypes),
        help="2D plot task; see config.Plot2DTypes")
    parser.add_argument("--ec2_start", nargs="*", help="AWS EC2 instance start")
    parser.add_argument("--ec2_list", nargs="*", help="AWS EC2 instance list")
    parser.add_argument(
        "--ec2_terminate", nargs="*", help="AWS EC2 instance termination")
    parser.add_argument(
        "--notify", nargs="*",
        help="Notification message URL, message, and attachment strings")
    parser.add_argument(
        "--grid_search", help="Grid search hyperparameter tuning profile(s)")

    # profile arguments
    parser.add_argument(
        "--roi_profile", nargs="*",
        help="ROI profile, which can be separated by underscores "
             "for multiple profiles and given as paths to custom profiles "
             "in YAML format. Multiple profile groups can be given, which "
             "will each be applied to the corresponding channel. See "
             "docs/settings.md for more details.")
    parser.add_argument(
        "--atlas_profile",
        help="Atlas profile, which can be separated by underscores "
             "for multiple profiles and given as paths to custom profiles "
             "in YAML format. See docs/settings.md for more details.")
    parser.add_argument(
        "--theme", nargs="*", type=str.lower,
        choices=libmag.enum_names_aslist(config.Themes),
        help="UI theme, which can be given as multiple themes to apply "
             "on top of one another")

    # grouped arguments
    parser.add_argument(
        "--truth_db", nargs="*",
        help="Truth database; see config.TruthDB for settings and "
             "config.TruthDBModes for modes")
    parser.add_argument(
        "--labels", nargs="*",
        help=_get_args_dict_help(
            "Atlas labels; see config.AtlasLabels.", config.AtlasLabels))
    parser.add_argument(
        "--transform", nargs="*",
        help=_get_args_dict_help(
            "Image transformations; see config.Transforms.", config.Transforms))
    parser.add_argument(
        "--reg_suffixes", nargs="*",
        help=_get_args_dict_help(
            "Registered image suffixes; see config.RegSuffixes for keys "
            "and config.RegNames for values", config.RegSuffixes))
    parser.add_argument(
        "--plot_labels", nargs="*",
        help=_get_args_dict_help(
            "Plot label customizations; see config.PlotLabels ",
            config.PlotLabels))
    parser.add_argument(
        "--set_meta", nargs="*",
        help="Set metadata values; see config.MetaKeys for settings")

    # image and figure display arguments
    parser.add_argument(
        "--plane", type=str.lower, choices=config.PLANE,
        help="Planar orientation")
    parser.add_argument(
        "--show", nargs="?", const="1",
        help="If applicable, show images after completing the given task")
    parser.add_argument(
        "--alphas",
        help="Alpha opacity levels, which can be comma-delimited for "
             "multichannel images")
    parser.add_argument(
        "--vmin",
        help="Minimum intensity levels, which can be comma-delimited "
             "for multichannel images")
    parser.add_argument(
        "--vmax",
        help="Maximum intensity levels, which can be comma-delimited "
             "for multichannel images")
    parser.add_argument("--seed", help="Random number generator seed")

    # export arguments
    parser.add_argument(
        "--save_subimg", action="store_true",
        help="Save sub-image as separate file")
    parser.add_argument("--slice", help="Slice given as start,stop,step")
    parser.add_argument("--delay", help="Animation delay in ms")
    parser.add_argument(
        "--savefig", help="Extension for saved figures")
    parser.add_argument(
        "--groups", nargs="*", help="Group values corresponding to each image")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output to assist with debugging")
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
        # set the channels
        config.channel = args.channel
        print("Set channel to {}".format(config.channel))
    
    series_list = [config.series]  # list of series
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
        # save figure with file type of this extension; remove leading period
        config.savefig = args.savefig.lstrip(".")
        print("Set savefig extension to {}".format(config.savefig))

    if args.verbose:
        # verbose mode, including printing longer Numpy arrays for debugging
        config.verbose = args.verbose
        np.set_printoptions(linewidth=200, threshold=10000)
        print("Set verbose to {}".format(config.verbose))

    # parse sub-image offsets and sizes;
    # expects x,y,z input but stores as z,y,x by convention
    if args.subimg_offset is not None:
        config.subimg_offsets = _parse_coords(args.subimg_offset, True)
        print("Set sub-image offsets to {} (z,y,x)"
              .format(config.subimg_offsets))
    if args.subimg_size is not None:
        config.subimg_sizes = _parse_coords(args.subimg_size, True)
        print("Set sub-image sizes to {} (z,y,x)"
              .format(config.subimg_sizes))

    # parse ROI offsets and sizes, which are relative to any sub-image;
    # expects x,y,z input and output
    if args.offset is not None:
        config.roi_offsets = _parse_coords(args.offset)
        if config.roi_offsets:
            config.roi_offset = config.roi_offsets[0]
        print("Set ROI offsets to {}, current offset {} (x,y,z)"
              .format(config.roi_offsets, config.roi_offset))
    if args.size is not None:
        config.roi_sizes = _parse_coords(args.size)
        if config.roi_sizes:
            config.roi_size = config.roi_sizes[0]
        print("Set ROI sizes to {}, current size {} (x,y,z)"
              .format(config.roi_sizes, config.roi_size))
    
    if args.cpus is not None:
        # set maximum number of CPUs
        config.cpus = (None if args.cpus.lower() in ("none", "0")
                       else int(args.cpus))
        print("Set maximum number of CPUs for multiprocessing tasks to",
              config.cpus)

    if args.load is not None:
        # flag loading data sources, specified by the source alone to use a
        # default path for it, or giving a path as a sub-arg
        config.load_data = args_to_dict(
            args.load, config.LoadData, config.load_data, default="1")
        print("Set to load the data types: {}".format(config.load_data))

    # set up main processing mode
    if args.proc is not None:
        config.proc_type = args.proc
        print("processing type set to {}".format(config.proc_type))
    proc_type = libmag.get_enum(config.proc_type, config.ProcessTypes)
    if config.proc_type and proc_type not in config.ProcessTypes:
        libmag.warn(
            "\"{}\" processing type not found".format(config.proc_type))

    if args.set_meta is not None:
        # set individual metadata values, currently used for image import
        # TODO: take precedence over loaded metadata archives
        config.meta_dict = args_to_dict(
            args.set_meta, config.MetaKeys, config.meta_dict, sep_vals="|")
        print("Set metadata values to {}".format(config.meta_dict))
        res = config.meta_dict[config.MetaKeys.RESOLUTIONS]
        if res:
            # set image resolutions, taken as a single set of x,y,z and
            # converting to a nested list of z,y,x
            res_split = res.split(",")
            if len(res_split) >= 3:
                res_float = tuple(float(i) for i in res_split)[::-1]
                config.resolutions = [res_float]
                print("Set resolutions to {}".format(config.resolutions))
            else:
                res_float = None
                print("Resolution ({}) should be given as 3 values (x,y,z)"
                      .format(res))
            # store single set of resolutions, similar to input
            config.meta_dict[config.MetaKeys.RESOLUTIONS] = res_float
        mag = config.meta_dict[config.MetaKeys.MAGNIFICATION]
        if mag:
            # set objective magnification
            config.magnification = mag
            print("Set magnification to {}".format(config.magnification))
        zoom = config.meta_dict[config.MetaKeys.ZOOM]
        if zoom:
            # set objective zoom
            config.zoom = zoom
            print("Set zoom to {}".format(config.zoom))
        shape = config.meta_dict[config.MetaKeys.SHAPE]
        if shape:
            # parse shape, storing only in dict
            config.meta_dict[config.MetaKeys.SHAPE] = [
                int(n) for n in shape.split(",")[::-1]]

    # set up ROI and register profiles
    setup_profiles(args.roi_profile, args.atlas_profile, args.grid_search)

    if args.plane is not None:
        config.plane = args.plane
        print("Set plane to {}".format(config.plane))
    if args.save_subimg:
        config.save_subimg = args.save_subimg
        print("Set to save the sub-image")
    
    if args.labels:
        # set up atlas labels
        setup_labels(args.labels)

    if args.transform is not None:
        # image transformations such as flipping, rotation
        config.transform = args_to_dict(
            args.transform, config.Transforms, config.transform)
        print("Set transformations to {}".format(config.transform))

    if args.register:
        # register type to process in register module
        config.register_type = args.register
        print("Set register type to {}".format(config.register_type))
    
    if args.df:
        # data frame processing task
        config.df_task = args.df
        print("Set data frame processing task to {}".format(config.df_task))
    
    if args.plot_2d:
        # 2D plot type to process in plot_2d module
        config.plot_2d_type = args.plot_2d
        print("Set plot_2d type to {}".format(config.plot_2d_type))

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

    if args.show:
        # show images after task is performed, if supported
        config.show = _is_arg_true(args.show)
        print("Set show to {}".format(config.show))

    if args.groups:
        config.groups = args.groups
        print("Set groups to {}".format(config.groups))
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
    # set up Matplotlib styles/themes
    plot_2d.setup_style()

    # prep filename
    filename_base = None
    if config.filename:
        filename_base = importer.filename_to_base(
            config.filename, config.series)
    
    if not skip_dbs:
        setup_dbs(filename_base, args.db, args.truth_db)
    
    # set multiprocessing start method
    chunking.set_mp_start_method()

    # POST-ARGUMENT PARSING

    if process_args_only:
        return
    
    # if command-line driven task specified, start task and shut down
    if config.register_type:
        register.main()
    elif config.notify_url:
        notify.main()
    elif config.plot_2d_type:
        plot_2d.main()
    elif config.df_task:
        df_io.main()
    elif config.grid_search_profile:
        _grid_search(series_list)
    elif config.ec2_list or config.ec2_start or config.ec2_terminate:
        # defer importing AWS module to avoid making its dependencies
        # required for MagellanMapper
        from magmap.cloud import aws
        aws.main()
    else:
        # set up image and perform any whole image processing tasks;
        # do not shut down if not a command-line proc task
        _process_files(series_list)
        if proc_type is None or proc_type is config.ProcessTypes.LOAD:
            return
    shutdown()


def setup_dbs(filename_base, db_path=None, truth_db_config=None):
    """Set up databases for the given image file if the given database has
    not been set up already.
    
    Args:
        filename_base (str): Image base path.
        db_path (str): Main database path; defaults to None to use a default
            path.
        truth_db_config (List[str]): Sequence of truth database configuration
            settings; defaults to None to not load truth-related databases.
    
    """
    if db_path:
        config.db_name = db_path
        print("Set database name to {}".format(config.db_name))
    
    # load "truth blobs" from separate database for viewing
    if truth_db_config is not None:
        # set the truth database mode
        config.truth_db_params = args_to_dict(
            truth_db_config, config.TruthDB, config.truth_db_params,
            sep_vals="|")
        mode = config.truth_db_params[config.TruthDB.MODE]
        config.truth_db_mode = libmag.get_enum(mode, config.TruthDBModes)
        libmag.printv(config.truth_db_params)
        print("Mapped \"{}\" truth_db mode to {}"
              .format(mode, config.truth_db_mode))
    truth_db_path = config.truth_db_params[config.TruthDB.PATH]
    truth_db_name_base = filename_base if filename_base else sqlite.DB_NAME_BASE
    if config.truth_db_mode is config.TruthDBModes.VIEW:
        # loads truth DB as a separate database in parallel with the given 
        # editable database, with name based on filename by default unless 
        # truth DB name explicitly given
        path = truth_db_path if truth_db_path else truth_db_name_base
        try:
            sqlite.load_truth_db(path)
        except FileNotFoundError as e:
            print(e)
            print("Could not load truth DB from current image path")
    elif config.truth_db_mode is config.TruthDBModes.VERIFY:
        if not config.verified_db:
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
                os.path.basename(truth_db_name_base), sqlite.DB_SUFFIX_TRUTH)
        print("Editing truth database at {}".format(config.db_name))
    
    if config.db is None:
        config.db = sqlite.ClrDB()
        config.db.load_db(None, False)


def setup_profiles(roi_profiles_names, atlas_profiles_names,
                   grid_search_profiles_names):
    """Setup ROI, atlas, and grid search profiles.

    If a profile is None, only a default set of profile settings
    will be generated. Also sets up colormaps based on ROI profiles. Any
    previously set up profile will be replaced.

    Args:
        roi_profiles_names (List[str]): Sequence of ROI and atlas profiles
            names to use for the corresponding channel.
        atlas_profiles_names (str): Atlas profiles names.
        grid_search_profiles_names (str): Grid search profiles names.

    """
    # initialize ROI profile settings and update with modifiers
    config.roi_profile = roi_prof.ROIProfile()
    config.roi_profiles = [config.roi_profile]
    if roi_profiles_names is not None:
        for i, roi_prof_name in enumerate(roi_profiles_names):
            print("Updating ROI profile for channel", i)
            if i == 0:
                settings = config.roi_profile
            else:
                settings = roi_prof.ROIProfile()
                config.roi_profiles.append(settings)
            settings.update_settings(roi_prof_name)
    for i, prof in enumerate(config.roi_profiles):
        if i == 0:
            print("Set default (channel 0) ROI profile: {}"
                  .format(prof[prof.NAME_KEY]))
        else:
            print("Added channel {} ROI profile: {}".format(
                  i, prof[prof.NAME_KEY]))
    colormaps.setup_colormaps(np_io.get_num_channels(config.image5d))

    # initialize atlas profile and update with modifiers
    config.atlas_profile = atlas_prof.AtlasProfile()
    if atlas_profiles_names is not None:
        config.atlas_profile.update_settings(atlas_profiles_names)
    print("Set atlas profile to {}"
          .format(config.atlas_profile[config.atlas_profile.NAME_KEY]))

    if grid_search_profiles_names:
        # parse grid search profiles
        config.grid_search_profile = grid_search_prof.GridSearchProfile()
        config.grid_search_profile.update_settings(grid_search_profiles_names)
        print("Set grid search profile to {}".format(
            config.grid_search_profile[config.grid_search_profile.NAME_KEY]))
        print(config.grid_search_profile)


def update_profiles():
    """Update profiles if any profile file has been modified since it
    was last loaded.

    Profiles in both :attr:`config.process_settings_list` and
    :attr:`config.register_settings_list` will be checked to update.

    """
    for i, prof in enumerate(config.roi_profiles):
        prof.refresh_profile(True)
    config.atlas_profile.refresh_profile(True)


def setup_labels(labels_arg):
    """Set up atlas labels.
    
    Args:
        labels_arg (str): Path to labels reference file, such as a labels
            ontology file.

    """
    # atlas labels as positional or dictionary-like args
    config.atlas_labels = args_to_dict(
        labels_arg, config.AtlasLabels, config.atlas_labels)
    config.load_labels = config.atlas_labels[config.AtlasLabels.PATH_REF]
    config.labels_level = config.atlas_labels[config.AtlasLabels.LEVEL]
    print("Set labels to {}".format(config.atlas_labels))


def _iterate_file_processing(path, series, subimg_offsets, subimg_sizes):
    """Processes files iteratively based on offsets.
    
    Args:
        path (str): Path to image from which MagellanMapper-style paths will 
            be generated.
        series (int): Image series number.
        subimg_offsets (List[List[int]]): Nested list of sub-image offset sets
            given as ``[[offset_z1, offset_y1, offset_x1], ...]``.
        subimg_sizes (List[List[int]]): Nested list of sub-image size sets
            given as ``[[offset_z1, offset_y1, offset_x1], ...]`` and
            corresponding to ``subimg_offsets``.
    
    Returns:
        :obj:`np.ndarray`, str: Summed stats array and concatenated summaries.
    """
    stat = np.zeros(3)

    # use whole image if sub-image parameters are not set
    if subimg_offsets is None:
        subimg_offsets = [None]
    if subimg_sizes is None:
        subimg_sizes = [None]
    roi_sizes_len = len(subimg_sizes)

    summaries = []
    for i in range(len(subimg_offsets)):
        size = (subimg_sizes[i] if roi_sizes_len > 1
                else subimg_sizes[0])
        np_io.setup_images(
            path, series, subimg_offsets[i], size, config.proc_type)
        stat_roi, fdbk = process_file(
            path, config.proc_type, series, subimg_offsets[i], size)
        if stat_roi is not None:
            stat = np.add(stat, stat_roi)
        summaries.append(
            "Offset {}:\n{}".format(subimg_offsets[i], fdbk))
    return stat, summaries


def _grid_search(series_list):
    # grid search(es) for the specified hyperparameter groups
    if not config.filename:
        print("No image filename set for grid search, skipping")
        return
    for series in series_list:
        # process each series, typically a tile within an microscopy image
        # set or a single whole image
        stats_dict = mlearn.grid_search(
            config.grid_search_profile, _iterate_file_processing,
            config.filename, series, config.subimg_offsets,
            config.subimg_sizes)
        parsed_dict, stats_dfs = mlearn.parse_grid_stats(stats_dict)
        for stats_df in stats_dfs:
            # plot ROC curve
            plot_2d.plot_roc(stats_df, config.show)


def _process_files(series_list):
    # wrapper to process files for each series, typically a tile within
    # an microscopy image set or a single whole image, setting up the
    # image before each processing
    if not config.filename:
        print("No image filename set for processing files, skipping")
        return
    for series in series_list:
        # process each series
        offset = config.subimg_offsets[0] if config.subimg_offsets else None
        size = config.subimg_sizes[0] if config.subimg_sizes else None
        np_io.setup_images(
            config.filename, series, offset, size, config.proc_type)
        process_file(
            config.filename, config.proc_type, series, offset, size,
            config.roi_offsets[0] if config.roi_offsets else None,
            config.roi_sizes[0] if config.roi_sizes else None)


def process_file(path, proc_mode, series=None, subimg_offset=None,
                 subimg_size=None, roi_offset=None, roi_size=None):
    """Processes a single image file non-interactively.

    Assumes that the image has already been set up.
    
    Args:
        path (str): Path to image from which MagellanMapper-style paths will 
            be generated.
        proc_mode (str): Processing mode, which should be a key in
            :class:`config.ProcessTypes`, case-insensitive.
        series (int): Image series number; defaults to None.
        subimg_offset (List[int]): Sub-image offset as (z,y,x) to load;
            defaults to None.
        subimg_size (List[int]): Sub-image size as (z,y,x) to load;
            defaults to None.
        roi_offset (List[int]): Region of interest offset as (x, y, z) to
            process; defaults to None.
        roi_size (List[int]): Region of interest size of region to process,
            given as (x, y, z); defaults to None.
    
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
            db, config.image5d, config.channel, filename_base,
            config.plot_labels[config.PlotLabels.PADDING],
            config.unit_factor, config.truth_db_mode,
            os.path.basename(config.filename))
        
    elif proc_type is config.ProcessTypes.TRANSFORM:
        # transpose, rescale, and/or resize whole large image
        transformer.transpose_img(
            path, series, plane=config.plane, 
            rescale=config.transform[config.Transforms.RESCALE],
            target_size=config.roi_size)
        
    elif proc_type in (
            config.ProcessTypes.EXTRACT, config.ProcessTypes.ANIMATED):
        # generate animated GIF or extract single plane
        export_stack.stack_to_img(
            config.filenames, roi_offset, roi_size, series, subimg_offset,
            subimg_size, proc_type is config.ProcessTypes.ANIMATED,
            config.suffix)
    
    elif proc_type is config.ProcessTypes.EXPORT_BLOBS:
        # export blobs to CSV file
        from magmap.io import export_rois
        export_rois.blobs_to_csv(config.blobs.blobs, filename_base)
        
    elif proc_type in (
            config.ProcessTypes.DETECT, config.ProcessTypes.DETECT_COLOC):
        # detect blobs in the full image, +/- co-localization
        coloc = proc_type is config.ProcessTypes.DETECT_COLOC
        stats, fdbk, segments_all = stack_detect.detect_blobs_large_image(
            filename_base, config.image5d, subimg_offset, subimg_size,
            config.truth_db_mode is config.TruthDBModes.VERIFY, 
            not config.grid_search_profile, config.image5d_is_roi, coloc)

    elif proc_type is config.ProcessTypes.COLOC_MATCH:
        if config.blobs is not None and config.blobs.blobs is not None:
            # colocalize blobs in separate channels by matching blobs
            shape = (config.image5d.shape[1:] if subimg_size is None
                     else subimg_size)
            matches = colocalizer.StackColocalizer.colocalize_stack(
                shape, config.blobs.blobs)
            
            # insert matches into database
            offset = (0, 0, 0) if subimg_offset is None else subimg_offset
            colocalizer.insert_matches(config.db, offset, shape[:3], matches)
        else:
            print("No blobs loaded to colocalize, skipping")

    elif proc_type is config.ProcessTypes.EXPORT_PLANES:
        # export each plane as a separate image file
        export_stack.export_planes(
            config.image5d, config.prefix, config.savefig, config.channel)
    
    elif proc_type is config.ProcessTypes.EXPORT_RAW:
        # export the main image as a raw data file
        out_path = libmag.combine_paths(config.filename, ".raw", sep="")
        libmag.backup_file(out_path)
        np_io.write_raw_file(config.image5d, out_path)

    elif proc_type is config.ProcessTypes.PREPROCESS:
        # pre-process a whole image and save to file
        # TODO: consider chunking option for larger images
        profile = config.get_roi_profile(0)
        out_path = config.prefix
        if not out_path:
            out_path = libmag.insert_before_ext(config.filename, "_preproc")
        transformer.preprocess_img(
            config.image5d, profile["preprocess"], config.channel, out_path)

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
