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
    offset::
        
        $ python -m magmap.cli --img /path/to/file.czi --offset 30,50,205 \\
            --size 150,150,10

For a table of command-line arguments and their usage, see:
https://github.com/sanderslab/magellanmapper/blob/master/docs/cli.md

"""

import argparse
import dataclasses
from enum import Enum
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, \
    Union

import numpy as np

from magmap.atlas import register, transformer
from magmap.cloud import notify
from magmap.cv import chunking, classifier, colocalizer, stack_detect
from magmap.io import df_io, export_stack, importer, libmag, naming, np_io, \
    sqlite
from magmap.plot import colormaps, plot_2d
from magmap.settings import atlas_prof, config, grid_search_prof, logs, \
    prefs_prof, roi_prof
from magmap.stats import mlearn

_logger = config.logger.getChild(__name__)


def _parse_coords(arg: str, rev: bool = False) -> List[Tuple[int, ...]]:
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


def _parse_none(
        arg: Any, fn: Optional[Callable] = None) -> Any:
    """Parse arguments with support for conversion to None.
    
    Args:
        arg: Argument to potentially convert.
        fn: Function to apply to ``arg`` if not converted; defaults to None.

    Returns:
        None if ``arg`` is "none" or "0"; otherwise, returns ``fn(arg)`` if
        ``fn`` is given, or ``arg`` unchanged.

    """
    if arg.lower() in ("none", "0"):
        return None
    return arg if fn is None else fn(arg)


def _is_arg_true(arg: str) -> bool:
    return arg.lower() == "true" or arg == "1"


def args_with_dict(args: List[str]) -> List[Union[Dict[str, Any], str, int]]:
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


def args_to_dict(
        args: List[str], keys: Union[Type[Enum], Type["config.DataClass"]],
        args_dict: Optional[Dict[Enum, Any]] = None, sep_args: str = "=",
        sep_vals: str = ",", default: Optional[Any] = None
) -> Union[Dict[Enum, Any], Type["config.DataClass"]]:
    """Parse positional and keyword-based arguments to an enum-keyed dictionary.
    
    Args:
        args: List of arguments with positional values followed by
            ``sep_args``-delimited values. Positional values will be entered
            in the existing order of ``keys`` based on member values,
            while keyword-based values will be entered if a member
            corresponding to the keyword exists. Entries can also be
            ``sep_vals``-delimited to specify lists.
        keys: Enum class or data class instance whose fields will be used as
            keys for dictionary. Enum values are assumed to range from 1 to
            number of members as output by the default Enum functional API.
        args_dict: Dictionary to be filled or updated with keys from
            ``keys_enum``. Defaults to None, which will assign an empty dict.
            Ignored if ``keys_class`` is a data class instance, which will
            be updated instead.
        sep_args: Separator between arguments and values; defaults to "=".
        sep_vals: Separator within values; defaults to ",".
        default: Default value for each argument. Effectively turns off
            positional argument assignments since all args become
            ``<keyword>=<default>``. Defaults to None. If a str, will
            undergo splitting by ``sep_vals``.
    
    Returns:
        Dictionary or data class corresponding to ``keys``, filled with
        arguments. Values that contain commas will be split into
        comma-delimited lists. All values will be converted to ints if possible.
    
    """
    is_data = dataclasses.is_dataclass(keys)
    if is_data:
        # use fields as keys
        keys_data = [f.name for f in dataclasses.fields(keys)]
        nkeys = len(keys_data)
        out = keys
    else:
        # use Enum members as keys
        keys_data = None
        nkeys = len(keys)
        out = {} if args_dict is None else args_dict
    
    by_position = True
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
            if n > nkeys:
                _logger.warn(
                    "No further parameters in '%s' to assign '%s' by "
                    "position, skipping", keys, arg)
                continue
            key = keys_data[n] if is_data else keys(n)
        
        elif len_arg_split < 2:
            _logger.warn(
                "Parameter '%s' does not contain a keyword, skipping", arg)
        
        else:
            # assign based on keyword if its equivalent enum exists
            key_str = arg_split[0]
            vals = arg_split[1]
            try:
                key = key_str.lower() if is_data else keys[key_str.upper()]
            except KeyError:
                _logger.warn(
                    "Unable to find '%s' in %s, skipping", key_str, keys)
                continue
        
        if key:
            if isinstance(vals, str):
                # split delimited strings
                vals_split = vals.split(sep_vals)
                if len(vals_split) > 1:
                    vals = vals_split
                # cast to numeric type if possible
                vals = libmag.get_int(vals)
            # assign to found enum to data class
            if is_data:
                setattr(out, key, vals)
            else:
                out[key] = vals
    
    return out


def _get_args_dict_help(
        msg: str, keys: Union[Type[Enum], Type["config.DataClass"]]) -> str:
    """Get the help message for command-line arguments that are converted
    to a dictionary.
    
    Args:
        msg: Message to prepend to the help message.
        keys: Keys as an Enumeration.

    Returns:
        Help message with available keys.

    """
    if dataclasses.is_dataclass(keys):
        # get all fields from data class
        names = [f.name for f in dataclasses.fields(keys)]
    else:
        # use enum names
        names = libmag.enum_names_aslist(keys)
    
    return (f"{msg} Available keys, which follow this order positionally "
            f"until the first key=value pair is given: {names}")


def process_cli_args():
    """Parse command-line arguments.
    
    Typically stores values as :mod:`magmap.settings.config` attributes.
    
    """
    parser = argparse.ArgumentParser(
        description="Setup environment for MagellanMapper")
    parser.add_argument(
        "--version", action="store_true",
        help="Show version information and exit")

    # image specification arguments
    
    # image path(s) specified as an optional argument; takes precedence
    # over positional argument
    parser.add_argument(
        "--img", nargs="*", default=None,
        help="Main image path(s); after import, the filename is often "
             "given as the original name without its extension")
    # alternatively specified as the first and only positional parameter
    # with as many arguments as desired
    parser.add_argument(
        "img_paths", nargs="*", default=None,
        help="Main image path(s); can also be given as --img, which takes "
             "precedence over this argument")
    
    parser.add_argument(
        "--meta", nargs="*",
        help="Metadata path(s), which can be given as multiple files "
             "corresponding to each image")
    parser.add_argument(
        "--prefix", nargs="*", type=str,
        help="Path prefix(es), typically used as the base path for file output")
    parser.add_argument(
        "--prefix_out", nargs="*", type=str,
        help="Path prefix(es), typically used as the base path for file output "
             "when --prefix modifies the input path")
    parser.add_argument(
        "--suffix", nargs="*", type=str,
        help="Path suffix(es), typically inserted just before the extension")
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
        "--proc", nargs="*",
        help=_get_args_dict_help(
            "Image processing mode; see config.ProcessTypes for keys "
            "and config.PreProcessKeys for PREPROCESS values",
            config.ProcessTypes))
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
        "--grid_search",
        help="Grid search hyperparameter tuning profile(s), which can be "
             "separated by underscores for multiple profiles and given as "
             "paths to custom profiles in YAML format. See docs/settings.md "
             "for more details.")
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
    parser.add_argument(
        "--classifier", nargs="*",
        help=_get_args_dict_help(
            "Classifier values; see config.ClassifierKeys for settings.",
            config.ClassifierData))

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
    parser.add_argument(
        "--rgb", action="store_true",
        help="Open images as RGB(A) color images")
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
        "-v", "--verbose", nargs="*",
        help=_get_args_dict_help(
            "Verbose output to assist with debugging; see config.Verbosity.",
            config.Verbosity))
    
    # only parse recognized arguments to avoid error for unrecognized ones
    args, args_unknown = parser.parse_known_args()

    # set up application directories
    user_dir = config.user_app_dirs.user_data_dir
    if not os.path.isdir(user_dir):
        # make application data directory
        if os.path.exists(user_dir):
            # backup any non-directory file
            libmag.backup_file(user_dir)
        os.makedirs(user_dir)

    if args.verbose is not None:
        # verbose mode and logging setup
        config.verbose = True
        config.verbosity = args_to_dict(
            args.verbose, config.Verbosity, config.verbosity)
        if config.verbosity[config.Verbosity.LEVEL] is None:
            # default to debug mode if any verbose flag is set without level
            config.verbosity[config.Verbosity.LEVEL] = logging.DEBUG
        logs.update_log_level(
            config.logger, config.verbosity[config.Verbosity.LEVEL])
        
        # print longer Numpy arrays for debugging
        np.set_printoptions(linewidth=200, threshold=10000)
        _logger.info("Set verbose to %s", config.verbosity)
    
    # set up logging to given file unless explicitly given an empty string
    log_path = config.verbosity[config.Verbosity.LOG_PATH]
    if log_path != "":
        if log_path is None:
            log_path = os.path.join(
                config.user_app_dirs.user_data_dir, "out.log")
        # log to file
        config.log_path = logs.add_file_handler(config.logger, log_path)
    
    # redirect standard out/error to logging
    sys.stdout = logs.LogWriter(config.logger.info)
    sys.stderr = logs.LogWriter(config.logger.error)
    
    # load preferences file
    config.prefs = prefs_prof.PrefsProfile()
    config.prefs.add_profiles(str(config.PREFS_PATH))
    
    if args.version:
        # print version info and exit
        _logger.info(f"{config.APP_NAME}-{libmag.get_version(True)}")
        shutdown()

    # log the app launch path
    path_launch = (sys._MEIPASS if getattr(sys, "frozen", False)
                   and hasattr(sys, "_MEIPASS") else sys.argv[0])
    _logger.info(f"Launched MagellanMapper from {path_launch}")
    
    if args.img is not None or args.img_paths:
        # set image file path and convert to basis for additional paths
        config.filenames = args.img if args.img else args.img_paths
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
    
    config.series_list = [config.series]  # list of series
    if args.series is not None:
        series_split = args.series.split(",")
        config.series_list = []
        for ser in series_split:
            ser_split = ser.split("-")
            if len(ser_split) > 1:
                ser_range = np.arange(int(ser_split[0]), int(ser_split[1]) + 1)
                config.series_list.extend(ser_range.tolist())
            else:
                config.series_list.append(int(ser_split[0]))
        config.series = config.series_list[0]
        print("Set to series_list to {}, current series {}".format(
              config.series_list, config.series))

    if args.savefig is not None:
        # save figure with file type of this extension; remove leading period
        config.savefig = _parse_none(args.savefig.lstrip("."))
        print("Set savefig extension to {}".format(config.savefig))

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
        config.cpus = _parse_none(args.cpus.lower(), int)
        print("Set maximum number of CPUs for multiprocessing tasks to",
              config.cpus)

    if args.load is not None:
        # flag loading data sources with default sub-arg indicating that the
        # data should be loaded from a default path; otherwise, load from
        # path given by the sub-arg; change delimiter to allow paths with ","
        config.load_data = args_to_dict(
            args.load, config.LoadData, config.load_data, sep_vals="|",
            default=True)
        print("Set to load the data types: {}".format(config.load_data))

    # set up main processing mode
    if args.proc is not None:
        config.proc_type = args_to_dict(
            args.proc, config.ProcessTypes, config.proc_type, default=True)
        print("Set main processing tasks to:", config.proc_type)

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
    setup_roi_profiles(args.roi_profile)
    setup_atlas_profiles(args.atlas_profile)
    setup_grid_search_profiles(args.grid_search)

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
        config.slice_vals = [_parse_none(val.lower(), int)
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
    
    if args.prefix is not None:
        # path input/output prefixes
        config.prefixes = args.prefix
        config.prefix = config.prefixes[0]
        print("Set path prefixes to {}".format(config.prefixes))
    
    if args.prefix_out is not None:
        # path output prefixes
        config.prefixes_out = args.prefix_out
        config.prefix_out = config.prefixes_out[0]
        print("Set path prefixes to {}".format(config.prefixes_out))
    
    if args.suffix is not None:
        # path suffixes
        config.suffixes = args.suffix
        config.suffix = config.suffixes[0]
        print("Set path suffixes to {}".format(config.suffixes))
    
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
    
    if args.rgb:
        # flag to open images as RGB
        config.rgb = args.rgb
        _logger.info("Set RGB to %s", config.rgb)
    
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
    
    if args.classifier is not None:
        # classifier settings
        args_to_dict(args.classifier, config.classifier)
        print("Set classifier to {}".format(config.classifier))

    if args.db:
        # set main database path to user arg
        config.db_path = args.db
        print("Set database name to {}".format(config.db_path))
    else:
        # set default path
        config.db_path = os.path.join(user_dir, config.db_path)

    if args.truth_db:
        # set settings for separate database of "truth blobs"
        config.truth_db_params = args_to_dict(
            args.truth_db, config.TruthDB, config.truth_db_params,
            sep_vals="|")
        mode = config.truth_db_params[config.TruthDB.MODE]
        config.truth_db_mode = libmag.get_enum(mode, config.TruthDBModes)
        libmag.printv(config.truth_db_params)
        print("Mapped \"{}\" truth_db mode to {}"
              .format(mode, config.truth_db_mode))
    
    # notify user of full args list, including unrecognized args
    _logger.debug(f"All command-line arguments: {sys.argv}")
    if args_unknown:
        _logger.info(
            f"The following command-line arguments were unrecognized and "
            f"ignored: {args_unknown}")


def setup_image(
        path: str, series: Optional[int] = None,
        proc_tasks: Optional[Dict["config.ProcessTypes", Any]] = None):
    """Set up the main image from CLI args and process any tasks.
    
    Args:
        path: Image path.
        series: Image series, such as a tile; defaults to None.
        proc_tasks: Dictionary of processing tasks; defaults to None.

    """
    # deconstruct user-supplied image filename
    filename, offset, size, reg_suffixes = importer.deconstruct_img_name(path)
    set_subimg, _ = importer.parse_deconstructed_name(
        filename, offset, size, reg_suffixes)
    
    if not set_subimg:
        # sub-image parameters set in filename takes precedence for
        # the loaded image, but fall back to user-supplied args
        offset = config.subimg_offsets[0] if config.subimg_offsets else None
        size = config.subimg_sizes[0] if config.subimg_sizes else None
    
    if proc_tasks:
        for proc_task, proc_val in proc_tasks.items():
            # set up image for the given task
            np_io.setup_images(
                filename, series, offset, size, proc_task,
                fallback_main_img=False)
            process_file(
                filename, proc_task, proc_val, series, offset, size,
                config.roi_offsets[0] if config.roi_offsets else None,
                config.roi_sizes[0] if config.roi_sizes else None)
        
    else:
        # set up image without a task specified, eg for display
        np_io.setup_images(filename, series, offset, size)
    

def process_proc_tasks(
        path: Optional[str] = None,
        series_list: Optional[Sequence[int]] = None
) -> Optional[Dict["config.ProcessTypes", Any]]:
    """Apply processing tasks.
    
    Args:
        path: Base path to main image file; defaults to None, in which case
            :attr:`config.filename` will be used.
        series_list: Sequence of images series, such as tiles; defaults
            to None.

    Returns:
        Dictionary of set processing types.

    """
    if path is None:
        path = config.filename
    if not path:
        print("No image filename set for processing files, skipping")
        return None
    if series_list is None:
        series_list = config.series_list
    
    # filter out unset tasks
    proc_tasks = {k: v for k, v in config.proc_type.items() if v}
    for series in series_list:
        # process files for each series, typically a tile within a
        # microscopy image set or a single whole image
        setup_image(path, series, proc_tasks)
    
    return proc_tasks
    

def process_tasks():
    """Process command-line tasks.
    
    Perform tasks set by the ``--proc`` parameter or any other entry point,
    such as ``--register`` tasks. Only the first identified task will be
    performed.

    """
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
        _grid_search(config.series_list)
    elif config.ec2_list or config.ec2_start or config.ec2_terminate:
        # defer importing AWS module to avoid making its dependencies
        # required for MagellanMapper
        from magmap.cloud import aws
        aws.main()
    else:
        # processing tasks
        proc_tasks = process_proc_tasks()
        if not proc_tasks or config.ProcessTypes.LOAD in proc_tasks:
            # do not shut down since not a command-line task or if loading files
            return
    shutdown()


def setup_dbs():
    """Set up databases for the given image file.
    
    Only sets up each database if it has not been set up already.
    
    """
    # prep filename
    filename_base = None
    if config.filename:
        filename_base = importer.filename_to_base(
            config.filename, config.series)
    
    # get any user-supplied truth database path, falling back to name based
    # on filename or default name
    truth_db_path = config.truth_db_params[config.TruthDB.PATH]
    user_dir = config.user_app_dirs.user_data_dir
    truth_db_name_base = filename_base if filename_base else os.path.join(
        user_dir, sqlite.DB_NAME_BASE)
    
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
            config.verified_db.load_db(
                os.path.join(user_dir, sqlite.DB_NAME_VERIFIED), True)
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
        path = os.path.join(user_dir, sqlite.DB_NAME_VERIFIED)
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
        config.db_path = truth_db_path
        if not config.db_path: 
            config.db_path = "{}{}".format(
                os.path.basename(truth_db_name_base), sqlite.DB_SUFFIX_TRUTH)
        print("Editing truth database at {}".format(config.db_path))
    
    if config.db is None:
        # load the main database
        config.db = sqlite.ClrDB()
        config.db.load_db(None, False)


def main(process_args_only: bool = False, skip_dbs: bool = False):
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    
    Args:
        process_args_only: Processes command-line arguments and
            returns; defaults to False.
        skip_dbs: True to skip loading databases; defaults to False.
    """
    # parse command-line arguments
    process_cli_args()
    
    if not skip_dbs:
        # load databases
        setup_dbs()
    
    # set multiprocessing start method
    chunking.set_mp_start_method()
    
    if process_args_only:
        return
    # process tasks
    process_tasks()


def setup_roi_profiles(roi_profiles_names: List[str]):
    """Set up ROI profiles.

    If a profile is None, only a default set of profile settings
    will be generated. Also sets up colormaps based on ROI profiles. Any
    previously set up profile will be replaced.

    Args:
        roi_profiles_names: Sequence of ROI and atlas profiles
            names to use for the corresponding channel.

    """
    # initialize ROI profile settings and update with modifiers
    config.roi_profile = roi_prof.ROIProfile()
    config.roi_profiles = [config.roi_profile]
    if roi_profiles_names is not None:
        for i, roi_prof_name in enumerate(roi_profiles_names):
            _logger.debug("Updating ROI profile for channel %s", i)
            if i == 0:
                settings = config.roi_profile
            else:
                settings = roi_prof.ROIProfile()
                config.roi_profiles.append(settings)
            settings.add_profiles(roi_prof_name)
    for i, prof in enumerate(config.roi_profiles):
        if i == 0:
            _logger.info(
                "Set default (channel 0) ROI profile: %s", prof[prof.NAME_KEY])
        else:
            _logger.info(
                "Added channel %s ROI profile: %s", i, prof[prof.NAME_KEY])
    colormaps.setup_colormaps(np_io.get_num_channels(config.image5d))


def setup_atlas_profiles(atlas_profiles_names: str, reset: bool = True):
    """Set up atlas profiles.

    If a profile is None, only a default set of profile settings
    will be generated. Any previously set up profile will be replaced.

    Args:
        atlas_profiles_names: Atlas profiles names.
        reset: True to reset profiles before setting profiles from
            ``atlas_profile_names``; defaults to True.

    """
    # initialize atlas profile and update with modifiers
    if reset:
        config.atlas_profile = atlas_prof.AtlasProfile()
    if atlas_profiles_names is not None:
        config.atlas_profile.add_profiles(atlas_profiles_names)
    _logger.info(
        "Set atlas profile to %s",
        config.atlas_profile[config.atlas_profile.NAME_KEY])


def setup_grid_search_profiles(grid_search_profiles_names: str):
    """Setup grid search profiles.

    If a profile is None, only a default set of profile settings
    will be generated. Any previously set up profile will be replaced.

    Args:
        grid_search_profiles_names: Grid search profiles names.

    """
    if grid_search_profiles_names:
        # parse grid search profiles
        config.grid_search_profile = grid_search_prof.GridSearchProfile()
        config.grid_search_profile.add_profiles(grid_search_profiles_names)
        _logger.info(
            "Set grid search profile to %s",
            config.grid_search_profile[config.grid_search_profile.NAME_KEY])
        _logger.debug(config.grid_search_profile)


def update_profiles():
    """Update profiles if any profile file has been modified since it
    was last loaded.

    Profiles in both :attr:`config.process_settings_list` and
    :attr:`config.register_settings_list` will be checked to update.

    """
    for i, prof in enumerate(config.roi_profiles):
        prof.refresh_profile(True)
    config.atlas_profile.refresh_profile(True)


def setup_labels(labels_arg: List[str]):
    """Set up atlas labels.
    
    Args:
        labels_arg: Path to labels reference file, such as a labels
            ontology file.

    """
    # atlas labels as positional or dictionary-like args
    config.atlas_labels = args_to_dict(
        labels_arg, config.AtlasLabels, config.atlas_labels)
    config.load_labels = config.atlas_labels[config.AtlasLabels.PATH_REF]
    config.labels_level = config.atlas_labels[config.AtlasLabels.LEVEL]
    print("Set labels to {}".format(config.atlas_labels))


def _detect_subimgs(
        path: str, series: int, subimg_offsets: List[List[int]],
        subimg_sizes: List[List[int]]
) -> Tuple[Union[np.ndarray, Any], List[str]]:
    """Detect blobs in an image across sub-image offsets.
    
    Args:
        path: Path to image from which MagellanMapper-style paths will 
            be generated.
        series: Image series number.
        subimg_offsets: Nested list of sub-image offset sets
            given as ``[[offset_z1, offset_y1, offset_x1], ...]``.
        subimg_sizes: Nested list of sub-image size sets
            given as ``[[offset_z1, offset_y1, offset_x1], ...]`` and
            corresponding to ``subimg_offsets``.
    
    Returns:
        Summed stats array and concatenated summaries.
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
        np_io.setup_images(path, series, subimg_offsets[i], size)
        stat_roi, fdbk, _ = stack_detect.detect_blobs_stack(
            importer.filename_to_base(path, series), subimg_offsets[i], size)
        if stat_roi is not None:
            stat = np.add(stat, stat_roi)
        summaries.append(
            "Offset {}:\n{}".format(subimg_offsets[i], fdbk))
    return stat, summaries


def _grid_search(series_list: List[int]):
    # grid search(es) for the specified hyperparameter groups
    if not config.filename:
        print("No image filename set for grid search, skipping")
        return
    for series in series_list:
        # process each series, typically a tile within an microscopy image
        # set or a single whole image
        stats_dict = mlearn.grid_search(
            config.grid_search_profile.hyperparams, _detect_subimgs,
            config.filename, series, config.subimg_offsets,
            config.subimg_sizes)
        parsed_dict, stats_df = mlearn.parse_grid_stats(stats_dict)
        
        # plot ROC curve
        plot_2d.plot_roc(stats_df, config.show)


def process_file(
        path: str, proc_type: Enum, proc_val: Optional[Any] = None,
        series: Optional[int] = None,
        subimg_offset: Optional[List[int]] = None,
        subimg_size: Optional[List[int]] = None,
        roi_offset: Optional[List[int]] = None,
        roi_size: Optional[List[int]] = None
) -> Tuple[Optional[Any], Optional[str]]:
    """Processes a single image file non-interactively.

    Assumes that the image has already been set up.
    
    Args:
        path: Path to image from which MagellanMapper-style paths will
            be generated.
        proc_type: Processing type, which should be a one of
            :class:`config.ProcessTypes`.
        proc_val: Processing value associated with ``proc_type``; defaults to
            None.
        series: Image series number; defaults to None.
        subimg_offset: Sub-image offset as (z,y,x) to load; defaults to None.
        subimg_size: Sub-image size as (z,y,x) to load; defaults to None.
        roi_offset: Region of interest offset as (x, y, z) to process;
            defaults to None.
        roi_size: Region of interest size of region to process, given as
            ``(x, y, z)``; defaults to None.
    
    Returns:
        Tuple of stats from processing, or None if no stats, and
        text feedback from the processing, or None if no feedback.
    
    """
    # PROCESS BY TYPE
    stats = None
    fdbk = None
    filename_base = importer.filename_to_base(path, series)
    
    print("{}\n".format("-" * 80))
    if proc_type is config.ProcessTypes.LOAD:
        # loading completed
        return None, None

    elif proc_type is config.ProcessTypes.LOAD:
        # already imported so does nothing
        print("imported {}, will exit".format(path))
    
    elif proc_type is config.ProcessTypes.EXPORT_ROIS:
        # export ROIs; assumes that metadata was already loaded to give smaller
        # region from which smaller ROIs from the truth DB will be extracted
        from magmap.io import export_rois
        db = config.db if config.truth_db is None else config.truth_db
        export_path = naming.make_subimage_name(
            filename_base, subimg_offset, subimg_size)
        export_rois.export_rois(
            db, config.img5d, config.channel, export_path,
            config.plot_labels[config.PlotLabels.PADDING],
            config.unit_factor, config.truth_db_mode,
            os.path.basename(export_path))
        
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
        stats, fdbk, _ = stack_detect.detect_blobs_stack(
            filename_base, subimg_offset, subimg_size, coloc)

    elif proc_type is config.ProcessTypes.COLOC_MATCH:
        if config.blobs is not None and config.blobs.blobs is not None:
            # colocalize blobs in separate channels by matching blobs
            shape = subimg_size
            if shape is None:
                # get shape from loaded image, falling back to its metadata
                if config.image5d is not None:
                    shape = config.image5d.shape[1:]
                else:
                    shape = config.img5d.meta[config.MetaKeys.SHAPE][1:]
            matches = colocalizer.StackColocalizer.colocalize_stack(
                shape, config.blobs, config.channel)
            # insert matches into database
            colocalizer.insert_matches(config.db, matches)
        else:
            print("No blobs loaded to colocalize, skipping")
    
    elif proc_type is config.ProcessTypes.CLASSIFY:
        # classify blobs
        try:
            classifier.ClassifyImage.classify_whole_image()
            config.blobs.save_archive()
        except FileNotFoundError as e:
            _logger.debug(e)

    elif proc_type in (config.ProcessTypes.EXPORT_PLANES,
                       config.ProcessTypes.EXPORT_PLANES_CHANNELS):
        # export each plane as a separate image file
        export_stack.export_planes(
            config.image5d, config.savefig, config.channel,
            proc_type is config.ProcessTypes.EXPORT_PLANES_CHANNELS)
    
    elif proc_type is config.ProcessTypes.EXPORT_RAW:
        # export the main image as a raw data file
        out_path = libmag.combine_paths(config.filename, ".raw", sep="")
        libmag.backup_file(out_path)
        np_io.write_raw_file(config.image5d, out_path)

    elif proc_type is config.ProcessTypes.EXPORT_TIF:
        # export the main image as a TIF files for each channel
        np_io.write_tif(config.image5d, config.filename)

    elif proc_type is config.ProcessTypes.PREPROCESS:
        # pre-process a whole image and save to file
        # TODO: consider chunking option for larger images
        out_path = config.prefix
        if not out_path:
            out_path = libmag.insert_before_ext(config.filename, "_preproc")
        transformer.preprocess_img(
            config.image5d, proc_val, config.channel, out_path)

    return stats, fdbk


def shutdown():
    """Clean up and shutdown MagellanMapper.

    Stops any running Java virtual machine and closes any main database.
    """
    importer.stop_jvm()
    if config.db is not None:
        config.db.conn.close()
    if config.prefs is not None:
        config.prefs.save_settings(config.PREFS_PATH)
    sys.exit()

    
if __name__ == "__main__":
    _logger.info("Starting MagellanMapper command-line interface...")
    main()
