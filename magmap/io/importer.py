# Image stack importer
# Author: David Young, 2017, 2020
"""Imports image stacks using Bioformats.

Bioformats is access through Python-Bioformats and Javabridge.
Images will be imported into a 4/5D Numpy array.

Attributes:
    PIXEL_DTYPE: Dictionary of corresponding data types for 
        output given by Bioformats library. Alternatively, should detect
        pixel data type directly using parse_ome_raw().
    IMAGE5D_NP_VER: image5d Numpy saved array version number, which should
        be incremented with any change to the image5d or its support "info"
        save array format.
"""

from collections import OrderedDict
import os
from time import time
import glob
import re
from xml import etree as et

import numpy as np
from PIL import Image
from skimage import color
from skimage import io

from magmap.io import libmag, np_io, yaml_io
from magmap.plot import plot_3d
from magmap.settings import config

_logger = config.logger.getChild(__name__)

try:
    import javabridge as jb
    import bioformats as bf
except (ImportError, ValueError, RuntimeError) as e:
    # Javabridge gives a JVMNotFoundError that extends ValueError if
    # Java cannot be initialized, or a RuntimeError if Java home dir not found 
    jb = None
    bf = None
    _logger.warn(
        "%s could not be found, so there will be error when attempting to "
        "import images into Numpy format",
        e.name if isinstance(e, ImportError) else "Java")

# pixel type enumeration based on:
# http://downloads.openmicroscopy.org/bio-formats-cpp/5.1.8/api/classome_1_1xml_1_1model_1_1enums_1_1PixelType.html
# http://downloads.openmicroscopy.org/bio-formats-cpp/5.1.8/api/PixelType_8h_source.html
PIXEL_DTYPE = {
    0: np.int8,
    1: np.uint8,
    2: np.int16,
    3: np.uint16,
    4: np.int32,
    5: np.uint32,
    6: np.float32,
    7: np.double
}

# image5d info archive versions:
# 10: started at 10 because of previous versions prior to numbering; 
#     fixed saved resolutions to contain only the given series
# 11: sizes uses the image5d shape rather than the original image's size
# 12: fixed replacing dtype, near_min/max when saving image in transpose_npy
# 13: change near_min/max to array with element for each channel; add 
#     "scaling" and "plane" fields for transposed images
# 14: removed pixel_type since redundant with image5d.dtype; 
#     avoids storing object array, which requires loading by pickling
# 15: file format changed from NPZ to YAML
IMAGE5D_NP_VER = 15  # image5d Numpy saved array version number

#: str: String preceding channel number for multi-channel image import.
CHANNEL_SEPARATOR = "_ch_"

#: str: Default filename base for directory import image output.
DEFAULT_IMG_STACK_NAME = "myvolume"

_KEY_ANY_CHANNEL = "1+"  # 1+ channel files

_logger = config.logger.getChild(__name__)


def is_javabridge_loaded():
    """Check if Javabridge and Python-Bioformats have been loaded.
    
    Returns:
        bool: True if the modules have both been loaded, False otherwise.

    """
    if jb is None or bf is None:
        libmag.warn(
            "Python-Bioformats or Python-Javabridge not available, "
            "multi-page images cannot be imported")
        return False
    return True


def start_jvm(heap_size="8G"):
    """Starts the JVM for Python-Bioformats.
    
    Can only start Javabridge once per session. Calling this function
    repeatedly without stopping the JVM will have no effect, however.
    To use the JVM in differ threads, use :meth:`jb.attach` and
    :meth:`jb.detach`.
    
    Args:
        heap_size (str): JVM heap size, defaulting to 8G.
    """
    _logger.info(f"Starting Java for Bioformats using JAVA_HOME set to: "
                 f"{os.getenv('JAVA_HOME')}")
    if not jb:
        libmag.warn("Python-Javabridge not available, cannot start JVM")
        return
    jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)


def stop_jvm():
    """Stop the Javabridge JVM.
    
    Javabridge should only be started/stopped once per session.

    """
    if not jb:
        libmag.warn("Python-Javabridge not available, cannot stop JVM")
        return
    jb.kill_vm()


def parse_ome(filename):
    """Parses metadata for image name and size information using Bioformats'
    OME XML wrapper.
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
    
    Returns:
        names: array of names of seriess within the file.
        sizes: array of tuples with dimensions for each series. Dimensions
            will be given as (time, z, y, x, channels).
    
    Deprecated: 1.6.0
        Use :meth:`parse_ome_raw` instead.
    """
    time_start = time()
    metadata = bf.get_omexml_metadata(filename)
    ome = bf.OMEXML(metadata)
    count = ome.image_count
    names, sizes = [], []
    for i in range(count):
        image = ome.image(i)
        names.append(image.Name)
        pixel = image.Pixels
        size = (pixel.SizeT, pixel.SizeZ, pixel.SizeY, pixel.SizeX, pixel.SizeC)
        sizes.append(size)
    print("names: {}\nsizes: {}".format(names, sizes))
    print('time for parsing OME XML: %f' %(time() - time_start))
    return names, sizes


def parse_ome_raw(metadata: str):
    """Parse Open Microscopy Environment XML to extract key metadata.
    
    Args:
        metadata: Metadata as a string in OME XML format.
    
    Returns:
        names: array of names of seriess within the file.
        sizes: array of tuples with dimensions for each series. Dimensions
            will be given as (time, z, y, x, channels).
        resolution: array of resolutions, also known as scaling, in the same
            dimensions as sizes.
        magnification: Objective magnification.
        zoom: Zoom level.
        pixel_type: Pixel data type as a string.
    
    """
    array_order = "TZYXC"  # desired dimension order
    names, sizes, resolutions = [], [], []
    # names for sizes in all dimensions
    size_tags = ["Size" + c for c in array_order]
    # names for resolutions only in XYZ dimensions
    spatial_array_order = [c for c in array_order if c in "XYZ"]
    res_tags = ["PhysicalSize" + c for c in spatial_array_order]
    zoom = 1
    magnification = 1
    pixel_type = None
    metadata_root = et.ElementTree.fromstring(metadata)
    for child in metadata_root:
        # tag name is at end of a long string of other identifying info
        print("tag: {}".format(child.tag))
        if child.tag.endswith("Instrument"):
            # microscope info
            for grandchild in child:
                if grandchild.tag.endswith("Detector"):
                    zoom = grandchild.attrib.get("Zoom")
                    if zoom is not None:
                        zoom = float(zoom)
                elif grandchild.tag.endswith("Objective"):
                    magnification = grandchild.attrib["NominalMagnification"]
                    if magnification is not None:
                        magnification = float(magnification)
        elif child.tag.endswith("Image"):
            # image file info
            if "Name" in child.attrib:
                names.append(child.attrib["Name"])
            for grandchild in child:
                if grandchild.tag.endswith("Pixels"):
                    att = grandchild.attrib
                    # get image shape for the series
                    sizes.append(tuple(
                        [int(att[t]) if t in att else 1 for t in size_tags]))
                    # get image resolutions for the series
                    resolutions.append(tuple(
                        [float(att[t]) if t in att else 1.0 for t in res_tags]))
                    # assumes pixel type is same for all images
                    if pixel_type is None:
                        pixel_type = att.get("Type")
    print("names: {}".format(names))
    print("sizes: {}".format(sizes))
    print("resolutions: {}".format(resolutions))
    print("zoom: {}, magnification: {}".format(zoom, magnification))
    print("pixel_type: {}".format(pixel_type))
    return names, sizes, resolutions, magnification, zoom, pixel_type


def find_sizes(filename):
    """Finds image size information using the ImageReader using Bioformats'
    wrapper to access a small subset of image properities.
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
    
    Returns:
        sizes: array of tuples with dimensions for each series. Dimensions
            will be given as (time, z, y, x, channels).
    """
    time_start = time()
    sizes = []
    with bf.ImageReader(filename) as rdr:
        format_reader = rdr.rdr
        count = format_reader.getSeriesCount()
        for i in range(count):
            format_reader.setSeries(i)
            size = (format_reader.getSizeT(), format_reader.getSizeZ(), 
                    format_reader.getSizeY(), format_reader.getSizeX(), 
                    format_reader.getSizeC())
            print(size)
            sizes.append(size)
        pixel_type = format_reader.getPixelType()
        dtype = PIXEL_DTYPE[pixel_type]
        print("pixel type: {}, dtype: {}".format(pixel_type, dtype))
    print("time for finding sizes: ", time() - time_start)
    return sizes, dtype


def make_filenames(filename, series=None, modifier=""):
    """Make MagellanMapper-oriented image and image metadata filenames.
    
    Args:
        filename (str): Original path from which MagellanMapper-oriented
            filenames  will be derived.
        series (int): Image series; defaults to None.
        modifier (str): Separator for image series; defaults to an empty string.
    
    Returns:
        Tuple of path to the main image and path to metadata.
    """
    print("filename: {}".format(filename))
    filename_base = filename_to_base(filename, series, modifier)
    print("filename_base: {}".format(filename_base))
    filename_image5d = libmag.combine_paths(
        filename_base, config.SUFFIX_IMAGE5D)
    filename_meta = libmag.combine_paths(filename_base, config.SUFFIX_META)
    return filename_image5d, filename_meta


def filename_to_base(filename, series=None, modifier=""):
    """Convert an image path to a base path with an optional modifier.
    
    Args:
        filename (str): Path to original image.
        series (int): Series (eg tile) within image; defaults to None.
            Currently ignored, but may be implemented in the future to
            track different tiles or timepoints.
        modifier (str): Modifier string prior to series; defaults to an
            empty string.
    """
    path = libmag.splitext(filename)[0]
    if modifier:
        path = libmag.combine_paths(path, modifier)
    return path


def deconstruct_img_name(np_filename, sep="_", keep_subimg=False):
    """Deconstruct Numpy or registered image filename to the original name
    from which it was based.

    Also parses sub-image offset and shape information if the filename
    is in sub-image format as generated by
    :meth:`magmap.cv.stack_detect.make_subimg_name`.
    
    Args:
        np_filename (str): Numpy image filename.
        sep (str): Separator for any registration suffixes; defaults to "_".
        keep_subimg (bool): True to keep the sub-image part of the name.
    
    Returns:
        str, str, str, dict[str, str]: ``filename``, the deconstructed
        filename, without suffixes and ending in a period. If no matching
        suffix is found, simply returns ``np_filename``.
        ``offset``, the sub-image offset, of None if not a sub-image.
        ``size``, the sub-image shape, or None if not a sub-image.
        ``reg_suffixes``, a dictionary of :class:`config.RegSuffixes` to
        :class:`config.RegNames` values if ``np_filename`` is a registered
        image path. The registered suffix in this path is assigned to the
        :obj:`config.RegSuffixes.ATLAS` suffix.
    
    """
    base_path = None
    offset = None
    size = None
    reg_suffixes = None
    if np_filename is None:
        return base_path, offset, size, reg_suffixes
    len_np_filename = len(np_filename)
    for suffix in (config.SUFFIX_IMAGE5D, config.SUFFIX_META,
                   config.SUFFIX_SUBIMG):
        # identify whether path is to a Numpy image file
        suffix_full = "_{}".format(suffix)
        if np_filename.endswith(suffix_full):
            # strip suffix to get base path
            filename = np_filename[:len_np_filename-len(suffix_full)]
            if suffix is config.SUFFIX_SUBIMG:
                # extract sub-image offset and shape
                subimg = filename[filename.rindex("_"):]
                regex_coords = re.compile(r"\([0-9]*,[0-9]*,[0-9]*\)")
                coords = re.findall(regex_coords, subimg)
                if len(coords) >= 2:
                    # convert each match to tuple of ints and reverse order
                    # from x,y,z to z,y,x
                    coords = [c.strip("()").split(",")[::-1] for c in coords]
                    coords = [tuple(int(s) for s in c) for c in coords]
                    offset, size = coords[:2]
                    if not keep_subimg:
                        filename = filename[:len(filename)-len(subimg)]
            base_path = "{}.".format(filename)
            break
    if base_path is None:
        for suffix in config.RegNames:
            # identify whether path is to a registered image file by checking
            # for an extension-less suffix just before the filename ext
            suffix = suffix.value
            suffix_noext = libmag.get_filename_without_ext(suffix)
            suffixi = np_filename.rfind(suffix_noext)
            if suffixi != -1 and libmag.splitext(np_filename)[0].endswith(
                    suffix_noext):
                # strip suffix and any ending separator to get base path
                filename = np_filename[:suffixi]
                base_path = (filename[:-1] if filename.endswith(sep)
                             else filename)
                reg_suffixes = {config.RegSuffixes.ATLAS: suffix}
                break
    if base_path is None:
        # default to returning path as-is
        base_path = np_filename
    return base_path, offset, size, reg_suffixes


def parse_deconstructed_name(filename, offset, size, reg_suffixes):
    """Parse deconstructed image name into :module:`config` settings.
    
    Args:
        filename (str): Deconstructed image path.
        offset (tuple[int, int, int]): Deconstructed sub-image offset.
        size (tuple[int, int, int]): Deconstructed sub-image size.
        reg_suffixes (dict): Registered image suffixes.

    Returns:
        bool, bool: True if the sub-image parameters were set, True if the
        registered suffixes were set.

    """
    config.filename = filename
    _logger.debug("Changed filename to %s", config.filename)
    set_subimg = offset is not None and size is not None
    if set_subimg:
        config.subimg_offsets = [offset]
        config.subimg_sizes = [size]
        _logger.debug("Change sub-image offset to {}, size to {}"
                      .format(config.subimg_offsets, config.subimg_sizes))
    # TODO: consider loading processed images, blobs, etc
    set_reg_suffixes = False
    if reg_suffixes:
        config.reg_suffixes.update(reg_suffixes)
        set_reg_suffixes = True
        _logger.debug("Update registered image suffixes to:", reg_suffixes)
    return set_subimg, set_reg_suffixes


def save_image_info(filename_info_npz, names, sizes, resolutions, 
                    magnification, zoom, near_min, near_max, 
                    scaling=None, plane=None):
    """Save image metadata to YAML file format.
    
    Args:
        filename_info_npz (str): Output path.
        names (Sequence): Sequence of names for each series.
        sizes (Sequence): Sequence of sizes for each series.
        resolutions (Sequence): Sequence of resolutions for each series.
        magnification (float): Objective magnification.
        zoom (float): Objective zoom.
        near_min (list[float]): Sequence of near minimum intensities, with
            each element in turn holding a sequence with values for each
            channel.
        near_max (list[float]): Sequence of near maximum intensities, with
            each element in turn holding a sequence with values for each
            channel.
        scaling (float): Rescaling value for up/downsampled images; defaults 
            to None.
        plane (str): Planar orientation compared with original for transposed 
            images; defaults to None.

    Returns:
        dict: The saved metadata as a dictionary.
    
    """
    data = {
        "ver": IMAGE5D_NP_VER,
        "names": names,
        "sizes": sizes,
        "resolutions": resolutions,
        "magnification": magnification,
        "zoom": zoom,
        "near_min": near_min,
        "near_max": near_max,
        "scaling": scaling,
        "plane": plane,
    }
    yaml_io.save_yaml(filename_info_npz, data, True)
    return data


def _update_image5d_np_ver(curr_ver, image5d, info, filename_info_npz):
    # update image archive metadata using dictionary of values successfully 
    # loaded from the archive
    
    if curr_ver >= IMAGE5D_NP_VER:
        # no updates necessary
        return False
    
    print("Updating image metadata to version {}".format(IMAGE5D_NP_VER))
    print("Original metadata:\n{}".format(info))
    
    if curr_ver <= 10:
        # ver 10 -> 11
        # no change except ver since most likely won't encounter any difference
        pass
    
    if curr_ver <= 11:
        # ver 11 -> 12
        if info["pixel_type"] != image5d.dtype:
            # Numpy transpositions did not update pixel type and min/max
            info["pixel_type"] = image5d.dtype
            info["near_min"], info["near_max"] = np.percentile(
                image5d, (0.5, 99.5))
            print("updated pixel type to {}, near_min to {}, near_max to {}"
                  .format(info["pixel_type"], info["near_min"], 
                          info["near_max"]))
    
    if curr_ver <= 12:
        # ver 12 -> 13
        
        # default to simply converting the existing scalar to a one-element 
        # list of repeated existing value, assuming single-channel
        near_mins = [info["near_min"]]
        near_maxs = [info["near_max"]]
        scaling = None
        
        # assumed that 2nd filename given is the original file from which to 
        # calculate exact scaling
        if len(config.filenames) > 1:
            img5d = read_file(
                config.filenames[1], config.series, update_info=False)
            image5d_orig = img5d.img
            scaling = calc_scaling(image5d_orig, image5d)
            # image5d is a scaled, smaller image, so bounds will be 
            # calculated since the calculation requires loading full image 
            # into memory; otherwise, defer to re-importing the image
            lows, highs = calc_intensity_bounds(image5d)
        elif image5d.ndim >= 5:
            # recalculate near min/max for multichannel
            print("updating near min/max (this may take awhile)")
            lows = []
            highs = []
            for i in range(len(image5d[0])):
                low, high = calc_intensity_bounds(image5d[0, i], dim_channel=2)
                print("bounds for plane {}: {}, {}".format(i, low, high))
                lows.append(low)
                highs.append(high)
            near_mins, near_maxs = _calc_near_intensity_bounds(
                near_mins, near_maxs, lows, highs)
        info["near_min"] = near_mins
        info["near_max"] = near_maxs
        info["scaling"] = scaling
        info["plane"] = config.plane
    
    if curr_ver <= 13:
        # ver 13 -> 14
        
        # pixel_type no longer saved since redundant with image5d.dtype
        if "pixel_type" in info:
            del info["pixel_type"]

    # backup and save updated info
    print("Updating image5d metadata:\n", info)
    libmag.backup_file(
        filename_info_npz, modifier="_v{}".format(curr_ver))
    info["ver"] = IMAGE5D_NP_VER
    yaml_io.save_yaml(filename_info_npz, info, True)
    
    return True


def load_metadata(path, check_ver=False, assign=True):
    """Load image info, such as saved microscopy data and image ranges, 
    storing some values into appropriate module level variables.
    
    Args:
        path (str): Path to image info file.
        check_ver (bool): True to stop loading if the archive's version number  
            is less than :const:``IMAGE5D_NP_VER``; defaults to False.
        assign (bool): True to assign values to module-level settings.
    
    Returns:
        Tuple of ``output``, the dictionary with image info, and 
        ``image5d_ver_num``, the version number of the info file, 
        which is -1 if the key could not be found.
    """
    print("Reading image metadata from {}".format(path))
    image5d_ver_num = -1
    try:
        # load metadata in YAML format (v1.4+)
        output = yaml_io.load_yaml(path)
        if output:
            # metadata is in first document
            output = output[0]
    except FileNotFoundError as err:
        # fall back to pre-v1.4 NPZ file format
        _logger.warn("Could not load metadata file '%s', will check NPZ format",
                     path)
        _logger.debug(err)
        path_npz = f"{os.path.splitext(path)[0]}.npz"
        try:
            # load NPZ and resave as YML format
            output = np_io.read_np_archive(np.load(path_npz))
            path_yml = f"{os.path.splitext(path)[0]}.yml"
            yaml_io.save_yaml(path_yml, output, True)
            _logger.info(
                "Metadata file from '%s' updated to '%s'", path_npz, path_yml)
        except FileNotFoundError as err2:
            _logger.warn(
                "Could not load metadata file '%s', skipping", path_npz)
            _logger.debug(err2)
            return None, image5d_ver_num
    
    try:
        # find the info version number
        image5d_ver_num = output["ver"]
        print("loaded image5d version number {}".format(image5d_ver_num))
    except KeyError:
        print("could not find image5d version number")
    if assign and (not check_ver or image5d_ver_num >= IMAGE5D_NP_VER):
        # load into various module variables unless checking version 
        # and below current version to avoid errors during loading
        assign_metadata(output)
    return output, image5d_ver_num


def assign_metadata(md):
    """Assign values from a metadata dictionary to module variables. 
    
    Args:
        md (dict): Dictionary of metadata.

    """
    try:
        names = md["names"]
        print("names: {}".format(names))
    except KeyError:
        print("could not find names")
    try:
        config.image5d_shapes = md["sizes"]
        print("sizes {}".format(config.image5d_shapes))
    except KeyError:
        print("could not find sizes")
    try:
        config.resolutions = np.array(md["resolutions"])
        print("set resolutions to {}".format(config.resolutions))
    except KeyError:
        print("could not find resolutions")
    try:
        config.magnification = md["magnification"]
        print("magnification: {}".format(config.magnification))
    except KeyError:
        print("could not find magnification")
    try:
        config.zoom = md["zoom"]
        print("zoom: {}".format(config.zoom))
    except KeyError:
        print("could not find zoom")
    try:
        config.near_min = md["near_min"]
        print("set near_min to {}".format(config.near_min))
    except KeyError:
        print("could not find near_max")
    try:
        config.near_max = md["near_max"]
        print("set near_max to {}".format(config.near_max))
        if config.vmaxs is None:
            config.vmax_overview = np.multiply(config.near_max, 1.1)
        print("Set vmax_overview to {}".format(config.vmax_overview))
    except KeyError:
        print("could not find near_max")


def read_file(filename, series=None, offset=None, size=None, return_info=False,
              update_info=True):
    """Reads an image file in Numpy format.

    An offset and size can be given to load an only an ROI of the image.
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
        series: Series index to load. Defaults to None, which will use 0.
        offset: Tuple of offset given as (x, y, z) from which to start
            loading z-plane (x, y ignored for now). If Numpy image info already 
            exists, this tuple will be used to load only an ROI of the image. 
            If importing a new Numpy image, offset[2] will be used an a 
            z-offset from which to start importing. Defaults to None.
        size: Tuple of ROI size given as (x, y, z). If Numpy image info already 
            exists, this tuple will be used to load only an ROI of the image. 
            Defaults to None.
        return_info: True if the Numpy info file should be returned for a 
            dictionary of image properties; defaults to False.
        update_info: True if the associated image5d info file should be 
            updated; defaults to True.
    
    Returns:
        :obj:`np_io.Image5d`, dict: The 5D image object, or None if it could
        not be loaded. If ``return_info`` is True, a dictionary of image
        properties will also be returned.
    
    Raises:
        FileNotFoundError: If metadata was set to be updated, but the
        main image could not be found to update the metadata.
    """
    if series is None:
        series = 0
    filename_image5d, filename_meta = make_filenames(
        filename, series)
    image5d_ver_num = -1
    metadata = None
    try:
        # load image5d metadata; if updating, only fully load if curr ver
        metadata, image5d_ver_num = load_metadata(filename_meta, update_info)

        # load original image, using mem-mapped accessed for the image
        # file to minimize memory requirement, only loading on-the-fly
        image5d = np.load(filename_image5d, mmap_mode="r")
        print("image5d shape: {}".format(image5d.shape))
        if offset is not None and size is not None:
            # simplifies to reducing the image to a subset as an ROI if
            # offset and size given
            image5d = plot_3d.prepare_roi(image5d, offset, size)
            image5d = roi_to_image5d(image5d)

        if update_info:
            # if metadata < latest ver, update and load info
            load_info = _update_image5d_np_ver(
                image5d_ver_num, image5d, metadata, filename_meta)
            if load_info:
                # load updated archive
                metadata, image5d_ver_num = load_metadata(filename_meta)
        img5d = np_io.Image5d(
            image5d, filename_image5d, filename_meta, config.LoadIO.NP)
        if return_info:
            return img5d, metadata
        return img5d
    except OSError as e:
        print("Could not load image files for", filename)
        print(e)
        if update_info and -1 < image5d_ver_num < IMAGE5D_NP_VER:
            # set to update metadata but could not because image5d
            # was not available;
            # TODO: override option since older metadata vers may
            # still work, or image5d may not be necessary for upgrade
            raise FileNotFoundError(
                "image5d metadata is from an older version ({}, "
                "current version {}) and could not be updated because "
                "the original image5d file was not found."
                .format(image5d_ver_num, IMAGE5D_NP_VER))
        if return_info:
            return None, metadata
        return None


def setup_import_multipage(filename):
    """Find matching files for multipage image import.

    Multiple channels are assumed to be stored in separate files with
    :const:``CHANNEL_SEPARATOR`` followed by an integer corresponding to
    the channel number (0-based indexing), eg ``/path/to/image_ch_0.tif``.
    If any such file is found, only files with these channel designators
    will be taken. If none are found, ``filename`` will be taken directly.


    Args:
        filename (str): Path to image file to import. Files for separate
        channels with names based on this path will first be checked,
        falling back to use this name directly.

    Returns:
        dict[Any, List[str]], str: Ordered dictionary of channel numbers to
        sequences of image file paths to import, and the base path of the
        extracted files.
    
    Raises:
        FileNotFoundError: No existing files related to ``filename`` could
        be found.
    
    """
    path_split = libmag.splitext(filename)
    ext = path_split[1].lower()
    base_path = path_split[0]
    
    # find separate files for each channel; if selected file is in channel
    # format, deconstruct it to remove chl
    reg_iter = re.finditer(r"{}[0-9]+".format(CHANNEL_SEPARATOR), base_path)
    iter_ind = [m.start(0) for m in reg_iter]
    if len(iter_ind) > 0:
        base_path = base_path[:iter_ind[-1]]
    
    # get all files matching channel format
    path_base = "{}{}*".format(base_path, CHANNEL_SEPARATOR)
    filenames = []
    print("Looking for files for multi-channel images matching "
          "the format: {}{}".format(path_base, ext))
    matches = glob.glob(path_base)
    
    for match in matches:
        # prune files to matching extensions
        match_split = os.path.splitext(match)
        if match_split[1].lower() == ext:
            filenames.append(match)
    filenames = sorted(filenames)
    
    if filenames:
        # get dict of channels by files
        print("Found matching file(s), where each file will be imported "
              "as a separate channel:", filenames)
        chl_paths = _parse_import_chls(filenames)
    else:
        # take file directly with key specifying it could have multiple channels
        print("Using the given single file {}".format(filename))
        if not os.path.exists(filename):
            # if no related files to filename could be found, filename must
            # itself exist
            raise FileNotFoundError(
                "Multipage file to import does not exist:", filename)
        chl_paths = {_KEY_ANY_CHANNEL: [filename]}
    
    # parse to dict by channel
    return chl_paths, base_path


def _is_raw(path):
    """Check if a path is a RAW file based on extension.
    
    Args:
        path (str): Path to check

    Returns:
        bool: True if ``path``'s extension is RAW, case insensitive.

    """
    return os.path.splitext(path)[1].lower() == ".raw"


def setup_import_metadata(chl_paths, channel=None, series=None, z_max=-1):
    """Extract metadata and determine output image shape for importing
    multipage file(s).
    
    Args:
        chl_paths (dict[Any, List[str]]): Ordered dictionary of channel
            numbers to sequences of image file paths to import.
        channel (List[int]): Sequence of channel indices to import; defaults
            to None to import all channels.
        series (int): Series index to load. Defaults to None, which will use 0.
        z_max (int): Number of z-planes to load; defaults to -1 to load all.

    Returns:
        dict[:obj:`config.MetaKeys`]: Dictionary of metadata. RAW files will
        simply return a metadata dictionary populated with None values.

    """
    print("Extracting metadata for image import, may take awhile...")
    if series is None:
        series = 0
    path = tuple(chl_paths.values())[0][0]
    md = dict.fromkeys(config.MetaKeys)
    if _is_raw(path) or not is_javabridge_loaded():
        # RAW files will need to have metadata supplied manually; return
        # based on this extension to avoid startup time for Javabridge
        return md

    start_jvm()
    jb.attach()
    shape = None
    try:
        # get available embedded metadata via Bioformats
        names, sizes, res, md[config.MetaKeys.MAGNIFICATION], \
            md[config.MetaKeys.ZOOM], md[config.MetaKeys.DTYPE] = \
            parse_ome_raw(bf.get_omexml_metadata(path))
        # unlike config.resolutions, keep only single list for simplicity
        if res and len(res) > series:
            md[config.MetaKeys.RESOLUTIONS] = res[series]
        if sizes and len(sizes) > series:
            shape = list(sizes[series])
    except jb.JavaException as err:
        print(err)
    
    if shape is None:
        try:
            # fall back to getting a subset of metadata, also through Bioformats
            # TODO: see if necessary or improves performance
            sizes, dtype = find_sizes(path)
            if dtype:
                md[config.MetaKeys.DTYPE] = dtype.name
            shape = list(sizes[0])
        except (jb.JavaException, AttributeError) as err:
            # Python-Bioformats (v1.1) attempts to access currently non-existing
            # message attribute in JavaException from Javabridge (v1.0.18)
            print(err)
    
    if shape:
        shape = _update_shape_for_channels(shape, chl_paths, channel)[1]
        if z_max != -1:
            shape[1] = z_max
        md[config.MetaKeys.SHAPE] = shape
    jb.detach()
    
    return md


def _update_shape_for_channels(shape, chl_paths, channel):
    """Change image shape to match specified number of channels.
    
    Args:
        shape (List[int]): Image shape, with last dimenion for channels.
        chl_paths (dict): Dictionary of channels by files.
        channel (List[int]): Sequence of channels to keep.

    Returns:
        List[int], List[int]: Shape for input files; shape for output file
        as a copy of ``shape`` with channel size adjusted.

    """
    shape_out = list(shape)
    shape_in = shape_out
    if _KEY_ANY_CHANNEL in chl_paths:
        # file present with unspecified channel, potentially multichannel,
        # with shape assumed to be based on this file
        if channel:
            # limit channels to set parameter
            shape_out[-1] = len(channel)
    else:
        # assume only one channel per input file
        shape_out[-1] = len(channel) if channel else len(chl_paths.keys())
        shape_in = shape_out[:-1]
    return shape_in, shape_out


def import_multiplane_images(chl_paths, prefix, import_md, series=None,
                             offset=0, channel=None, fn_feedback=None):
    """Imports single or multiplane file(s) into Numpy format.
    
    For multichannel images, this import currently supports either a single
    file with multiple channels or multiple files each containing a single
    channel. Files will be loaded by Bioformats, with fallback by Numpy
    as RAW files. Output files are written plane-by-plane to memory-mapped
    files to bypass keeping the full input or output image in RAM.

    Args:
        chl_paths (dict[Any, List[str]]): Ordered dictionary of channel
            numbers to sequences of image file paths to import.
        prefix (str): Ouput base path.
        import_md (dict[:obj:`config.MetaKeys`]): Import metadata dictionary,
            used to set up the shape, data type (for RAW file import), and
            output image metadata (resolutions, zoom, magnification).
        series (int): Series index to load. Defaults to None, which will use 0.
        offset (int): z-plane offset from which to start importing.
            Defaults to 0.
        channel (List[int]): Sequence of channel indices to import; defaults
            to None to import all channels.
        fn_feedback (func): Callback function to give feedback strings
            during import; defaults to None.

    Returns:
        :obj:`np_io.Image5d: The 5D image object.
    
    """
    if not is_javabridge_loaded():
        return None

    time_start = time()
    if series is None:
        series = 0
    filename_image5d, filename_meta = make_filenames(prefix, series)
    libmag.printcb("Initializing multiplane image import planes to \"{}\", "
                   "may take awhile..."
                   .format(filename_image5d), fn_feedback)
    
    # set up channels in case chl_paths was updated after shape determination
    # and to get channels to extract from each file
    shape_in, shape = _update_shape_for_channels(
        import_md[config.MetaKeys.SHAPE], chl_paths, channel)
    if _KEY_ANY_CHANNEL in chl_paths:
        # unspecified channel file, potentially multichannel, takes
        # precedence over all other files
        chls_load = channel if channel else range(shape[-1])
        chl_paths = {_KEY_ANY_CHANNEL: chl_paths[_KEY_ANY_CHANNEL]}
    else:
        # assuming each file is single channel and skip if channel not in
        # channel parameter
        chls_load = [0]
        chl_paths = OrderedDict(
            [(k, v) for k, v in chl_paths.items()
             if not channel or k in channel])

    jb_attached = False
    image5d = None
    near_mins = []
    near_maxs = []
    chli = 0
    for chl, paths in chl_paths.items():
        # assume only one file per channel, ignoring others in same channel
        img_path = paths[0]
        
        if image5d is None:
            if shape[-1] == 1:
                shape = shape[:-1]  # remove channel dim if single channel
            shape = tuple(shape)
        
        # set up image reader
        rdr = None
        img_raw = None
        libmag.printcb(
            "Loading file {} for import".format(img_path), fn_feedback)
        if not _is_raw(img_path):
            # open non-RAW image with Python-Bioformats
            try:
                if not jb_attached:
                    # start JVM and attach to current thread
                    start_jvm()
                    jb.attach()
                    jb_attached = True
                rdr = bf.ImageReader(img_path, perform_init=True)
            except (jb.JavaException, AttributeError) as err:
                print(err)
        if rdr is None:
            # open image file as a RAW 3D array
            img_raw = np.memmap(
                img_path, dtype=import_md[config.MetaKeys.DTYPE],
                shape=tuple(shape_in[1:]), mode="r")
        
        len_shape = len(shape)
        len_shape_in = len(shape_in)
        plane_shape = None
        for chl_load in chls_load:
            lows = []
            highs = []
            for t in range(shape[0]):
                for z in range(shape[1]):
                    # import by channel plane
                    libmag.printcb(
                        "loading planes from time {}, z {}, channel {}"
                        .format(t, z, chl_load), fn_feedback)
                    if img_raw is not None:
                        # access plane from RAW memmapped file
                        img = (img_raw[z, ..., chl_load] if len_shape_in >= 5
                               else img_raw[z])
                    else:
                        # read plane with Bioformats reader; chl_load may be
                        # ignored for some formats, yielding multichannel planes
                        img = rdr.read(z=(z + offset), t=t, c=chl_load,
                                       series=series, rescale=False)
                    plane_shape = img.shape
                    
                    if image5d is None:
                        # open output file as memmap to directly write to disk,
                        # much faster than outputting to RAM first; supports
                        # NPY directly, unlike np.memmap
                        os.makedirs(
                            os.path.dirname(filename_image5d), exist_ok=True)
                        image5d = np.lib.format.open_memmap(
                            filename_image5d, mode="w+", dtype=img.dtype,
                            shape=shape)
                        print("setting image5d array for series {} with shape: "
                              "{}".format(series, image5d.shape))
                    
                    # near max/min bounds per channel for the given plane
                    low, high = calc_intensity_bounds(img, dim_channel=2)
                    lows.append(low)
                    highs.append(high)
                    if len_shape >= 5 and len(img.shape) == 2:
                        # squeeze 2D plane inside if separate file per channel
                        image5d[t, z, :, :, chli] = img
                    else:
                        image5d[t, z] = img
            if len(plane_shape) > 2 and plane_shape[2] > 1:
                # assume all planes were multichannel so all channels imported
                print("Multiple channels imported per plane, will assume "
                      "all channels are imported and end channel import")
                break
            near_mins, near_maxs = _calc_near_intensity_bounds(
                near_mins, near_maxs, lows, highs)
            chli += 1
        if rdr is not None:
            rdr.close()
        if img_raw is not None:
            img_raw.flush()
    
    # finalize import and save metadata
    image5d.flush()  # may not be necessary but ensure contents to disk
    print("file import time: {}".format(time() - time_start))
    #print("lows: {}, highs: {}".format(lows, highs))
    # TODO: consider saving resolutions as 1D rather than 2D array
    # with single resolution tuple
    md = save_image_info(
        filename_meta, [os.path.basename(prefix)], [shape],
        [import_md[config.MetaKeys.RESOLUTIONS]],
        import_md[config.MetaKeys.MAGNIFICATION],
        import_md[config.MetaKeys.ZOOM], near_mins, near_maxs)
    assign_metadata(md)
    libmag.printcb("Completed multiplane image import planes to \"{}\" "
                   "with metadata:\n{}"
                   .format(filename_image5d, md), fn_feedback)
    if jb_attached:
        jb.detach()
    return np_io.Image5d(
        image5d, filename_image5d, filename_meta, config.LoadIO.NP)


def _parse_import_chls(paths):
    """Sorts paths in channel format based on their channel number.
    
    Channel format is, ``<path>_ch_<n>``, where ``n`` is an integer. Paths
    that are not in channel format default to channel 0.
    
    Args:
        paths (List[str]): Sequence of paths.

    Returns:
        dict[int, List[str]]: Ordered dictionary of channel numbers to
        sequences of image file paths to import.

    """
    regex_chls = re.compile(r"{}[0-9]+".format(CHANNEL_SEPARATOR))
    chls = OrderedDict()
    len_sep = len(CHANNEL_SEPARATOR)
    for f in paths:
        # extract channel identifier and group file by channel, defaulting
        # to channel 0 if not in channel format
        f_chls = re.findall(regex_chls, f)
        chl = int(f_chls[0][len_sep:]) if f_chls else 0
        chls.setdefault(chl, []).append(f)
        print("chl: {}, path: {}".format(chl, f))
    return chls


def setup_import_dir(path):
    """Setup import of image files in an entire directory.
    
    All files in the folder will be gathered. Files from different channesl
    should have `_ch_<n>` just before the extension, where `n` is the
    channel number. Files without these channel designators will be
    assumed to be in channel 0.
    
    Args:
        path (str): Path to directory.

    Returns:
        dict[int, List[str]], dict[:obj:`config.MetaKeys`]: Ordered dictionary
        of channel numbers to sequences of image file paths to import and
        dictionary of metadata.
    
    Raise:
        FileNotFoundError: if no file from the directory can be loaded as
            an image.

    """
    # all files in the given folder will be imported in alphabetical order
    print("Importing files in directory {}:".format(path))
    paths = sorted(glob.glob(os.path.join(path, "*")))
    
    # set up paths for each channel and metadata dict
    chl_paths = _parse_import_chls(paths)
    md = dict.fromkeys(config.MetaKeys)
    
    # set shape and data type based on first loadable image in first channel
    chl_files = tuple(chl_paths.values())[0]
    shape = [1, len(chl_files), 0, 0, len(chl_paths.keys())]
    img = None
    for chl_file in chl_files:
        try:
            # load standard image types; does not read RAW files
            img = io.imread(chl_file)
            shape[2:4] = img.shape[:2]
            md[config.MetaKeys.DTYPE] = img.dtype.str
            break
        except ValueError:
            _logger.info(
                "Could not read %s as an image, reading next", chl_file)
    if img is None:
        _logger.warn(
            "Could not find image files in the directory: %s", path)
    md[config.MetaKeys.SHAPE] = shape
    return chl_paths, md


def import_planes_to_stack(chl_paths, prefix, import_md, rgb_to_grayscale=True,
                           fn_feedback=None):
    """Import single plane image files into a single volumetric image stack.

    Each file in ``chl_paths`` is assumed to be a 2D plane in a volumetric
    image with either a single channel or an RGB channel.

    Args:
        chl_paths (dict[int, List[str]]): Ordered dictionary of channel
            numbers to sequences of image file paths to import.
        prefix (str): Ouput base path; defaults to None to output to the
            ``path`` directory, also using the directory name as the
            image filename.
        import_md (dict[:obj:`config.MetaKeys`]): Import metadata dictionary,
            used for output image metadata (resolutions, zoom, magnification).
            Shape and data type are ignored since they are determined
            during import in case these values may have changed.
        rgb_to_grayscale (bool): Files with a three value third dimension
            are assumed to be RGB and will be converted to grayscale;
            defaults to True.
        fn_feedback (func): Callback function to give feedback strings
            during import; defaults to None.

    Returns:
        :obj:`np_io.Image5d: The 5D image object.

    """
    def import_files():
        # import files for the current channel
        lows = []
        highs = []
        img5d = image5d
        for filei, file in enumerate(chl_files):
            libmag.printcb("importing {}".format(file), fn_feedback)
            try:
                try:
                    # load standard image types
                    img = io.imread(file)
                except ValueError:
                    # load as a RAW image file
                    img = np.memmap(
                        file, dtype=import_md[config.MetaKeys.DTYPE],
                        shape=tuple(import_md[config.MetaKeys.SHAPE][2:4]),
                        mode="r")
                    
                if rgb_to_grayscale and img.ndim >= 3 and img.shape[2] == 3:
                    # assume that 3-channel images are RGB
                    # TODO: remove rgb_to_grayscale since must give single chl?
                    libmag.printcb(
                        "Converted from 3-channel (assuming RGB) to grayscale",
                        fn_feedback)
                    img = color.rgb2gray(img)
    
                if img5d is None:
                    # generate an array for all planes and channels based on
                    # dims of the first extracted plane and any channel keys
                    shape = [1, len(chl_files), *img.shape]
                    if num_chls > 1:
                        shape.append(num_chls)
                    os.makedirs(
                        os.path.dirname(filename_image5d_npz), exist_ok=True)
                    img5d = np.lib.format.open_memmap(
                        filename_image5d_npz, mode="w+", dtype=img.dtype,
                        shape=tuple(shape))
    
                # insert plane, without using channel dimension if no channel
                # designators were found in file names
                if num_chls > 1:
                    img5d[0, filei, ..., chli] = img
                else:
                    img5d[0, filei] = img
    
                # measure near low/high intensity values
                low, high = np.percentile(img, (0.5, 99.5))
                lows.append(low)
                highs.append(high)
            except ValueError as e1:
                libmag.printcb(
                    f"Could not load '{file}'; skipping it because of error: "
                    f"{e1}", fn_feedback)

        lows_chls.append(min(lows))
        highs_chls.append(max(highs))
        return img5d
    
    # each key is assumed to represent a distinct channel
    num_chls = len(chl_paths.keys())
    if num_chls < 1:
        return None

    # allow import of arbitrarily large images
    Image.MAX_IMAGE_PIXELS = None
    
    print("prefix", prefix)
    filename_image5d_npz, filename_info_npz = make_filenames(prefix + ".")
    libmag.printcb("Importing single-plane images into multiplane Numpy format "
                   "file: {}".format(filename_image5d_npz), fn_feedback)
    image5d = None
    lows_chls = []
    highs_chls = []
    chli = 0
    for chl_files in chl_paths.values():
        # import files for the given channel
        image5d = import_files()
        chli += 1

    # save metadata and load for immediate use
    md = save_image_info(
        filename_info_npz, [prefix], [image5d.shape],
        [import_md[config.MetaKeys.RESOLUTIONS]],
        import_md[config.MetaKeys.MAGNIFICATION],
        import_md[config.MetaKeys.ZOOM], lows_chls, highs_chls)
    assign_metadata(md)
    libmag.printcb("Saved image to \"{}\" with the following metadata:\n{}"
                   .format(filename_image5d_npz, md), fn_feedback)
    return np_io.Image5d(
        image5d, filename_image5d_npz, filename_info_npz, config.LoadIO.NP)


def calc_intensity_bounds(image5d, lower=0.5, upper=99.5, dim_channel=4):
    """Calculate image intensity boundaries for the given percentiles, 
    including boundaries for each channel in multichannel images.
    
    Assume that the image will be small enough to load entirely into 
    memory rather than calculating bounds plane-by-plane, but can also 
    be given an individual plane. Also assume that bounds for all channels 
    will be calculated.
    
    Args:
        image5d: Image as a 5D (t, z, y, x, c) array, or a 4D array if only 
            1 channel is present.
        lower: Lower bound as a percentile; defaults to 0.5.
        upper: Upper bound as a percentile; defaults to 99.5.
        dim_channel: Axis number of channel; defaults to 4, where the
            channel value is collapsed into the x-axis.
    
    Returns:
        Tuple of ``lows`` and ``highs``, each of which is a list of the 
        low and high values at the given percentile cutoffs for each channel.
    """
    multichannel, channels = plot_3d.setup_channels(image5d, None, dim_channel)
    lows = []
    highs = []
    for i in channels:
        image5d_show = image5d[..., i] if multichannel else image5d
        low, high = np.percentile(image5d_show, (lower, upper))
        lows.append(low)
        highs.append(high)
    return lows, highs


def _calc_near_intensity_bounds(near_mins, near_maxs, lows, highs):
    # get the extremes from lists of near-min/max vals
    if lows:
        num_channels = len(lows[0])
        if num_channels <= 1:
            # get min/max from list of 1-element arrays
            near_mins.append(min(lows)[0])
            near_maxs.append(max(highs)[0])
        else:
            # get min/max from columns of 2D array
            near_mins = np.amin(np.array(lows), 0)
            near_maxs = np.amax(np.array(highs), 0)
    return near_mins, near_maxs


def save_np_image(image, filename, series=None):
    """Save Numpy image to file.
    
    Assumes that the image or another image with similar parameters 
    has already been loaded so that the info file 
    can be constructed from the currently set parameters. Near min/max values 
    are generated from the entire image.
    
    Args:
        image: Numpy array.
        filename: Filename of original file, which will be passed to 
            :func:``make_filenames`` to create output filenames.
        series: Image series; defaults to None.
    """
    # save the image as a Numpy archive
    filename_image5d_npz, filename_info_npz = make_filenames(
        filename, series)
    with open(filename_image5d_npz, "wb") as out_file:
        np.save(out_file, image)

    # save a metadata file using the current settings and updating the
    # near min/max values
    lows, highs = calc_intensity_bounds(image)
    save_image_info(
        filename_info_npz, [os.path.basename(filename)], [image.shape], 
        config.resolutions, config.magnification, config.zoom, 
        lows, highs)


def calc_scaling(image5d, scaled, image5d_shape=None, scaled_shape=None):
    """Calculate the exact scaling between two images where one image had 
    been scaled from the other.
    
    Args:
        image5d (:obj:`np.ndarray`): Original image in 5D (time included,
            channel optional) format.
        scaled (:obj:`np.ndarray`): Scaled image, assumed to be in either
            3D or 5D format (3D with channel not currently supported).
        image5d_shape (List): ``image5d`` shape, which can be given if
            ``image5d`` is None; defaults to None.
        scaled_shape (List): ``scaled`` shape, which can be given if
            ``scaled`` is None; defaults to None.
    
    Returns:
        Array of (z, y, x) scaling factors from the original to the scaled
        image.
    """
    if image5d_shape is None:
        image5d_shape = image5d.shape
    if scaled_shape is None:
        scaled_shape = scaled.shape
    # remove time dimension if necessary
    if len(image5d_shape) >= 4:
        image5d_shape = image5d_shape[1:4]
    # TODO: assume only 3D (including 3D + channel) format?
    if len(scaled_shape) >= 4:
        scaled_shape = scaled_shape[1:4]
    scaling = np.divide(scaled_shape[:3], image5d_shape[:3])
    print("image scaling compared to image5d: {}".format(scaling))
    return scaling


def roi_to_image5d(roi):
    """Convert from ROI image to image5d format, which simply adds a time 
    dimension as the first dimension.
    
    Args:
        roi: ROI as a 3D (or 4D if channel dimension) array.
    
    Returns:
        ROI with additional time dimension prepended.
    """
    return roi[None]


if __name__ == "__main__":
    print("MagellanMapper importer manipulations")
    from magmap.io import cli

    cli.main(True)
