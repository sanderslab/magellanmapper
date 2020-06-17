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

import os
from time import time
import glob
import re
from xml import etree as et
import warnings

from magmap.settings import config
from magmap.io import np_io
from magmap.plot import plot_3d
from magmap.io import libmag

import numpy as np
try:
    import javabridge as jb
except ImportError as e:
    jb = None
    warnings.warn(
        "Python-Javabridge could not be found, so there will be error when "
        "attempting to import images into Numpy format", ImportWarning)
try:
    import bioformats as bf
except ImportError as e:
    bf = None
    warnings.warn(
        "Python-Bioformats could not be found, so there will be error when "
        "attempting to import images into Numpy format", ImportWarning)
from PIL import Image
from skimage import color
from skimage import io

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
IMAGE5D_NP_VER = 14  # image5d Numpy saved array version number

CHANNEL_SEPARATOR = "_ch_"
_EXT_TIFFS = (".tif", ".tiff")


def start_jvm(heap_size="8G"):
    """Starts the JVM for Python-Bioformats.
    
    Args:
        heap_size: JVM heap size, defaulting to 8G.
    """
    if not jb:
        libmag.warn("Python-Javabridge not available, cannot start JVM")
        return
    jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)


def stop_jvm():
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


def parse_ome_raw(filename):
    """Parses the microscope's XML file directly, pulling out salient info
    for further processing.
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
    
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
    metadata = bf.get_omexml_metadata(filename)
    metadata_root = et.ElementTree.fromstring(metadata)
    for child in metadata_root:
        # tag name is at end of a long string of other identifying info
        print("tag: {}".format(child.tag))
        if child.tag.endswith("Instrument"):
            # microscope info
            for grandchild in child:
                if grandchild.tag.endswith("Detector"):
                    zoom = float(grandchild.attrib["Zoom"])
                elif grandchild.tag.endswith("Objective"):
                    magnification = float(grandchild.attrib["NominalMagnification"])
            print("zoom: {}, magnification: {}".format(zoom, magnification))
        elif child.tag.endswith("Image"):
            # image file info
            names.append(child.attrib["Name"])
            for grandchild in child:
                if grandchild.tag.endswith("Pixels"):
                    att = grandchild.attrib
                    sizes.append(tuple([int(att[t]) for t in size_tags]))
                    resolutions.append(tuple([float(att[t]) for t in res_tags]))
                    # assumes pixel type is same for all images
                    if pixel_type is None:
                        pixel_type = att["Type"]
                        print("pixel_type: {}".format(pixel_type))
    print("names: {}".format(names))
    print("sizes: {}".format(sizes))
    print("resolutions: {}".format(resolutions))
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


def deconstruct_np_filename(np_filename):
    """Deconstruct Numpy image filename to the original name from which it
    was based.

    Also parses sub-image offset and shape information if the filename
    is in sub-image format as generated by
    :meth:`magmap.cv.stack_detect.make_subimg_name`.
    
    Args:
        np_filename (str): Numpy image filename.
    
    Returns:
        str, str, str: ``filename``, the deconstructed filename, without
        suffixes and ending in a period. If no matching suffix is found,
        simply returns ``np_filename``. ``offset``, the sub-image offset,
        of None if not a sub-image. ``size``, the sub-image shape, or
        None if not a sub-image.
    """
    filename = np_filename
    offset = None
    size = None
    for suffix in (config.SUFFIX_IMAGE5D, config.SUFFIX_META,
                   config.SUFFIX_SUBIMG):
        suffix_full = "_{}".format(suffix)
        if np_filename.endswith(suffix_full):
            # strip suffix to get base path
            filename = np_filename[:len(np_filename)-len(suffix_full)]
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
                    filename = filename[:len(filename)-len(subimg)]
            filename = "{}.".format(filename)
            break
    return filename, offset, size


def save_image_info(filename_info_npz, names, sizes, resolutions, 
                    magnification, zoom, near_min, near_max, 
                    scaling=None, plane=None):
    """Save image metadata.
    
    Args:
        filename_info_npz: Output path.
        names: Sequence of names for each series.
        sizes: Sequence of sizes for each series.
        resolutions: Sequence of resolutions for each series.
        magnification: Objective magnification.
        zoom: Objective zoom.
        near_min: Sequence of near minimum intensities, with each element 
            in turn holding a sequence with values for each channel.
        near_max: Sequence of near maximum intensities, with each element 
            in turn holding a sequence with values for each channel.
        scaling: Rescaling value for up/downsampled images; defaults 
            to None.
        plane: Planar orientation compared with original for transposed 
            images; defaults to None.

    Returns:
        :dict: The saved metadata as a dictionary.
    
    """
    outfile_info = open(filename_info_npz, "wb")
    time_start = time()
    np.savez(outfile_info, ver=IMAGE5D_NP_VER, names=names, sizes=sizes, 
             resolutions=resolutions, magnification=magnification, zoom=zoom, 
             near_min=near_min, near_max=near_max, scaling=scaling, plane=plane)
    outfile_info.close()
    print("info file saved to {}".format(filename_info_npz))
    print("file save time: {}".format(time() - time_start))
    
    # reload and show info file contents
    print("Saved image metadata:")
    info = np.load(filename_info_npz)
    output = np_io.read_np_archive(info)
    info.close()
    for key, value in output.items():
        print("{}: {}".format(key, value))
    return output


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
            image5d_orig = read_file(
                config.filenames[1], config.series, update_info=False)
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
            num_channels = image5d.shape[4]
            near_mins, near_maxs = _calc_near_intensity_bounds(
                num_channels, near_mins, near_maxs, lows, highs)
        info["near_min"] = near_mins
        info["near_max"] = near_maxs
        info["scaling"] = scaling
        info["plane"] = config.plane
    
    if curr_ver <= 13:
        # ver 13 -> 14
        
        # pixel_type no longer saved since redundant with image5d.dtype
        if "pixel_type" in info:
            del info["pixel_type"]

    print("Updated metadata:\n{}".format(info))
    # backup and save updated info
    libmag.backup_file(
        filename_info_npz, modifier="_v{}".format(curr_ver))
    info["ver"] = IMAGE5D_NP_VER
    outfile_info = open(filename_info_npz, "wb")
    np.savez(outfile_info, **info)
    outfile_info.close()
    
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
        archive = np.load(path)
    except FileNotFoundError:
        libmag.warn("Could not find metadata file {}".format(path))
        return None, image5d_ver_num
    output = np_io.read_np_archive(archive)
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
        config.resolutions = md["resolutions"]
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
            config.vmax_overview = config.near_max * 1.1
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
        :obj:`np.ndarray`, dict: The image array as a 5D array in the format,
        ``t, z, y, x[, c]``. If ``return_info`` is True, a dictionary of
        image properties will also be returned.
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
            image5d = plot_3d.prepare_roi(image5d, size, offset)
            image5d = roi_to_image5d(image5d)

        if update_info:
            # if metadata < latest ver, update and load info
            load_info = _update_image5d_np_ver(
                image5d_ver_num, image5d, metadata, filename_meta)
            if load_info:
                # load updated archive
                metadata, image5d_ver_num = load_metadata(filename_meta)
        if return_info:
            return image5d, metadata
        return image5d
    except IOError as e:
        print(e)
        if update_info and image5d_ver_num < IMAGE5D_NP_VER:
            # set to update metadata but could not because image5d
            # was not available;
            # TODO: override option since older metadata vers may
            # still work, or image5d may not be necessary for upgrade
            raise IOError(
                "image5d metadata is from an older version ({}, "
                "current version {}) could not be loaded because "
                "the original image filedoes not exist. Please "
                "reopen with the original image file to update "
                "the metadata.".format(image5d_ver_num, IMAGE5D_NP_VER))
        if return_info:
            return None, metadata
        return None


def import_bioformats(filename, series=None, z_max=-1, offset=0,
                      channel=None, prefix=None, fn_feedback=None):
    """Imports single or multiplane file(s) into Numpy format via Bioformats.

    For TIFF images, multiple channels will assume to be stored in separate
    files with :const:``CHANNEL_SEPARATOR`` followed by an integer
    corresponding to the channel number (0-based indexing), eg
    ``/path/to/image_ch_0.tif``.

    Args:
        filename: Image file, assumed to have metadata in OME XML format.
        series: Series index to load. Defaults to None, which will use 0.
        z_max: Number of z-planes to load, or -1 if all should be loaded
            (default).
        offset (int): z-plane offset from which to start importing.
            Defaults to 0.
        channel: Channel number, currently used only to load a channel when
            a Numpy ROI image exists. Otherwise, all channels available will
            be imported into a new Numpy image. Defaults to None.
        prefix (str): Ouput base path; defaults to None to use ``filename``.
        fn_feedback (func): Callback function to give feedback strings
            during import; defaults to None.

    Returns:
        :obj:`np.ndarray`: The image array as a 5D array in the format,
        ``t, z, y, x[, c]``.
    """
    if jb is None or bf is None:
        libmag.warn(
            "Python-Bioformats or Python-Javabridge not available, "
            "multi-page images cannot be imported")
        return None

    time_start = time()
    if series is None:
        series = 0
    path_split = libmag.splitext(filename)
    ext = path_split[1].lower()
    filename_image5d, filename_meta = make_filenames(
        filename, series)
    start_jvm()
    image5d = None
    num_files = 1
    if ext in _EXT_TIFFS:
        # import multipage TIFFs
        print("Loading multipage TIFF...")
        
        # find files for each channel, defaulting to load all channels
        # available and allowing any TIFF-like extension
        name = os.path.basename(filename)
        channel_num = "*" if channel is None else "{}*".format(channel)
        tif_base = "{}{}{}".format(
            path_split[0], CHANNEL_SEPARATOR, channel_num)
        filenames = []
        tif_searches = [tif_base]
        print("Looking for TIFF files for multi-channel images matching "
              "the format:", tif_base)
        matches = glob.glob(tif_base)
        if not matches:
            # fall back to matching any file with the same name regardless
            # of extension, typically for single-channel images
            tif_base = "{}.*".format(path_split[0])
            tif_searches.append(tif_base)
            print("Looking for TIFF files matching the format:", tif_base)
            matches = glob.glob(tif_base)
        for match in matches:
            # prune files to any TIFF-like name
            match_split = os.path.splitext(match)
            if match_split[1].lower() in _EXT_TIFFS:
                filenames.append(match)
        filenames = sorted(filenames)
        print("Found matching TIFF file(s), where each file will be imported "
              "as a separate channel:", filenames)
        if not filenames:
            raise IOError(
                "No filenames matching the format(s), \"{}\" with "
                "extensions of types {}".format(
                    tif_searches, ", ".join(_EXT_TIFFS)))
        num_files = len(filenames)
        
        # require resolution information as it will be necessary for 
        # detections, etc.
        if config.resolutions is None:
            raise IOError("Could not import {}. Please specify resolutions, "
                          "magnification, and zoom.".format(filenames[0]))
        sizes, dtype = find_sizes(filenames[0])
        shape = list(sizes[0])
        
        # fit the final shape to the number of channels
        if num_files > 1:
            shape[-1] = num_files
        if shape[-1] == 1:
            # remove channel dimension if only single channel
            shape = shape[:-1]
        shape = tuple(shape)
        print(shape)
    else:
        # default import mode, which assumes parseable OME header, tested 
        # on CZI files
        print("Loading {} file...".format(ext))
        
        # parses the XML tree directly
        filenames = [filename]
        names, sizes, config.resolutions, config.magnification, \
            config.zoom, pixel_type = parse_ome_raw(filenames[0])
        shape = sizes[series]
        if z_max != -1:
            shape[1] = z_max
        #dtype = getattr(np, pixel_type)
        if shape[4] <= 1 or channel is not None:
            # remove channel dimension if only single channel or channel is 
            # explicitly specified
            shape = shape[:-1]
            if channel is None:
                # default to channel 0 if not specified by only single channel
                channel = 0
        name = names[series]

    if prefix:
        # output files to a given prefix-based name
        filename_image5d, filename_meta = make_filenames(prefix)
    near_mins = []
    near_maxs = []
    for img_path in filenames:
        # multiple images for multichannel TIFF files, but only single 
        # image if integrated CZI
        rdr = bf.ImageReader(img_path, perform_init=True)
        lows = []
        highs = []
        if num_files > 1:
            channel_num = int(
                os.path.splitext(img_path)[0].split(CHANNEL_SEPARATOR)[1])
            print("adding {} to channel {}".format(img_path, channel_num))
        for t in range(shape[0]):
            for z in range(shape[1]):
                msg = "loading planes from [{}, {}]".format(t, z)
                print(msg)
                if fn_feedback:
                    fn_feedback(msg)
                img = rdr.read(z=(z + offset), t=t, c=channel,
                               series=series, rescale=False)
                if image5d is None:
                    # open file as memmap to directly output to disk, which is much 
                    # faster than outputting to RAM and saving to disk
                    image5d = np.lib.format.open_memmap(
                        filename_image5d, mode="w+", dtype=img.dtype,
                        shape=shape)
                    print("setting image5d array for series {} with shape: {}"
                          .format(series, image5d.shape))
                # near max/min bounds per channel for the given plane
                low, high = calc_intensity_bounds(img, dim_channel=2)
                lows.append(low)
                highs.append(high)
                if num_files > 1:
                    # squeeze plane inside if separate file per channel
                    image5d[t, z, :, :, channel_num] = img
                else:
                    image5d[t, z] = img
        near_mins, near_maxs = _calc_near_intensity_bounds(
            num_files, near_mins, near_maxs, lows, highs)
    print("file import time: {}".format(time() - time_start))
    image5d.flush()  # may not be necessary but ensure contents to disk
    #print("lows: {}, highs: {}".format(lows, highs))
    # TODO: consider saving resolutions as 1D rather than 2D array
    # with single resolution tuple
    md = save_image_info(
        filename_meta, [name], [shape], [config.resolutions[series]],
        config.magnification, config.zoom, near_mins, near_maxs)
    assign_metadata(md)
    return image5d


def _parse_import_chls(paths):
    """Sorts paths in channel format based on their channel number.
    
    Channel format is, ``<path>_ch_<n>``, where ``n`` is an integer. Paths
    that are not in channel format default to channel 0.
    
    Args:
        paths (List[str]): Sequence of paths.

    Returns:
        dict[int, List[str]]: Dictionary of channel numbers to sequences of
        image file paths to import.

    """
    regex_chls = re.compile(r"{}[0-9]+".format(CHANNEL_SEPARATOR))
    chls = {}
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
    
    Args:
        path (str): Path to directory.

    Returns:
        dict[int, List[str]]: Dictionary of channel numbers to sequences of
        image file paths to import.

    """
    # all files in the given folder will be imported in alphabetical order
    print("Importing files in directory {}:".format(path))
    paths = sorted(glob.glob(os.path.join(path, "*")))
    return _parse_import_chls(paths)


def import_planes_to_stack(chls, prefix, rgb_to_grayscale=True,
                           fn_feedback=None):
    """Import single plane image files into a single volumetric image stack.

    Each file is assumed to be a 2D plane in a volumetric image, ordered
    alphabetically. All files in the folder will be imported. Files from
    different channesl should have `_ch_<n>` just before the extension,
    where `n` is the channel number. If any such file is found, only
    files with these channel designators will be imported.

    Args:
        chls (dict[int, List[str]]): Dictionary of channel numbers to
            sequences of image file paths to import.
        prefix (str): Ouput base path; defaults to None to output to the
            ``path`` directory, also using the directory name as the
            image filename.
        rgb_to_grayscale (bool): Files with a three value third dimension
            are assumed to be RGB and will be converted to grayscale;
            defaults to True.
        fn_feedback (func): Callback function to give feedback strings
            during import; defaults to None.

    Returns:
        :obj:`np.ndarray`: The imported image as a Numpy array.

    """
    def import_files():
        # import files for the current channel
        lows = []
        highs = []
        img5d = image5d
        for filei, file in enumerate(chl_files):
            libmag.printcb("importing {}".format(file), fn_feedback)
            img = io.imread(file)
            if rgb_to_grayscale and img.ndim >= 3 and img.shape[2] == 3:
                # assume that 3-value 3rd channel images are RGB
                print("converted from 3-channel (assuming RGB) to grayscale")
                img = color.rgb2gray(img)

            if img5d is None:
                # generate an array for all planes and channels based on
                # dimensions of the first extracted plane and any channel keys
                shape = [1, len(chl_files), *img.shape]
                if num_chls > 1:
                    shape.append(num_chls)
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

        lows_chls.append(min(lows))
        highs_chls.append(max(highs))
        return img5d
    
    # each key is assumed to represent a distinct channel
    num_chls = len(chls.keys())
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
    for chli, chl_files in chls.items():
        # import files for the given channel
        image5d = import_files()

    # save metadata and load for immediate use
    md = save_image_info(
        filename_info_npz, [prefix], [image5d.shape], config.resolutions,
        config.magnification, config.zoom, lows_chls, highs_chls)
    assign_metadata(md)
    libmag.printcb("Saved image to \"{}\" with the following metadata:\n{}"
                   .format(filename_image5d_npz, md), fn_feedback)
    return image5d


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


def _calc_near_intensity_bounds(num_channels, near_mins, near_maxs, lows, 
                                highs):
    # get the extremes from lists of near-min/max vals
    if num_channels > 1:
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
