#!/bin/bash
# Image stack importer
# Author: David Young, 2017, 2019
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
    SUFFIX_IMAGE5D: Suffix for the image5d Numpy array archive.
    SUFFIX_INFO: Suffix for the image5d Numpy array "info" support archive.
"""

import os
from time import time
import glob
import re
from xml import etree as et
import warnings

import SimpleITK as sitk

from clrbrain import config
from clrbrain import detector
from clrbrain import plot_3d
from clrbrain import lib_clrbrain

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
IMAGE5D_NP_VER = 14 # image5d Numpy saved array version number
SUFFIX_IMAGE5D = "_image5d.npz" # should actually be .npy
SUFFIX_INFO = "_info.npz"

CHANNEL_SEPARATOR = "_ch_"

def start_jvm(heap_size="8G"):
    """Starts the JVM for Python-Bioformats.
    
    Args:
        heap_size: JVM heap size, defaulting to 8G.
    """
    if not jb:
        lib_clrbrain.warn("Python-Javabridge not available, cannot start JVM")
        return
    jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)

def stop_jvm():
    if not jb:
        lib_clrbrain.warn("Python-Javabridge not available, cannot stop JVM")
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
    array_order = "TZYXC" # desired dimension order
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
    print('time for finding sizes: %f' %(time() - time_start))
    return sizes, dtype

def make_filenames(filename, series, modifier=""):
    """Make Clrbrain-oriented image and image metadata filenames.
    
    Args:
        filename: Original path from which Clrbrain-oriented filenames 
            will be derived.
        series: Image series.
        modifier: Separator for image series; defaults to "_"
    
    Returns:
        Tuple of path to the main image and path to metadata.
    """
    print("filename: {}".format(filename))
    filename_base = filename_to_base(filename, series, modifier)
    print("filename_base: {}".format(filename_base))
    filename_image5d_npz = filename_base + SUFFIX_IMAGE5D
    filename_info_npz = filename_base + SUFFIX_INFO
    return filename_image5d_npz, filename_info_npz

def filename_to_base(filename, series, modifier=""):
    """Convert an image path to a base path including series information 
    for use within Clrbrain.
    
    Args:
        filename: Path to original image.
        series: Series (eg tile) within image.
        modifier: Modifier string prior to series; default to empty string.
    """
    return "{}_{}{}".format(
        lib_clrbrain.splitext(filename)[0], modifier, 
        lib_clrbrain.series_as_str(series))

def deconstruct_np_filename(np_filename, ext="czi"):
    """Deconstruct Numpy image filename to the appropriate image components.
    
    Args:
        np_filename: Numpy image filename, including series number.
        ext: Original image's extension; defaults to "czi".
    
    Returns:
        Tuple with ``filename`` as the path before the series section and 
        ``series`` as the series number as an integer. Both elements default 
        to None if the series component could not be found ``np_filename``.
    """
    series_reg = re.compile(r"\_[0-9]{5}")
    series_fill = re.findall(series_reg, np_filename)
    series = None
    filename = None
    if series_fill:
        series = int(series_fill[0][1:])
        filename = series_reg.split(np_filename)[0] + "." + ext
    return filename, series

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
    output = read_np_archive(info)
    info.close()
    for key, value in output.items():
        print("{}: {}".format(key, value))

def _update_image5d_np_ver(curr_ver, image5d, info, filename_info_npz):
    # update image archive metadata using dictionary of values successfully 
    # loaded from the archive
    
    if curr_ver >= IMAGE5D_NP_VER:
        # no updates necessary
        return False
    
    print("Updating image metadata to version {}".format(IMAGE5D_NP_VER))
    
    if curr_ver <= 10:
        # ver 10 -> 11
        # no change except ver since most likely won't encounter any difference
        pass
    
    if curr_ver <= 11:
        # ver 11 -> 12
        if info["pixel_type"] != image5d.dtype:
            # Numpy tranpositions did not update pixel type and min/max
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
        
    # backup and save updated info
    lib_clrbrain.backup_file(
        filename_info_npz, modifier="_v{}".format(curr_ver))
    info["ver"] = IMAGE5D_NP_VER
    outfile_info = open(filename_info_npz, "wb")
    np.savez(outfile_info, **info)
    outfile_info.close()
    
    return True

def read_np_archive(archive):
    """Load Numpy archive file into a dictionary, skipping any values 
    that cannot be loaded.
    
    Args:
        archive: Loaded Numpy archive.
    
    Returns:
        Dictionary with keys and values corresponding to that of the 
        Numpy archive, skipping any values that could not be loaded 
        such as those that would require pickling when not allowed.
    """
    output = {}
    for key in archive.keys():
        try:
            output[key] = archive[key]
        except ValueError as e:
            print("unable to load {} from archive, will ignore".format(key))
    return output

def read_info(filename_info_npz, check_ver=False):
    """Load image info, such as saved microscopy data and image ranges, 
    storing some values into appropriate module level variables.
    
    Args:
        filename_info_npz: Path to image info file.
        check_ver: True to stop loading if the archive's version number  
            is less than :const:``IMAGE5D_NP_VER``; defaults to False.
    
    Returns:
        Tuple of ``output``, the dictionary with image info, and 
        ``image5d_ver_num``, the version number of the info file, 
        which is -1 if the key could not be found.
    """
    print("Reading image metadata from {}".format(filename_info_npz))
    archive = np.load(filename_info_npz)
    output = read_np_archive(archive)
    image5d_ver_num = -1
    try:
        # find the info version number
        image5d_ver_num = output["ver"]
        print("loaded image5d version number {}".format(image5d_ver_num))
    except KeyError:
        print("could not find image5d version number")
    if not check_ver or image5d_ver_num >= IMAGE5D_NP_VER:
        # load into various module variables unless checking version 
        # and below current version to avoid errors during loading
        try:
            names = output["names"]
            print("names: {}".format(names))
        except KeyError:
            print("could not find names")
        try:
            sizes = output["sizes"]
            print("sizes {}".format(sizes))
        except KeyError:
            print("could not find sizes")
        try:
            config.resolutions = output["resolutions"]
            print("set resolutions to {}".format(config.resolutions))
        except KeyError:
            print("could not find resolutions")
        try:
            detector.magnification = output["magnification"]
            print("magnification: {}".format(detector.magnification))
        except KeyError:
            print("could not find magnification")
        try:
            detector.zoom = output["zoom"]
            print("zoom: {}".format(detector.zoom))
        except KeyError:
            print("could not find zoom")
        try:
            config.near_min = output["near_min"]
            print("set near_min to {}".format(config.near_min))
        except KeyError:
            print("could not find near_max")
        try:
            config.near_max = output["near_max"]
            print("set near_max to {}".format(config.near_max))
            vmax_orig = np.copy(config.vmax_overview)
            config.vmax_overview = config.near_max * 1.1
            len_vmax_overview = len(config.vmax_overview)
            for i, val in enumerate(vmax_orig):
                if i < len_vmax_overview and val is not None:
                    # replace with non-default vals, usually set at cmd-line
                    config.vmax_overview[i] = val
            print("Set vmax_overview to {}".format(config.vmax_overview))
        except KeyError:
            print("could not find near_max")
    return output, image5d_ver_num

def read_file(filename, series, load=True, z_max=-1, 
              offset=None, size=None, channel=None, return_info=False, 
              import_if_absent=True, update_info=True):
    """Reads in an imaging file.
    
    Loads a Numpy image if available as determined by 
    :func:``make_filenames``. An offset and size can be given to load an 
    only an ROI of the image. If the corresponding Numpy image cannot be 
    found, one will be generated from the given source image. For TIFF images, 
    multiple channels will assume to be stored in separate files with 
    :const:``CHANNEL_SEPARATOR`` followed by an integer corresponding to the 
    channel number (0-based indexing).
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
        series: Series index to load.
        load: If True, attempts to load a Numpy array from the same 
            location and name except for ".npz" appended to the end 
            (default). The array can be accessed as "output['image5d']".
        z_max: Number of z-planes to load, or -1 if all should be loaded
            (default).
        offset: Tuple of offset given as (x, y, z) from which to start
            loading z-plane (x, y ignored for now). If Numpy image info already 
            exists, this tuple will be used to load only an ROI of the image. 
            If importing a new Numpy image, offset[2] will be used an a 
            z-offset from which to start importing. Defaults to None.
        size: Tuple of ROI size given as (x, y, z). If Numpy image info already 
            exists, this tuple will be used to load only an ROI of the image. 
            Defaults to None.
        channel: Channel number, currently used only to load a channel when 
            a Numpy ROI image exists. Otherwise, all channels available will 
            be imported into a new Numpy image. Defaults to None.
        return_info: True if the Numpy info file should be returned for a 
            dictionary of image properties; defaults to False.
        import_if_absent: True if the image should be imported into a Numpy 
            image if it does not exist; defaults to True.
        update_info: True if the associated image5d info file should be 
            updated; defaults to True.
    
    Returns:
        image5d, the array of image data. If ``return_info`` is True, a 
        second value a dictionary of image properties will be returned.
    """
    path_split = lib_clrbrain.splitext(filename)
    ext = path_split[1]
    filename_image5d_npz, filename_info_npz = make_filenames(
        filename, series)
    if load:
        image5d_ver_num = -1
        try:
            time_start = time()
            # load image5d metadata; if updating, only fully load if curr ver
            output, image5d_ver_num = read_info(filename_info_npz, update_info)
            
            # load original image, using mem-mapped accessed for the image
            # file to minimize memory requirement, only loading on-the-fly
            image5d = np.load(filename_image5d_npz, mmap_mode="r")
            print("image5d shape: {}".format(image5d.shape))
            if offset is not None and size is not None:
                # simplifies to reducing the image to a subset as an ROI if 
                # offset and size given
                image5d = plot_3d.prepare_roi(image5d, size, offset)
                image5d = roi_to_image5d(image5d)
            
            if update_info:
                # if metadata < latest ver, update and load info
                load_info = _update_image5d_np_ver(
                    image5d_ver_num, image5d, output, filename_info_npz)
                if load_info:
                    # load updated archive
                    output, image5d_ver_num = read_info(filename_info_npz)
            if return_info:
                return image5d, output
            return image5d
        except IOError as e:
            print(e)
            if import_if_absent:
                print("will attempt to reload {}".format(filename))
            else:
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
                    return None, output
                return None
    
    if jb is None or bf is None:
        lib_clrbrain.warn(
            "Python-Bioformats or Python-Javabridge not available, "
            "multi-page images cannot be imported")
        return None
    
    start_jvm()
    time_start = time()
    image5d = None
    shape = None
    if offset is None:
        offset = (0, 0, 0) # (x, y, z)
    num_files = 1
    if ext == ".tiff" or ext == ".tif":
        # import multipage TIFFs
        print("Loading multipage TIFF...")
        
        # find files for each channel, defaulting to load all channels available
        name = os.path.basename(filename)
        channel_num = "*" if channel is None else "{}*".format(channel)
        filenames = sorted(glob.glob(
            os.path.splitext(filename)[0] + CHANNEL_SEPARATOR + channel_num))
        print(filenames)
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
        names, sizes, config.resolutions, detector.magnification, \
            detector.zoom, pixel_type = parse_ome_raw(filenames[0])
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
                print("loading planes from [{}, {}]".format(t, z))
                img = rdr.read(z=(z + offset[2]), t=t, c=channel,
                               series=series, rescale=False)
                if image5d is None:
                    # open file as memmap to directly output to disk, which is much 
                    # faster than outputting to RAM and saving to disk
                    image5d = np.lib.format.open_memmap(
                        filename_image5d_npz, mode="w+", dtype=img.dtype, 
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
    time_start = time()
    image5d.flush() # may not be necessary but ensure contents to disk
    print("flush time: {}".format(time() - time_start))
    #print("lows: {}, highs: {}".format(lows, highs))
    # TODO: consider saving resolutions as 1D rather than 2D array
    # with single resolution tuple
    save_image_info(
        filename_info_npz, [name], [shape], [config.resolutions[series]], 
        detector.magnification, detector.zoom, near_mins, near_maxs)
    return image5d

def read_file_sitk(filename_sitk, filename_np, series=0):
    """Read file through SimpleITK and export to Numpy array format, 
    loading associated metadata and formatting array into Clrbrain image5d 
    format.
    
    Args:
        filename_sitk: Path to file in a format that can be read by SimpleITK.
        filename_np: Path to basis for Clrbrain Numpy archive files, which 
            will be used to load metadata file. If this archive does not 
            exist, metadata will be determined from ``filename_sitk`` 
            as much as possible.
        series: Image series number used to find the associated Numpy 
            archive; defaults to 0.
    
    Returns:
        Image array in Clrbrain image5d format. Associated metadata will 
        have been loaded into module-level variables.
    
    Raises:
        ``FileNotFoundError`` if ``filename_sitk`` cannot be found, after 
        attempting to load metadata from ``filename_np``.
    """
    # get metadata from Numpy archive
    filename_image5d_npz, filename_info_npz = make_filenames(
        filename_np, series)
    if os.path.exists(filename_info_npz):
        output, image5d_ver_num = read_info(filename_info_npz)
    
    # load image via SimpleITK
    if not os.path.exists(filename_sitk):
        raise FileNotFoundError("could not find file {}".format(filename_sitk))
    img_sitk = sitk.ReadImage(filename_sitk)
    img_np = sitk.GetArrayFromImage(img_sitk)
    
    if config.resolutions is None:
        # fallback to determining metadata directly from sitk file
        lib_clrbrain.warn(
            "Clrbrain image metadata file not loaded; will fallback to {} "
            "for metadata".format(filename_sitk))
        config.resolutions = np.array([img_sitk.GetSpacing()[::-1]])
        print("set resolutions to {}".format(config.resolutions))
    
    image5d = img_np[None] # insert time axis as first dim
    return image5d

def import_dir(path):
    files = sorted(glob.glob(path))
    #files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    num_files = len(files)
    if num_files < 1:
        return None
    name = os.path.dirname(files[0])
    filename_image5d_npz, filename_info_npz = make_filenames(name + ".czi", 0)
    image5d = None
    lows = []
    highs = []
    i = 0
    for f in files:
        print("importing {}".format(f))
        img = io.imread(f)
        if image5d is None:
            #image5d = np.empty((1, len(files), *img.shape))
            image5d = np.lib.format.open_memmap(
                filename_image5d_npz, mode="w+", dtype=img.dtype, 
                shape=(1, len(files), *img.shape))
        image5d[0, i] = img
        low, high = np.percentile(img, (0.5, 99.5))
        lows.append(low)
        highs.append(high)
        i += 1
    save_image_info(
        filename_info_npz, [name], [image5d.shape], config.resolutions, 
        detector.magnification, detector.zoom, [min(lows)], [max(highs)])
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

def save_np_image(image, filename, series):
    """Save Numpy image to file.
    
    Assumes that the image or another image with similar parameters 
    has already been loaded so that the info file 
    can be constructed from the currently set parameters. Near min/max values 
    are generated from the entire image.
    
    Args:
        image: Numpy array.
        filename: Filename of original file, which will be passed to 
            :func:``make_filenames`` to create output filenames.
        series: Image series.
    """
    filename_image5d_npz, filename_info_npz = make_filenames(
        filename, series)
    out_file = open(filename_image5d_npz, "wb")
    np.save(out_file, image)
    out_file.close()
    lows, highs = calc_intensity_bounds(image)
    save_image_info(
        filename_info_npz, [os.path.basename(filename)], [image.shape], 
        config.resolutions, detector.magnification, detector.zoom, 
        lows, highs)

def calc_scaling(image5d, scaled):
    """Calculate the exact scaling between two images where one image had 
    been scaled from the other.
    
    Args:
        image5d: Original image in 5D (time included, channel optional) format.
        scaled: Scaled image.
    
    Returns:
        Array of (z, y, x) scaling factors from the original to the scaled
        image.
    """
    shape = image5d.shape
    scaled_shape = scaled.shape
    # remove time dimension
    if image5d.ndim >=4:
        shape = shape[1:4]
    if scaled.ndim >=4:
        scaled_shape = scaled_shape[1:4]
    scaling = np.divide(scaled_shape[0:3], shape[0:3])
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
    return np.array([roi])

if __name__ == "__main__":
    print("Clrbrain importer manipulations")
    from clrbrain import cli
    cli.main(True)
