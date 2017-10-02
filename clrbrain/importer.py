#!/bin/bash
# Image stack importer
# Author: David Young, 2017
"""Imports image stacks using Bioformats.

Bioformats is access through Python-Bioformats and Javabridge.
Images will be imported into a 4/5D Numpy array.

Attributes:
    PIXEL_DTYPE: Dictionary of corresponding data types for 
        output given by Bioformats library. Alternatively, should detect
        pixel data type directly using parse_ome_raw().
"""

import os
from time import time
import glob
from xml import etree as et
import numpy as np
import javabridge as jb
import bioformats as bf
from skimage import io
from skimage import transform

from clrbrain import detector
from clrbrain import plot_2d
from clrbrain import plot_3d
from clrbrain import lib_clrbrain

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

# image5d Numpy saved array version number
IMAGE5D_NP_VER = 10

def start_jvm(heap_size="8G"):
    """Starts the JVM for Python-Bioformats.
    
    Args:
        heap_size: JVM heap size, defaulting to 8G.
    """
    jb.start_vm(class_path=bf.JARS, max_heap_size=heap_size)

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

def _make_filenames(filename, series, modifier=""):
    filename_base = filename_to_base(filename, series, modifier)
    filename_image5d_npz = filename_base + "_image5d.npz"
    filename_info_npz = filename_base + "_info.npz"
    return filename_image5d_npz, filename_info_npz

def _save_image_info(filename_info_npz, names, sizes, resolutions, 
                     magnification, zoom, pixel_type, near_min, near_max):
    outfile_info = open(filename_info_npz, "wb")
    time_start = time()
    np.savez(outfile_info, ver=IMAGE5D_NP_VER, names=names, sizes=sizes, 
             resolutions=resolutions, 
             magnification=magnification, zoom=zoom, 
             pixel_type=pixel_type, near_min=near_min, 
             near_max=near_max)
    outfile_info.close()
    print("info file saved to {}".format(filename_info_npz))
    print("file save time: {}".format(time() - time_start))

def read_file(filename, series, save=True, load=True, z_max=-1, 
              offset=None, size=None, channel=-1, return_info=False):
    """Reads in an imaging file.
    
    Can load the file from a saved Numpy array and also for only a series
    of z-planes if asked.
    
    Args:
        filename: Image file, assumed to have metadata in OME XML format.
        series: Series index to load.
        save: True to save the resulting Numpy array (default).
        load: If True, attempts to load a Numpy array from the same 
            location and name except for ".npz" appended to the end 
            (default). The array can be accessed as "output['image5d']".
        z_max: Number of z-planes to load, or -1 if all should be loaded
            (default).
        offset: Tuple of offset given as (x, y, z) from which to start
            loading z-plane (x, y ignored for now). Defaults to 
            (0, 0, 0).
    
    Returns:
        image5d: array of image data.
        size: tuple of dimensions given as (time, z, y, x, channels).
    """
    filename_image5d_npz, filename_info_npz = _make_filenames(filename, series)
    if load:
        try:
            time_start = time()
            # loads stored Numpy arrays, using mem-mapped accessed for the image
            # file to minimize memory requirement, only loading on-the-fly
            output = np.load(filename_info_npz)
            image5d = np.load(filename_image5d_npz, mmap_mode="r")
            if offset is not None and size is not None:
                # simplifies to reducing the image to a subset as an ROI if 
                # offset and size given
                image5d = plot_3d.prepare_roi(image5d, channel, size, offset)
            '''
            # convert old monolithic archive into 2 separate archives
            filename_npz = filename + str(series).zfill(5) + ".npz" # old format
            output = np.load(filename_npz)
            outfile_image5d = open(filename_image5d_npz, "wb")
            np.save(outfile_image5d, output["image5d"])
            outfile_image5d.close()
            outfile_info = open(filename_info_npz, "wb")
            np.savez(outfile_info, names=output["names"], sizes=output["sizes"], 
                     resolutions=output["resolutions"], magnification=output["magnification"], 
                     zoom=output["zoom"], pixel_type=output["pixel_type"])
            outfile_info.close()
            print('file opening time: %f' %(time() - time_start))
            return
            '''
            # find the info version number
            try:
                image5d_ver_num = output["ver"]
                print("loaded image5d version number {}".format(image5d_ver_num))
            except KeyError:
                print("could not find image5d version number")
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
                detector.resolutions = output["resolutions"]
                print("set resolutions to {}".format(detector.resolutions))
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
                plot_3d.near_max = output["near_max"]
                print("set near_max to {}".format(plot_3d.near_max))
            except KeyError:
                print("could not find near_max")
            if return_info:
                return image5d, output
            return image5d
        except IOError:
            print("Unable to load Numpy array files {} or {}, will attempt to reload {}"
                  .format(filename_image5d_npz, filename_info_npz, filename))
    start_jvm()
    # parses the XML tree directly
    names, sizes, detector.resolutions, magnification, zoom, pixel_type = parse_ome_raw(filename)
    time_start = time()
    image5d = None
    filename_image5d_npz, filename_info_npz = _make_filenames(filename, series)
    #sizes, dtype = find_sizes(filename)
    rdr = bf.ImageReader(filename, perform_init=True)
    # only loads one series for now though could create a loop for multiple series
    size = sizes[series]
    nt, nz = size[:2]
    if z_max != -1:
        nz = z_max
    if offset is None:
        offset = (0, 0, 0) # (x, y, z)
    dtype = getattr(np, pixel_type)
    # generate image stack dimensions based on whether channel dim exists
    if size[4] <= 1:
        shape = (nt, nz, size[2], size[3])
        load_channel = 0
    else:
        shape = (nt, nz, size[2], size[3], size[4])
        load_channel = None
    # open file as memmap to directly output to disk, which is much faster
    # than outputting to RAM and saving to disk
    image5d = np.lib.format.open_memmap(
        filename_image5d_npz, mode="w+", dtype=dtype, shape=shape)
    print("setting image5d array for series {} with shape: {}".format(
          series, image5d.shape))
    lows = []
    highs = []
    for t in range(nt):
        check_dtype = True
        for z in range(nz):
            print("loading planes from [{}, {}]".format(t, z))
            img = rdr.read(z=(z + offset[2]), t=t, c=load_channel,
                           series=series, rescale=False)
            low, high = np.percentile(img, (0.5, 99.5))
            lows.append(low)
            highs.append(high)
            #print("near_min: {}, near_max: {}, min: {}, max: {}"
            #      .format(low, high, np.min(img), np.max(img)))
            # checks predicted data type with actual one to ensure consistency, 
            # which was necessary in case the PIXEL_DTYPE dictionary became inaccurate
            # but shoudln't be an issue when parsing date type directly from XML
            if check_dtype:
                if img.dtype != image5d.dtype:
                    raise TypeError("Storing as data type {} "
                                    "when image is in type {}"
                                    .format(img.dtype, image5d.dtype))
                else:
                    print("Storing as data type {}".format(img.dtype))
                check_dtype = False
            image5d[t, z] = img
    print("file import time: {}".format(time() - time_start))
    # TODO: consider removing option since generally always want to save
    if save:
        time_start = time()
        image5d.flush() # may not be necessary but ensure contents to disk
        print("flush time: {}".format(time() - time_start))
        #print("lows: {}, highs: {}".format(lows, highs))
        # TODO: consider saving resolutions as 1D rather than 2D array
        # with single resolution tuple
        _save_image_info(filename_info_npz, [names[series]], 
                         [sizes[series]], [detector.resolutions[series]], 
                         magnification, zoom, 
                         pixel_type, min(lows), max(highs))
    return image5d

def import_dir(path):
    files = sorted(glob.glob(path))
    #files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    num_files = len(files)
    if num_files < 1:
        return None
    name = os.path.dirname(files[0])
    filename_image5d_npz, filename_info_npz = _make_filenames(name + ".czi", 0)
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
    _save_image_info(filename_info_npz, [name], 
                     [image5d.shape], detector.resolutions, 
                     detector.magnification, detector.zoom, image5d.dtype,
                     min(lows), max(highs))
    return image5d

def filename_to_base(filename, series, modifier=""):
    return filename.replace(".czi", "_") + modifier + str(series).zfill(5)

def transpose_npy(filename, series, plane=None, rescale=None):
    image5d, image5d_info = read_file(filename, series, return_info=True)
    info = dict(image5d_info)
    sizes = info["sizes"]
    filename_image5d_npz, filename_info_npz = _make_filenames(
        filename, series, "transposed")
    offset = 0 if image5d.ndim <= 3 else 1
    image5d_swapped = image5d
    if plane is not None and plane != plot_2d.PLANE[0]:
        # swap z-y to get (y, z, x) order for xz orientation
        image5d_swapped = np.swapaxes(image5d_swapped, offset, offset + 1)
        detector.resolutions[0] = lib_clrbrain.swap_elements(
            detector.resolutions[0], 0, 1)
        #sizes[0] = lib_clrbrain.swap_elements(sizes[0], 0, 1, offset)
        if plane == plot_2d.PLANE[2]:
            # swap new y-x to get (x, z, y) order for yz orientation
            image5d_swapped = np.swapaxes(image5d_swapped, offset, offset + 2)
            detector.resolutions[0] = lib_clrbrain.swap_elements(
                detector.resolutions[0], 0, 2)
            #sizes[0] = lib_clrbrain.swap_elements(sizes[0], 0, 2, offset)
    if rescale is not None:
        rescaled = image5d_swapped
        if offset > 0:
            rescaled = rescaled[0]
        multichannel = rescaled.shape[-1] > 1
        rescaled = transform.rescale(
            rescaled, rescale, mode="reflect", multichannel=multichannel)
        if offset > 0:
            image5d_swapped = np.array([rescaled])
        else:
            image5d_swapped = rescaled
        detector.resolutions = np.multiply(detector.resolutions, 1 / rescale)
    sizes[0] = image5d_swapped.shape
    print("new shape: {}".format(image5d_swapped.shape))
    print("detector.resolutions: {}".format(detector.resolutions))
    print("sizes: {}".format(sizes))
    image5d_transposed = np.lib.format.open_memmap(
        filename_image5d_npz, mode="w+", dtype=image5d_swapped.dtype, 
        shape=image5d_swapped.shape)
    image5d_transposed[:] = image5d_swapped[:]
    image5d.flush()
    info["resolutions"] = detector.resolutions
    info["sizes"] = sizes
    outfile_info = open(filename_info_npz, "wb")
    np.savez(outfile_info, **info)
    outfile_info.close()
    '''
    _save_image_info(filename_info_npz, None, 
                     None, detector.resolutions, 
                     None, None, 
                     None, None, None)
    '''
    print("saved transposed file to {} with shape {}".format(
        filename_image5d_npz, image5d_transposed.shape))

if __name__ == "__main__":
    print("Clrbrain importer manipulations")
    from clrbrain import cli
    cli.main(True)
    transpose_npy(cli.filename, cli.series, plot_2d.plane)
    #transpose_npy(cli.filename, cli.series, rescale=0.05)
