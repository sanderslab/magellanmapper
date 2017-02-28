#!/bin/bash
# Image stack importer
# Author: David Young, 2017
"""Imports image stacks using Bioformats.

Bioformats is access through Python-Bioformats and Javabridge.
Images will be imported into a 4/5D Numpy array.
"""

from time import time
from xml import etree as et
import numpy as np
import javabridge as jb
import bioformats as bf

from clrbrain import detector

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
        print("tag: {}".format(child.tag))
        if child.tag.endswith("Instrument"):
            for grandchild in child:
                if grandchild.tag.endswith("Detector"):
                    zoom = float(grandchild.attrib["Zoom"])
                elif grandchild.tag.endswith("Objective"):
                    magnification = float(grandchild.attrib["NominalMagnification"])
            print("zoom: {}, magnification: {}".format(zoom, magnification))
        elif child.tag.endswith("Image"):
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

def read_file(filename, series, save=True, load=True, z_max=-1, 
              offset=None):
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
    filename_npz = filename + str(series).zfill(5) + ".npz"
    if load:
        try:
            #time_start = time()
            output = np.load(filename_npz)
            #print('file opening time: %f' %(time() - time_start))
            image5d = output["image5d"]
            try:
                detector.set_scaling_factor(output["magnification"], output["zoom"])
            except KeyError:
                print("could not find magnification/zoom, defaulting to {}"
                      .format(detector.scaling_factor))
            try:
                detector.resolutions = output["resolutions"]
                print("set resolutions to {}".format(detector.resolutions))
            except KeyError:
                print("could not find resolutions")
            return image5d
        except IOError:
            print("Unable to load {}, will attempt to reload {}"
                  .format(filename_npz, filename))
    start_jvm()
    names, sizes, resolutions, magnification, zoom, pixel_type = parse_ome_raw(filename)
    detector.set_scaling_factor(magnification, zoom)
    #sizes, dtype = find_sizes(filename)
    rdr = bf.ImageReader(filename, perform_init=True)
    size = sizes[series]
    nt, nz = size[:2]
    if z_max != -1:
        nz = z_max
    if offset is None:
        offset = (0, 0, 0) # (x, y, z)
    dtype = getattr(np, pixel_type)
    if size[4] <= 1:
        image5d = np.empty((nt, nz, size[2], size[3]), dtype)
        load_channel = 0
    else:
        image5d = np.empty((nt, nz, size[2], size[3], size[4]), dtype)
        load_channel = None
    print("setting image5d array with shape: {}".format(image5d.shape))
    time_start = time()
    for t in range(nt):
        check_dtype = True
        for z in range(nz):
            print("loading planes from [{}, {}]".format(t, z))
            img = rdr.read(z=(z + offset[2]), t=t, c=load_channel,
                           series=series, rescale=False)
            if check_dtype:
                if img.dtype != image5d.dtype:
                    raise TypeError("Storing as data type {} "
                                    "when image is in type {}"
                                    .format(img.dtype, image5d.dtype))
                else:
                    print("Storing as data type {}".format(img.dtype))
                check_dtype = False
            image5d[t, z] = img
    print('file import time: %f' %(time() - time_start))
    outfile = open(filename_npz, "wb")
    if save:
        time_start = time()
        # could use compression (savez_compressed), but much slower
        np.savez(outfile, image5d=image5d, names=names, sizes=sizes, 
                 resolutions=resolutions, magnification=magnification, 
                 zoom=zoom, pixel_type=pixel_type)
        outfile.close()
        print('file save time: %f' %(time() - time_start))
    return image5d
