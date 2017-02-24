#!/bin/bash
# Command line parsing and setup
# Author: David Young, 2017
"""Command line parser and and environment setup for Clrbrain.

This module can be run either as a script to work in headless mode or 
loaded and initialized by calling main().

Examples:
    Launch the GUI with the given file at a particular size and offset::
        
        $ python -m clrbrain.cli img=/path/to/file.czi offset=30,50,205 \
            size=150,150,10

Command-line arguments in addition to those listed below:
    * scaling_factor: Zoom scaling (see detector.py). Only set if unable
        to be detected from the image file or if the saved numpy array
        does not have scaling information as it would otherwise
        override this setting.

Attributes:
    filename: The filename of the source images. A corresponding file with
        the subset as a 5 digit number (eg 00003) with .npz appended to 
        the end will be checked first based on this filename. Set with
        "img=path/to/file" argument.
    proc: Flag for loading processed files. "0" not to load (default), or
        "1" to load processed (ie denoised) image and segments.
    series: The series for multi-stack files, using 0-based indexing. Set
        with "series=n" argument.
    channel: The channel to view. Set with "channel=n" argument.
    roi_size: The size in pixels of the region of interest. Set with
        "size=x,y,z" argument, where x, y, and z are integers.
    offset: The bottom corner in pixels of the region of interest. Set 
        with "offset=x,y,z" argument, where x, y, and z are integers.
    MLAB_3D_TYPES: Tuple of types of 3D visualizations.
        * "surface": Renders surfaces in Mayavi contour and
          surface.
        * "point": Renders as raw points, minus points under
          the intensity_min threshold.
        * "headless": Processes and segments the entire image
          stack and exits non-interactively.
        * "importonly": Imports an image stack and exists non-
          interactively.
    mlab_3d: The chosen type.
"""

import os
import sys
from time import time
import numpy as np

from clrbrain import importer
from clrbrain import sqlite
from clrbrain import plot_3d
from clrbrain import detector
from clrbrain import chunking

filename = None
series = 0 # series for multi-stack files
channel = 0 # channel of interest
roi_size = [100, 100, 15] # region of interest
offset = None

image5d = None # numpy image array
load_proc = False
image5d_proc = None
segments_proc = None
conn = None # sqlite connection
cur = None # sqlite cursor

MLAB_3D_TYPES = ("surface", "point", "headless", "importonly")
mlab_3d = MLAB_3D_TYPES[1]

ARG_IMG = "img"
ARG_PROC = "proc"
ARG_OFFSET = "offset"
ARG_CHANNEL = "channel"
ARG_SERIES = "series"
ARG_SIDES = "size"
ARG_3D = "3d"
ARG_SCALING = "scaling"
ARG_SAVEFIG = "savefig"

def main():
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    """
    # command-line arguments
    global filename, series, channel, roi_size, offset, load_proc, mlab_3d
    for arg in sys.argv:
        arg_split = arg.split("=")
        if len(arg_split) == 1:
            print("Skipped argument: {}".format(arg_split[0]))
        elif len(arg_split) >= 2:
            if arg_split[0] == ARG_OFFSET:
                offset_split = arg_split[1].split(",")
                if len(offset_split) >= 3:
                    offset = tuple(int(i) for i in offset_split)
                    print("Set offset: {}".format(offset))
                else:
                    print("Offset ({}) should be given as 3 values (x, y, z)"
                          .format(arg_split[1]))
            elif arg_split[0] == ARG_IMG:
                filename = arg_split[1]
                print("Opening image file: {}".format(filename))
            elif arg_split[0] == ARG_PROC:
                load_proc = arg_split[1] == "1"
                print("Set to load processed file: {}".format(load_proc))
            elif arg_split[0] == ARG_CHANNEL:
                channel = int(arg_split[1])
                print("Set to channel: {}".format(channel))
            elif arg_split[0] == ARG_SERIES:
                series = int(arg_split[1])
                print("Set to series: {}".format(series))
            elif arg_split[0] == ARG_SCALING:
                scaling = float(arg_split[1])
                detector.scaling_factor = scaling
                print("Set scaling factor to: {}".format(scaling))
            elif arg_split[0] == ARG_SIDES:
                sides_split = arg_split[1].split(",")
                if len(sides_split) >= 3:
                    roi_size = tuple(int(i) for i in sides_split)
                    print("Set roi_size: {}".format(roi_size))
                else:
                    print("Size ({}) should be given as 3 values (x, y, z)"
                          .format(arg_split[1]))
            elif arg_split[0] == ARG_3D:
                if arg_split[1] in MLAB_3D_TYPES:
                    mlab_3d = arg_split[1]
                    print("3D rendering set to {}".format(mlab_3d))
                else:
                    print("Did not recognize 3D rendering type: {}"
                          .format(arg_split[1]))
            elif arg_split[0] == ARG_SAVEFIG:
                plot_2d.savefig = arg_split[1]
                print("Set savefig extension to: {}".format(plot_2d.savefig))
    
    # loads the image and GUI
    global image5d, conn, cur
    #np.set_printoptions(threshold=np.nan) # print full arrays
    conn, cur = sqlite.start_db()
    filename_proc = filename + str(series).zfill(5) + "_proc.npz"
    if load_proc:
        # loads from processed file
        try:
            output = np.load(filename_proc)
            global image5d_proc, segments_proc
            image5d_proc = output["roi"]
            segments_proc = output["segments"]
            return
        except IOError:
            print("Unable to load {}".format(filename_proc))
            load_proc = False
    image5d = importer.read_file(filename, series) #, z_max=cube_len)
    if mlab_3d == MLAB_3D_TYPES[2]:
        # denoises and segments the entire stack, saving processed image
        # and segments to file
        time_start = time()
        shape = image5d.shape
        roi = plot_3d.prepare_roi(image5d, channel, (shape[3], shape[2], shape[1]))
        chunking.scale_max_pixels()
        tol = chunking.calc_tolerance()
        print("tol: {}".format(tol))
        sub_rois, overlap, sub_rois_offsets = chunking.stack_splitter(roi)
        segments_all = None
        for z in range(sub_rois.shape[0]):
            for y in range(sub_rois.shape[1]):
                for x in range(sub_rois.shape[2]):
                    sub_roi = sub_rois[z, y, x]
                    print("processing sub_roi at {}, {}, {}, with shape {}...".format(z, y, x, sub_roi.shape))
                    sub_roi = plot_3d.denoise(sub_roi)
                    segments = detector.segment_blob(sub_roi)
                    offset = sub_rois_offsets[z, y, x]
                    # transpose segments
                    if segments is not None:
                        segments = np.add(segments, (offset[0], offset[1], offset[2], 0, 0))
                    # join segments
                    if segments_all is None:
                        segments_all = segments
                    elif segments is not None:
                        segments = chunking.remove_close_blobs(segments, segments_all, slice(0, 3), tol)
                        segments_all = np.concatenate((segments_all, segments))
                    sub_rois[z, y, x] = sub_roi
        merged = chunking.merge_split_stack(sub_rois, overlap)
        """
        if segments_all is not None:
            segments_all = chunking.remove_duplicate_blobs(segments_all, slice(0, 3))
            print("all segments: {}\n{}".format(segments_all.shape[0], segments_all))
        """
        print("total processing time (s): {}".format(time() - time_start))
        outfile = open(filename_proc, "wb")
        time_start = time()
        np.savez(outfile, roi=merged, segments=segments_all)
        outfile.close()
        print('file save time: %f' %(time() - time_start))
        # exit directly since otherwise appears to hang
        os._exit(os.EX_OK)
    elif mlab_3d == MLAB_3D_TYPES[3]:
        # already imported so now simply exits
        print("imported {}, will exit".format(filename))
        os._exit(os.EX_OK)
    
if __name__ == "__main__":
    print("Starting clrbrain command-line interface...")
    main()
    
