#!/bin/bash
# 3D image visualization
# Author: David Young, 2017
"""3D Visualization GUI.

This module is the main GUI for visualizing 3D objects from imaging stacks. 
The module can be run either as a script or loaded and initialized by 
calling main().

Examples:
    Launch the GUI with the given file at a particular size and offset::
        
        $ ./run img=/path/to/file.czi offset=30,50,205 size=150,150,10
    
    Alternatively, this module can be run as a script::
        
        $ python -m clrbrain.visualizer img=/path/to/file.czi

Command-line arguments in addition to those listed below:
    * scaling_factor: Zoom scaling (see detector.py).
    * 3d: 3D rendering type (see plot_3d.py).

Attributes:
    filename: The filename of the source images. A corresponding file with
        the subset as a 5 digit number (eg 00003) with .npz appended to 
        the end will be checked first based on this filename. Set with
        "img=path/to/file" argument.
    series: The series for multi-stack files, using 0-based indexing. Set
        with "series=n" argument.
    channel: The channel to view. Set with "channel=n" argument.
    roi_size: The size in pixels of the region of interest. Set with
        "size=x,y,z" argument, where x, y, and z are integers.
    offset: The bottom corner in pixels of the region of interest. Set 
        with "offset=x,y,z" argument, where x, y, and z are integers.
"""

import os
import sys
import datetime

import numpy as np
from traits.api import (HasTraits, Instance, on_trait_change, Button, 
                        Int, List, Array, push_exception_handler, Property)
from traitsui.api import (View, Item, HGroup, VGroup, Handler, 
                          RangeEditor, HSplit, TabularEditor)
from traitsui.tabular_adapter import TabularAdapter
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
import matplotlib.pylab as pylab

from clrbrain import importer
from clrbrain import detector
from clrbrain import plot_3d
from clrbrain import plot_2d
from clrbrain import sqlite

filename = None
series = 0 # series for multi-stack files
channel = 0 # channel of interest
roi_size = [100, 100, 15] # region of interest
offset = None

image5d = None # numpy image array
conn = None # sqlite connection
cur = None # sqlite cursor
params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'xx-small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small'}

ARG_IMG = "img"
ARG_OFFSET = "offset"
ARG_CHANNEL = "channel"
ARG_SERIES = "series"
ARG_SIDES = "size"
ARG_3D = "3d"
ARG_SCALING = "scaling"

def main():
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    """
    # command-line arguments
    global filename, series, channel, roi_size, offset
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
                if arg_split[1] in plot_3d.MLAB_3D_TYPES:
                    plot_3d.mlab_3d = arg_split[1]
                    print("3D rendering set to {}".format(arg_split[1]))
                else:
                    print("Did not recognize 3D rendering type: {}"
                          .format(arg_split[1]))
    
    # loads the image and GUI
    importer.start_jvm()
    #names, sizes = parse_ome(filename)
    #sizes = find_sizes(filename)
    global image5d, conn, cur
    image5d = importer.read_file(filename, series) #, z_max=cube_len)
    pylab.rcParams.update(params)
    np.set_printoptions(threshold=np.nan) # print full arrays
    conn, cur = sqlite.start_db()
    push_exception_handler(reraise_exceptions=True)
    visualization = Visualization()
    visualization.configure_traits()
    
def _fig_title(offset, roi_size):
    """Figure title parser.
    
    Arguments:
        offset: (x, y, z) image offset
        roi_size: (x, y, z) region of interest size
    
    Returns:
        Figure title string.
    """
    title = ("{}, series: {}\n"
             "offset: {}, ROI size: {}").format(os.path.basename(filename), series, 
                                                offset, roi_size)
    return title

class VisHandler(Handler):
    """Simple handler for Visualization object events.
    
    Closes the JVM when the window is closed.
    """
    def closed(self, info, is_ok):
        """Closes the Java VM when the GUI is closed.
        """
        importer.jb.kill_vm()
        global conn
        conn.close()

class SegmentsArrayAdapter(TabularAdapter):
    columns = [("i", "index"), ("z", 0), ("row", 1), ("col", 2), ("radius", 3)]
    index_text = Property
    
    def _get_index_text(self):
        return str(self.row)

class Visualization(HasTraits):
    """GUI for choosing a region of interest and segmenting it.
    
    TraitUI-based graphical interface for selecting dimensions of an
    image to view and segment.
    
    Attributes:
        x_low, x_high, ...: Low and high values for each offset.
        x_offset: Integer trait for x-offset.
        y_offset: Integer trait for y-offset.
        z_offset: Integer trait for z-offset.
        scene: The main scene
        btn_redraw_trait: Button editor for drawing the reiong of 
            interest.
        btn_segment_trait: Button editor for segmenting the ROI.
        roi: The ROI.
        segs_selected: List of indices of selected segments.
    """
    x_low = 0
    x_high = 100
    y_low = 0
    y_high = 100
    z_low = 0
    z_high = 100
    x_offset = Int
    y_offset = Int
    z_offset = Int
    roi_array = Array(Int, shape=(1, 3))
    scene = Instance(MlabSceneModel, ())
    btn_redraw_trait = Button("Redraw")
    btn_segment_trait = Button("Segment")
    btn_2d_trait = Button("2D Plots")
    btn_save_segments = Button("Save Segments")
    roi = None
    _segments = Array
    segs_selected = List # indices
    segs_table = TabularEditor(adapter=SegmentsArrayAdapter(), multi_select=True, 
                               selected_row="segs_selected")
    segs_cmap = None
    
    def save_segs(self):
        """Saves segments to database.
        
        Segments are selected from a table, and positions are transposed
        based on the current offset. Also inserts a new experiment based 
        on the filename if not already added.
        """
        if self.segments is None:
            print("no segments found")
            return
        elif self.segs_selected is None or len(self.segs_selected) < 1:
            #print(segs_selected)
            print("no segments selected")
            return
        segs_transposed = []
        for i in range(len(self.segments)):
            seg = self.segments[i]
            # for now, assumes incorrect if not in selected list
            confirmed = 1 if self.segs_selected.count(i) > 0 else 0
            seg_db = (seg[2] + self.x_offset, seg[1] + self.y_offset, 
                      seg[0] + self.z_offset, seg[3], confirmed)
            segs_transposed.append(seg_db)
        exp_id = sqlite.select_or_insert_experiment(conn, cur, 
                                                    os.path.basename(filename),
                                                    datetime.datetime(1000, 1, 1))
        sqlite.insert_blobs(conn, cur, exp_id, series, segs_transposed)
    
    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        # dimension max values in pixels
        size = image5d.shape
        self.z_high = size[1]
        self.y_high = size[2]
        self.x_high = size[3]
        curr_offset = offset
        # apply user-defined offsets
        if curr_offset is not None:
            self.x_offset = curr_offset[0]
            self.y_offset = curr_offset[1]
            self.z_offset = curr_offset[2]
        else:
            print("No offset, using standard one")
            curr_offset = self._curr_offset()
            #self.roi = show_roi(image5d, self, cube_len=cube_len)
        self.roi_array[0] = roi_size
        self.roi = plot_3d.show_roi(image5d, channel, self, self.roi_array[0], 
                                    offset=curr_offset)
        #self.segments, self.segs_cmap = detector.segment_roi(self.roi, self)
        self.segments = None
        #self.segs_selected = [0, 3]
        #self.save_segs()
        '''
        plot_2d.plot_2d_stack(self, _fig_title(curr_offset, self.roi_array[0]), image5d, 
                              channel, self.roi_array[0], curr_offset, 
                              self.segments, self.segs_cmap)
        '''
        
    @on_trait_change('x_offset,y_offset,z_offset')
    def update_plot(self):
        """Shows the chosen offset when an offset slider is moved.
        """
        print("x: {}, y: {}, z: {}".format(self.x_offset, self.y_offset, 
                                           self.z_offset))
    
    def _btn_redraw_trait_fired(self):
        # ensure that cube dimensions don't exceed array
        size = image5d.shape
        if self.x_offset + roi_size[0] > size[3]:
            self.x_offset = size[3] - roi_size[0]
        if self.y_offset + roi_size[1] > size[2]:
            self.y_offset = size[2] - roi_size[1]
        if self.z_offset + roi_size[2] > size[1]:
            self.z_offset = size[1] - roi_size[2]
        
        # show updated region of interest
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        self.roi = plot_3d.show_roi(image5d, channel, self, curr_roi_size, 
                                    offset=curr_offset)
        self.segments = None
    
    def _btn_segment_trait_fired(self):
        self.segments, self.segs_cmap = detector.segment_roi(self.roi, self)
    
    def _btn_2d_trait_fired(self):
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        print(curr_roi_size)
        plot_2d.plot_2d_stack(self, _fig_title(curr_offset, curr_roi_size), 
                              image5d, channel, curr_roi_size, 
                              curr_offset, self.segments, self.segs_cmap)
    
    def _btn_save_segments_fired(self):
        self.save_segs()
    
    def _curr_offset(self):
        return (self.x_offset, self.y_offset, self.z_offset)
    
    @property
    def segments(self):
        return self._segments
    
    @segments.setter
    def segments(self, val):
        if val is None:
            # need to include at least one row or else will crash
            self._segments = np.zeros((1, 4))
        else:
            self._segments = val
    
    # the layout of the dialog created
    view = View(
        HSplit(
            Item(
                'scene', 
                editor=SceneEditor(scene_class=MayaviScene),
                height=500, width=500, show_label=False
            ),
            VGroup(
                VGroup(
                    Item("roi_array", label="ROI dimensions (x,y,z)"),
                    Item(
                        "x_offset",
                        editor=RangeEditor(
                            low_name="x_low",
                            high_name="x_high",
                            mode="slider")
                    ),
                    Item(
                        "y_offset",
                        editor=RangeEditor(
                            low_name="y_low",
                            high_name="y_high",
                            mode="slider")
                    ),
                    Item(
                        "z_offset",
                        editor=RangeEditor(
                            low_name="z_low",
                            high_name="z_high",
                            mode="slider")
                    )
                ),
                HGroup(
                    Item("btn_redraw_trait", show_label=False), 
                    Item("btn_segment_trait", show_label=False), 
                    Item("btn_2d_trait", show_label=False)
                ),
                Item(
                    "_segments",
                    editor=segs_table,
                    show_label=False
                ),
                Item("btn_save_segments", show_label=False)
            )
        ),
        handler=VisHandler(),
        title="clrbrain",
        resizable=True
    )

if __name__ == "__main__":
    print("Starting visualizer...")
    main()
    