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
    * scaling_factor: Zoom scaling (see detector.py). Only set if unable
        to be detected from the image file or if the saved numpy array
        does not have scaling information as it would otherwise
        override this setting.
    * 3d: 3D rendering type (see plot_3d.py).

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
"""

import os
import sys
from time import time
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

from clrbrain import cli
from clrbrain import importer
from clrbrain import detector
from clrbrain import plot_3d
from clrbrain import plot_2d
from clrbrain import sqlite

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'xx-small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small'}

def main():
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    """
    cli.main()
    pylab.rcParams.update(params)
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
    title = ("{} (series {})\n"
             "offset {}, ROI size {}").format(os.path.basename(cli.filename), cli.series, 
                                                offset, tuple(roi_size))
    return title

class VisHandler(Handler):
    """Simple handler for Visualization object events.
    
    Closes the JVM when the window is closed.
    """
    def closed(self, info, is_ok):
        """Closes the Java VM when the GUI is closed.
        """
        importer.jb.kill_vm()
        cli.conn.close()

class SegmentsArrayAdapter(TabularAdapter):
    columns = [("i", "index"), ("z", 0), ("row", 1), ("col", 2), 
               ("radius", 3), ("confirmed", 4)]
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
        segments: Array of segments; if None, defaults to a Numpy array
            of zeros with one row.
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
    roi = None # combine with roi_array?
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
        segs_transposed = []
        for i in range(len(self.segments)):
            seg = self.segments[i]
            # ignores user added segments, where radius assumed to be 0,
            # that are no longer selected
            if self.segs_selected.count(i) <= 0 and np.allclose(seg[3], 0):
                print("ignoring unselected user added segment: {}".format(seg))
            else:
                seg_db = (seg[2] + self.x_offset, seg[1] + self.y_offset, 
                          seg[0] + self.z_offset, seg[3], seg[4])
                segs_transposed.append(seg_db)
        # inserts experiment if not already added, then segments
        exp_id = sqlite.select_or_insert_experiment(cli.conn, cli.cur, 
                                                    os.path.basename(cli.filename),
                                                    datetime.datetime(1000, 1, 1))
        sqlite.insert_blobs(cli.conn, cli.cur, exp_id, cli.series, segs_transposed)
    
    def show_3d(self):
        """Shows the 3D plot.
        
        If the processed image flag is true ("proc=1"), the region will be
        taken from the saved processed array. Type of 3D display depends
        on the "3d" flag.
        """
        # show updated region of interest
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        if cli.image5d_proc is None:
            self.roi = plot_3d.prepare_roi(cli.image5d, cli.channel, curr_roi_size, 
                                           offset=curr_offset)
            self.roi = plot_3d.denoise(self.roi)
        else:
            print("loading from previously processed image")
            self.roi = plot_3d.prepare_roi(cli.image5d_proc, cli.channel, curr_roi_size, 
                                           offset=curr_offset)
        mlab_3d = cli.mlab_3d
        if mlab_3d == cli.MLAB_3D_TYPES[0]:
            plot_3d.plot_3d_surface(self.roi, self)
        else:
            plot_3d.plot_3d_points(self.roi, self)
        self.segments = None
      
    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        # dimension max values in pixels
        size = cli.image5d.shape
        self.z_high = size[1]
        self.y_high = size[2]
        self.x_high = size[3]
        curr_offset = cli.offset
        # apply user-defined offsets
        if curr_offset is not None:
            self.x_offset = curr_offset[0]
            self.y_offset = curr_offset[1]
            self.z_offset = curr_offset[2]
        else:
            print("No offset, using standard one")
            curr_offset = self._curr_offset()
            #self.roi = show_roi(cli.image5d, self, cube_len=cube_len)
        self.roi_array[0] = cli.roi_size
        self.show_3d()
        #self.segs_selected = [0, 3]
        #self.save_segs()
        '''
        plot_2d.plot_2d_stack(self, _fig_title(curr_offset, self.roi_array[0]), cli.image5d, 
                              cli.channel, self.roi_array[0], curr_offset, 
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
        size = cli.image5d.shape
        curr_roi_size = self.roi_array[0].astype(int)
        if curr_roi_size[0] + self.x_offset > self.x_high:
            curr_roi_size[0] = self.x_high - self.x_offset
            self.roi_array = [curr_roi_size]
        if curr_roi_size[1] + self.y_offset > self.y_high:
            curr_roi_size[1] = self.y_high - self.y_offset
            self.roi_array = [curr_roi_size]
        if curr_roi_size[2] + self.z_offset > self.z_high:
            curr_roi_size[2] = self.z_high - self.z_offset
            self.roi_array = [curr_roi_size]
        print("ROI size: {}".format(self.roi_array[0].astype(int)))
        self.show_3d()
    
    def _btn_segment_trait_fired(self):
        mlab_3d = cli.mlab_3d
        if mlab_3d == cli.MLAB_3D_TYPES[0]:
            # segments using the Random-Walker algorithm
            self.segments = detector.segment_rw(self.roi)
            self.segs_cmap = plot_3d.show_surface_labels(self.segments, self)
        else:
            # segments using blob detection
            if cli.segments_proc is None:
                # blob detects the ROI
                self.segments = detector.segment_blob(self.roi)
                self.segs_cmap = plot_3d.show_blobs(self.segments, self)
            else:
                # uses blobs from loaded segments
                roi_x, roi_y, roi_z = self.roi_array[0].astype(int)
                x, y, z = self._curr_offset()
                segs = cli.segments_proc[np.all([cli.segments_proc[:, 0] >= z, 
                                                 cli.segments_proc[:, 0] < z + roi_z,
                                                 cli.segments_proc[:, 1] >= y, 
                                                 cli.segments_proc[:, 1] < y + roi_y,
                                                 cli.segments_proc[:, 2] >= x, 
                                                 cli.segments_proc[:, 2] < x + roi_x], 
                                                axis=0)]
                # transpose to make coordinates relative to offset
                segs = np.copy(segs)
                self.segments = np.subtract(segs, (z, y, x, 0, 0))
                self.segs_cmap = plot_3d.show_blobs(self.segments, self)
    
    def _btn_2d_trait_fired(self):
        # shows 2D plots
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        print(curr_roi_size)
        plot_2d.plot_2d_stack(self, _fig_title(curr_offset, curr_roi_size), 
                              cli.image5d, cli.channel, curr_roi_size, 
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
        """Sets segments.
        
        Args:
            val: Numpy array of (n, 5) shape with segments. Defaults to one
                row if None.
        """
        if val is None:
            # need to include at least one row or else will crash
            self._segments = np.zeros((1, 5))
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
    