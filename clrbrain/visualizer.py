#!/bin/bash
# 3D image visualization
# Author: David Young, 2017
"""3D Visualization GUI.

This module is the main GUI for visualizing 3D objects from imaging stacks. 
The module can be run either as a script or loaded and initialized by 
calling main().

Examples:
    Launch the GUI with the given file at a particular size and offset::
        
        $ ./run --img /path/to/file.czi --offset 30,50,205 \
            --size 150,150,10
    
    Alternatively, this module can be run as a script::
        
        $ python -m clrbrain.visualizer --img /path/to/file.czi

Attributes:
    params: Additional Matplotlib rc parameters.
"""

import os
import sys
from time import time
import datetime

import numpy as np
from traits.api import (HasTraits, Instance, on_trait_change, Button, Float, 
                        Int, List, Array, Str, Dict, push_exception_handler, 
                        Property)
from traitsui.api import (View, Item, HGroup, VGroup, Handler, 
                          RangeEditor, HSplit, TabularEditor, CheckListEditor)
from traitsui.tabular_adapter import TabularAdapter
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
import matplotlib.pylab as pylab

from clrbrain import cli
from clrbrain import config
from clrbrain import importer
from clrbrain import detector
from clrbrain import plot_3d
from clrbrain import plot_2d
from clrbrain import sqlite
from clrbrain import chunking

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
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
    # cannot round to decimal places or else tuple will further round
    roi_size_um = np.around(np.multiply(roi_size, detector.resolutions[0][::-1]))
    title = ("{} (series {})\n"
             "offset {}, ROI size {}{}").format(os.path.basename(cli.filename), 
                                                cli.series, offset, 
                                                tuple(roi_size_um),
                                                u'\u00b5m')
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

class ListSelections(HasTraits):
    selections = List(["test0", "test1"])

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
    rois_selections_class = Instance(ListSelections)
    rois_check_list = Str
    _rois_dict = None
    _roi_default = "None selected"
    _rois = None
    _segments = Array
    _segs_scale_low = 0.0
    _segs_scale_high = Float # needs to be trait to dynamically update
    segs_scale = Float
    segs_pts = None
    segs_selected = List # indices
    segs_table = TabularEditor(adapter=SegmentsArrayAdapter(), multi_select=True, 
                               selected_row="segs_selected")
    segs_cmap = None
    segs_feedback = Str("Segments output")
    labels = None
    _check_list_3d = List
    _DEFAULTS_3D = ["Side panes", "Side circles", "No raw"]
    _check_list_2d = List
    _DEFAULTS_2D = ["Filtered", "Only save inside"]
    _planes_2d = List
    _DEFAULTS_PLANES_2D = ["xy", "xz"]
    _styles_2d = List
    _DEFAULTS_STYLES_2D = ["Square", "Multi-zoom"]
    
    def _format_seg(self, seg):
        """Formats the segment as a strong for feedback.
        
        Params:
            seg: The segment as an array of (z, row, column, radius).
        """
        seg_str = seg[0:3].astype(int).astype(str).tolist()
        seg_str.append(str(round(seg[3], 3)))
        seg_str.append(str(int(seg[4])))
        return ", ".join(seg_str)
    
    def _append_roi(self, roi, rois_dict):
        """Append an ROI to the ROI dictionary.
        
        Params:
            roi: The ROI to save.
            rois_dict: Dictionary of saved ROIs.
        """
        label = "offset ({},{},{}) of size ({},{},{})".format(roi["offset_x"], roi["offset_y"], 
                                         roi["offset_z"], roi["size_x"], 
                                         roi["size_y"], roi["size_z"])
        rois_dict[label] = roi
    
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
        curr_roi_size = self.roi_array[0].astype(int)
        print("Preparing to insert segments to database with border widths {}"
              .format(self.border))
        feedback = [ "Preparing segments:" ]
        for i in range(len(self.segments)):
            seg = self.segments[i]
            if seg[4] == -1 and np.isclose(seg[3], 0):
                # ignores user added segments, where radius assumed to be 0,
                # that are no longer selected
                feedback.append("ignoring unselected user added segment: {}".format(seg))
            else:
                if (seg[0] >= self.border[2] and seg[0] < (curr_roi_size[2] - self.border[2])
                    and seg[1] >= self.border[1] and seg[1] < (curr_roi_size[1] - self.border[1])
                    and seg[2] >= self.border[0] and seg[2] < (curr_roi_size[0] - self.border[0])):
                    # transposes segments within inner ROI to absolute coordinates
                    feedback.append("{} to insert".format(self._format_seg(seg)))
                    seg_db = np.array([seg[2] + self.x_offset, seg[1] + self.y_offset, 
                                       seg[0] + self.z_offset, seg[3], seg[4]])
                    segs_transposed.append(seg_db)
                else:
                    feedback.append("{} outside, ignored".format(self._format_seg(seg)))
        
        segs_transposed_np = np.array(segs_transposed)
        if np.any(np.logical_and(segs_transposed_np[:, 4] == -1, 
                  np.logical_not(np.isclose(segs_transposed_np[:, 3], 0)))):
            feedback.insert(0, "Segments *NOT* added. Please ensure that all "
                               "segments in the ROI have been verified.\n")
        else:
            # inserts experiment if not already added, then segments
            feedback.append("\nInserting segments:")
            for seg in segs_transposed:
                feedback.append("{} inserted".format(self._format_seg(seg)))
            exp_id = sqlite.select_or_insert_experiment(cli.conn, cli.cur, 
                                                        os.path.basename(cli.filename),
                                                        datetime.datetime(1000, 1, 1))
            roi_id, out = sqlite.select_or_insert_roi(cli.conn, cli.cur, exp_id, cli.series, 
                                       np.add(self._curr_offset(), self.border).tolist(), 
                                       np.subtract(curr_roi_size, np.multiply(self.border, 2)).tolist())
            sqlite.insert_blobs(cli.conn, cli.cur, roi_id, segs_transposed)
            roi = sqlite.select_roi(cli.cur, roi_id)
            self._append_roi(roi, self._rois_dict)
            self.rois_selections_class.selections = list(self._rois_dict.keys())
            feedback.append(out)
        feedback_str = "\n".join(feedback)
        print(feedback_str)
        self.segs_feedback = feedback_str
    
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
            self.roi = plot_3d.prepare_roi(cli.image5d, cli.channel, 
                                           curr_roi_size, curr_offset)
            self.roi = plot_3d.denoise(self.roi)
        else:
            print("loading from previously processed image")
            self.roi = plot_3d.prepare_roi(cli.image5d_proc, cli.channel, 
                                           curr_roi_size, curr_offset)
        # show raw points unless selected not to
        if self._DEFAULTS_3D[2] not in self._check_list_3d:
            '''# turned off sufrace visualization since so noisy
            if plot_3d.mlab_3d == plot_3d.MLAB_3D_TYPES[0]:
                plot_3d.plot_3d_surface(self.roi, self)
            else:
                plot_3d.plot_3d_points(self.roi, self)
            '''
            plot_3d.plot_3d_points(self.roi, self)
        else:
            self.scene.mlab.clf()
        
        # show shadow images around the points if selected
        if self._DEFAULTS_3D[0] in self._check_list_3d:
            plot_3d.plot_2d_shadows(self.roi, self)
        
        # reset segments
        self.segments = None
        self.segs_pts = None
        
    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        self._set_border(True)
        
        # dimension max values in pixels
        if cli.image5d_proc is not None:
            size = cli.image5d_proc.shape[0:3]
        else:
            size = cli.image5d.shape[1:4]
        self.z_high, self.y_high, self.x_high = size
        curr_offset = cli.offset
        # apply user-defined offsets
        if curr_offset is not None:
            self.x_offset = curr_offset[0]
            self.y_offset = curr_offset[1]
            self.z_offset = curr_offset[2]
        else:
            print("No offset, using standard one")
            curr_offset = self._curr_offset()
        self.roi_array[0] = ([100, 100, 15] if cli.roi_size is None 
                             else cli.roi_size)
        
        # set up selector for loading past saved ROIs
        self._rois_dict = {self._roi_default: None}
        exps = sqlite.select_experiment(cli.cur, os.path.basename(cli.filename))
        self.rois_selections_class = ListSelections()
        if len(exps) > 0:
            self._rois = sqlite.select_rois(cli.cur, exps[0]["id"])
            if len(self._rois) > 0:
                for roi in self._rois:
                    self._append_roi(roi, self._rois_dict)
        self.rois_selections_class.selections = list(self._rois_dict.keys())
        self.rois_check_list = self._roi_default
        
        # 2D plot options setup
        self._planes_2d = [self._DEFAULTS_PLANES_2D[0]]
        self._styles_2d = [self._DEFAULTS_STYLES_2D[0]]
        
        # show the default ROI
        self.show_3d()
    
    @on_trait_change('x_offset,y_offset,z_offset')
    def update_plot(self):
        """Shows the chosen offset when an offset slider is moved.
        """
        print("x: {}, y: {}, z: {}".format(self.x_offset, self.y_offset, 
                                           self.z_offset))
    
    def _btn_redraw_trait_fired(self):
        # ensure that cube dimensions don't exceed array
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
        self.scene.mlab.orientation_axes()
        print("view: {}\nroll: {}".format(self.scene.mlab.view(), self.scene.mlab.roll()))
    
    @on_trait_change("scene.activated")
    def _orient_camera(self):
        # default camera position after initiation
        view = self.scene.mlab.view(75, 140, 170)
        roll = self.scene.mlab.roll(-175)
        self.scene.mlab.orientation_axes()
        #self.scene.mlab.outline() # affects zoom after segmenting
        #self.scene.mlab.axes() # need to adjust units to microns
    
    def _btn_segment_trait_fired(self, segs=None):
        if plot_3d.mlab_3d == plot_3d.MLAB_3D_TYPES[0]:
            # segments using the Random-Walker algorithm
            self.labels, self.walker = detector.segment_rw(self.roi)
            self.segs_cmap = plot_3d.show_surface_labels(self.labels, self)
        else:
            # segments using blob detection
            if cli.segments_proc is None:
                # blob detects the ROI
                if config.process_settings["random_walker"]:
                    self.labels, self.roi = detector.segment_rw(self.roi)
                self.segments = detector.segment_blob(self.roi)
            else:
                x, y, z = self._curr_offset()
                # segs is 0 for some reason if none given
                if segs is None or not isinstance(segs, np.ndarray):
                    # uses blobs from loaded segments
                    roi_x, roi_y, roi_z = self.roi_array[0].astype(int)
                    # adds additional padding to show surrounding segments
                    pad = plot_2d.padding # human (x, y, z) order
                    segs = cli.segments_proc[np.all([cli.segments_proc[:, 0] >= z - pad[2], 
                                                     cli.segments_proc[:, 0] < z + roi_z + pad[2],
                                                     cli.segments_proc[:, 1] >= y - pad[1], 
                                                     cli.segments_proc[:, 1] < y + roi_y + pad[1],
                                                     cli.segments_proc[:, 2] >= x - pad[0], 
                                                     cli.segments_proc[:, 2] < x + roi_x + pad[0]],
                                                    axis=0)]
                    # transpose to make coordinates relative to offset
                    segs = np.copy(segs)
                self.segments = np.subtract(segs, (z, y, x, 0, 0))
            show_shadows = self._DEFAULTS_3D[1] in self._check_list_3d
            self.segs_pts, self.segs_cmap, scale = plot_3d.show_blobs(self.segments, self, 
                                                                      show_shadows)
            self._segs_scale_high = scale * 2
            self.segs_scale = scale
            #detector.show_blob_surroundings(self.segments, self.roi)
        self.scene.mlab.outline()
    
    @on_trait_change('segs_scale')
    def update_segs_scale(self):
        """Updates the glyph scale factor.
        """
        if self.segs_pts is not None:
            self.segs_pts.glyph.glyph.scale_factor = self.segs_scale
    
    def _btn_2d_trait_fired(self):
        # shows 2D plots
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        # update verify flag
        plot_2d.verify = self._DEFAULTS_2D[1] in self._check_list_2d
        img = cli.image5d
        roi = None
        if self._DEFAULTS_2D[0] in self._check_list_2d:
            print("showing processed 2D images")
            if cli.image5d_proc is not None:
                # used for both overview and ROI images
                img = cli.image5d_proc
                '''# TESTING: if need an ROI
                roi = plot_3d.prepare_roi(img, cli.channel, curr_roi_size, 
                                          curr_offset)
                '''
            else:
                # denoised ROI processed during 3D display
                roi = self.roi
        elif cli.image5d is None:
            print("loading original image stack from file")
            cli.image5d = importer.read_file(cli.filename, cli.series)
            img = cli.image5d
        if self._styles_2d[0] == self._DEFAULTS_STYLES_2D[1]:
            # Multi-zoom style
            plot_2d.plot_2d_stack(self, _fig_title(curr_offset, curr_roi_size), 
                                  img, cli.channel, curr_roi_size, 
                                  curr_offset, self.segments, self.segs_cmap, 
                                  self.border, self._planes_2d[0].lower(), 
                                  (0, 0, 0), 3, True, "middle", roi, 
                                  labels=self.labels)
        else:
            # defaults to Square style
            plot_2d.plot_2d_stack(self, _fig_title(curr_offset, curr_roi_size), 
                                  img, cli.channel, curr_roi_size, 
                                  curr_offset, self.segments, self.segs_cmap, 
                                  self.border, self._planes_2d[0].lower(), 
                                  roi=roi, labels=self.labels)
    
    def _btn_save_segments_fired(self):
        self.save_segs()
    
    @on_trait_change('rois_check_list')
    def update_roi(self):
        print("got {}".format(self.rois_check_list))
        if self.rois_check_list != self._roi_default:
            # get chosen ROI reconstruct original ROI size and offset including border
            roi = self._rois_dict[self.rois_check_list]
            cli.roi_size = (roi["size_x"], roi["size_y"], roi["size_z"])
            cli.roi_size = tuple(np.add(cli.roi_size, np.multiply(self.border, 2)).astype(int).tolist())
            self.roi_array[0] = cli.roi_size
            cli.offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
            cli.offset = tuple(np.subtract(cli.offset, self.border).astype(int).tolist())
            self.x_offset, self.y_offset, self.z_offset = cli.offset
            
            # redraw the original ROI and prepare verify mode
            self._btn_redraw_trait_fired()
            _, blobs = sqlite.select_blobs(cli.cur, roi["id"])
            self._btn_segment_trait_fired(segs=blobs)
            plot_2d.verify = True
        else:
            print("no roi found")
            plot_2d.verify = False
    
    @on_trait_change('_check_list_2d')
    def update_2d_options(self):
        if self._DEFAULTS_2D[1] in self._check_list_2d:
            self._set_border()
        else:
            self._set_border(True)
    
    def _curr_offset(self):
        return (self.x_offset, self.y_offset, self.z_offset)
    
    def _set_border(self, reset=False):
        # TODO: change from (x, y, z) order?
        if reset:
            self.border = np.zeros(3)
            print("set border to zeros")
        else:
            self.border = np.ceil(np.multiply(detector.calc_scaling_factor(), 
                                              chunking.overlap_factor))[::-1]
        print("set border to {}".format(self.border))
        
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
                height=600, width=600, show_label=False
            ),
            VGroup(
                VGroup(
                    Item("rois_check_list", 
                         editor=CheckListEditor(name="object.rois_selections_class.selections"),
                         label="ROIs"),
                    Item("roi_array", label="Size (x,y,z)"),
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
                Item(
                     "_check_list_3d", 
                     editor=CheckListEditor(values=_DEFAULTS_3D, cols=3), 
                     style="custom",
                     label="3D options"
                ),
                HGroup(
                    Item(
                         "_check_list_2d", 
                         editor=CheckListEditor(values=_DEFAULTS_2D, cols=2), 
                         style="custom",
                         label="2D options"
                    ),
                    Item(
                         "_planes_2d", 
                         editor=CheckListEditor(values=_DEFAULTS_PLANES_2D), 
                         style="simple",
                         label="Plane"
                    ),
                ),
                Item(
                     "_styles_2d", 
                     editor=CheckListEditor(values=_DEFAULTS_STYLES_2D), 
                     style="simple",
                     label="2D Styles"
                ),
                HGroup(
                    Item("btn_redraw_trait", show_label=False), 
                    Item("btn_segment_trait", show_label=False), 
                    Item("btn_2d_trait", show_label=False)
                ),
                Item(
                    "segs_scale",
                    editor=RangeEditor(
                        low_name="_segs_scale_low",
                        high_name="_segs_scale_high",
                        mode="slider"),
                ),
                VGroup(
                    Item(
                        "_segments",
                        editor=segs_table,
                        show_label=False
                    ),
                    Item("segs_feedback", style="custom", show_label=False),
                    Item("btn_save_segments", show_label=False)
                )
            )
        ),
        handler=VisHandler(),
        title="clrbrain",
        resizable=True
    )

if __name__ == "__main__":
    print("Starting visualizer...")
    main()
    