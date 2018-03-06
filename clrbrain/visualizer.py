#!/bin/bash
# 3D image visualization
# Author: David Young, 2017, 2018
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
                        Property, File)
from traitsui.api import (View, Item, HGroup, VGroup, Handler, 
                          RangeEditor, HSplit, TabularEditor, CheckListEditor, 
                          FileEditor)
from traitsui.tabular_adapter import TabularAdapter
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

from clrbrain import chunking
from clrbrain import cli
from clrbrain import config
from clrbrain import detector
from clrbrain import importer
from clrbrain import plot_3d
from clrbrain import plot_2d
from clrbrain import register
from clrbrain import sqlite

def main():
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    """
    cli.main()
    push_exception_handler(reraise_exceptions=True)
    visualization = Visualization()
    visualization.configure_traits()
    
def _fig_title(atlas_region, offset, roi_size):
    """Figure title parser.
    
    Arguments:
        atlas_region: Name of the region in the atlas; if None, the region
            will be ignored.
        offset: (x, y, z) image offset
        roi_size: (x, y, z) region of interest size
    
    Returns:
        Figure title string.
    """
    region = ""
    if atlas_region is not None:
        region = "{} from ".format(atlas_region)
    # cannot round to decimal places or else tuple will further round
    roi_size_um = np.around(
        np.multiply(roi_size, detector.resolutions[0][::-1]))
    return "{}{} (series {})\noffset {}, ROI size {} [{}{}]".format(
        region, os.path.basename(config.filename), config.series, offset, 
        tuple(roi_size), tuple(roi_size_um), u'\u00b5m')

class VisHandler(Handler):
    """Simple handler for Visualization object events.
    
    Closes the JVM when the window is closed.
    """
    def closed(self, info, is_ok):
        """Closes the Java VM when the GUI is closed.
        """
        importer.jb.kill_vm()
        config.db.conn.close()

class ListSelections(HasTraits):
    selections = List(["test0", "test1"])

class SegmentsArrayAdapter(TabularAdapter):
    columns = [("i", "index"), ("z", 0), ("row", 1), ("col", 2), 
               ("radius", 3), ("confirmed", 4), ("channel", 6), ("abs_z", 7), 
               ("abs_y", 8), ("abs_x", 9)]
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
    btn_segment_trait = Button("Detect")
    btn_2d_trait = Button("2D Plots")
    btn_save_segments = Button("Save Segments")
    roi = None # combine with roi_array?
    rois_selections_class = Instance(ListSelections)
    rois_check_list = Str
    _rois_dict = None
    _roi_default = "None selected"
    _rois = None
    _segments = Array
    _segs_moved = [] # orig seg of moved blobs to track for deletion
    _segs_scale_low = 0.0
    _segs_scale_high = Float # needs to be trait to dynamically update
    segs_scale = Float
    segs_pts = None
    segs_selected = List # indices
    # multi-select to allow updating with a list, but segment updater keeps
    # selections to single when updating them
    segs_table = TabularEditor(
        adapter=SegmentsArrayAdapter(), multi_select=True, 
        selected_row="segs_selected")
    segs_in_mask = None # boolean mask for segments in the ROI
    segs_cmap = None
    segs_feedback = Str("Segments output")
    labels = None # segmentation labels
    _check_list_3d = List
    _DEFAULTS_3D = ["Side panes", "Side circles", "Raw"]
    _check_list_2d = List
    _DEFAULTS_2D = ["Filtered", "Border zone", "Outline", "Grid"]
    _planes_2d = List
    _border_on = False # remembers last border selection
    _DEFAULT_BORDER = np.zeros(3) # default ROI border size
    _DEFAULTS_PLANES_2D = ["xy", "xz", "yz"]
    _circles_2d = List
    _styles_2d = List
    _DEFAULTS_STYLES_2D = [
        "Square no oblique", "Square with oblique", "Single row", "Wide ROI", "Multi-zoom"]
    _atlas_label = None
    _structure_scale = Int # ontology structure levels
    _structure_scale_low = -1
    _structure_scale_high = 20
    _mlab_title = None
    _scene_3d_shown = False # 3D Mayavi display shown
    _circles_window_opened = True # 2D plots with circles window opened
    _filename = File # file browser
    _channel = Int # channel number, 0-based
    _channel_low = -1 # -1 used for None, which translates to "all"
    _channel_high = 0
    
    def _format_seg(self, seg):
        """Formats the segment as a strong for feedback.
        
        Args:
            seg: The segment as an array of (z, row, column, radius).
        """
        seg_str = seg[0:3].astype(int).astype(str).tolist()
        seg_str.append(str(round(seg[3], 3)))
        seg_str.append(str(int(seg[4])))
        return ", ".join(seg_str)
    
    def _append_roi(self, roi, rois_dict):
        """Append an ROI to the ROI dictionary.
        
        Args:
            roi: The ROI to save.
            rois_dict: Dictionary of saved ROIs.
        """
        label = "offset ({},{},{}) of size ({},{},{})".format(
           roi["offset_x"], roi["offset_y"], roi["offset_z"], roi["size_x"], 
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
        segs_to_delete = []
        curr_roi_size = self.roi_array[0].astype(int)
        print("Preparing to insert segments to database with border widths {}"
              .format(self.border))
        feedback = [ "Preparing segments:" ]
        for i in range(len(self.segments)):
            seg = self.segments[i]
            # uses absolute coordinates from end of seg
            seg_db = self._seg_for_db(seg)
            if seg[4] == -1 and seg[3] < config.POS_THRESH:
                # attempts to delete user added segments, where radius assumed to be 0,
                # that are no longer selected
                feedback.append(
                    "{} to delete (unselected user added or explicitly deleted)"
                    .format(seg_db))
                segs_to_delete.append(seg_db)
            else:
                if (seg[0] >= self.border[2] and seg[0] < (curr_roi_size[2] - self.border[2])
                    and seg[1] >= self.border[1] and seg[1] < (curr_roi_size[1] - self.border[1])
                    and seg[2] >= self.border[0] and seg[2] < (curr_roi_size[0] - self.border[0])):
                    # transposes segments within inner ROI to absolute coordinates
                    feedback.append("{} to insert".format(self._format_seg(seg_db)))
                    segs_transposed.append(seg_db)
                else:
                    feedback.append("{} outside, ignored".format(self._format_seg(seg_db)))
        
        segs_transposed_np = np.array(segs_transposed)
        if (len(segs_transposed_np) > 0 
            and np.any(
                np.logical_and(segs_transposed_np[:, 4] == -1, 
                np.logical_not(segs_transposed_np[:, 3] < config.POS_THRESH)))):
            feedback.insert(0, "Segments *NOT* added. Please ensure that all "
                               "segments in the ROI have been verified.\n")
        else:
            # inserts experiment if not already added, then segments
            feedback.append("\nInserting segments:")
            for seg in segs_transposed:
                feedback.append(self._format_seg(seg))
            if len(segs_to_delete) > 0:
                feedback.append("\nDeleting segments:")
                for seg in segs_to_delete:
                    feedback.append(self._format_seg(seg))
            exp_id = sqlite.select_or_insert_experiment(config.db.conn, config.db.cur, 
                                                        os.path.basename(config.filename),
                                                        None)
            roi_id, out = sqlite.select_or_insert_roi(config.db.conn, config.db.cur, exp_id, config.series, 
                                       np.add(self._curr_offset(), self.border).tolist(), 
                                       np.subtract(curr_roi_size, np.multiply(self.border, 2)).tolist())
            sqlite.delete_blobs(config.db.conn, config.db.cur, roi_id, segs_to_delete)
            
            # delete the original entry of blobs that moved since replacement
            # is based on coordinates, so moved blobs wouldn't be replaced
            for i in range(len(self._segs_moved)):
                self._segs_moved[i] = self._seg_for_db(self._segs_moved[i])
            sqlite.delete_blobs(
                config.db.conn, config.db.cur, roi_id, self._segs_moved)
            self._segs_moved = []
            
            # insert blobs into DB and save ROI in GUI
            sqlite.insert_blobs(config.db.conn, config.db.cur, roi_id, segs_transposed)
            roi = sqlite.select_roi(config.db.cur, roi_id)
            self._append_roi(roi, self._rois_dict)
            self.rois_selections_class.selections = list(self._rois_dict.keys())
            feedback.append(out)
        feedback_str = "\n".join(feedback)
        print(feedback_str)
        self.segs_feedback = feedback_str
    
    def _reset_segments(self):
        """Resets the saved segments.
        """
        self.segments = None
        self.segs_pts = None
        self.segs_in_mask = None
        self._circles_window_opened = False
        self.segs_feedback = ""
    
    def _update_structure_level(self, curr_offset, curr_roi_size):
        self._atlas_label = None
        if self._mlab_title is not None:
            self._mlab_title.remove()
            self._mlab_title = None
        if (config.labels_ref_lookup is not None and curr_offset is not None 
            and curr_roi_size is not None):
            center = np.add(
                curr_offset, 
                np.around(np.divide(curr_roi_size, 2)).astype(np.int))
            level = self._structure_scale
            if level == self._structure_scale_high:
                level = None
            self._atlas_label = register.get_label(
                center[::-1], config.labels_img, config.labels_ref_lookup, 
                config.labels_scaling, level)
            if self._atlas_label is not None:
                title = register.get_label_name(self._atlas_label)
                if title is not None:
                    self._mlab_title = self.scene.mlab.title(title)
    
    def show_3d(self):
        """Shows the 3D plot.
        
        If the processed image flag is true ("proc=1"), the region will be
        taken from the saved processed array. Type of 3D display depends
        on the "3d" flag.
        """
        # ensure that cube dimensions don't exceed array
        curr_roi_size = self.roi_array[0].astype(int)
        roi_size_orig = np.copy(curr_roi_size)
        if curr_roi_size[0] + self.x_offset > self.x_high:
            curr_roi_size[0] = self.x_high - self.x_offset
            self.roi_array = [curr_roi_size]
        if curr_roi_size[1] + self.y_offset > self.y_high:
            curr_roi_size[1] = self.y_high - self.y_offset
            self.roi_array = [curr_roi_size]
        if curr_roi_size[2] + self.z_offset > self.z_high:
            curr_roi_size[2] = self.z_high - self.z_offset
            self.roi_array = [curr_roi_size]
        print("using ROI size of {}".format(self.roi_array[0].astype(int)))
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        if np.any(np.not_equal(curr_roi_size, roi_size_orig)):
            feedback_str = (
                "Unable to fit ROI of size {} at offset {} into "
                "image of size ({}, {}, {}) (x, y, z) so resized ROI to {}"
                .format(roi_size_orig, curr_offset, self.x_high, 
                        self.y_high, self.z_high, curr_roi_size))
            self.segs_feedback = feedback_str
        
        # show raw 3D image unless selected not to
        if self._DEFAULTS_3D[2] in self._check_list_3d:
            # show region of interest based on raw image, using basic denoising 
            # to normalize values but not fully processing
            self.roi = plot_3d.prepare_roi(
                cli.image5d, curr_roi_size, curr_offset)
            
            vis = (plot_3d.mlab_3d, config.process_settings["vis_3d"])
            if plot_3d.MLAB_3D_TYPES[0] in vis:
                # surface rendering
                plot_3d.plot_3d_surface(self.roi, self)
                _scene_3d_shown = True
            else:
                # 3D point rendering
                _scene_3d_shown = plot_3d.plot_3d_points(
                    self.roi, self, config.channel)
            
            # process ROI in prep for showing filtered 2D view and segmenting
            self.roi = plot_3d.saturate_roi(self.roi, channel=config.channel)
            self.roi = plot_3d.denoise_roi(self.roi, config.channel)
        
        else:
            self.scene.mlab.clf()
        
        # show shadow images around the points if selected
        if self._DEFAULTS_3D[0] in self._check_list_3d:
            plot_3d.plot_2d_shadows(self.roi, self)
        
        # show title from labels reference if available
        self._update_structure_level(curr_offset, curr_roi_size)
        
        self._reset_segments()
        
    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        self._set_border()
        
        # set up file parameters
        self._filename = config.filename
        if cli.image5d.ndim >= 5:
            self._channel_high = cli.image5d.shape[4] - 1
        self._channel = config.channel
        
        # dimension max values in pixels
        size = cli.image5d.shape[1:4]
        # TODO: consider subtracting 1 to avoid max offset being 1 above
        # true max, but currently convenient to display size and checked 
        # elsewhere
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
        self._rois = config.db.get_rois(os.path.basename(config.filename))
        self.rois_selections_class = ListSelections()
        if self._rois is not None and len(self._rois) > 0:
            for roi in self._rois:
                self._append_roi(roi, self._rois_dict)
        self.rois_selections_class.selections = list(self._rois_dict.keys())
        self.rois_check_list = self._roi_default
        
        # default options setup
        self._circles_2d = [plot_2d.CIRCLES[0]]
        self._planes_2d = [self._DEFAULTS_PLANES_2D[0]]
        self._styles_2d = [self._DEFAULTS_STYLES_2D[0]]
        self._check_list_2d = [self._DEFAULTS_2D[1]]
        self._check_list_3d = [self._DEFAULTS_3D[2]]
        #self._structure_scale = self._structure_scale_high
        
        # show the default ROI
        self.show_3d()
    
    @on_trait_change("_filename")
    def update_filename(self):
        """Update the selected filename.
        """
        config.filename = self._filename
        print("Changed filename to {}".format(config.filename))
    
    @on_trait_change("_channel")
    def update_channel(self):
        """Update the selected channel.
        """
        config.channel = self._channel
        if config.channel == -1:
            config.channel = None
        print("Changed channel to {}".format(config.channel))
    
    @on_trait_change("x_offset,y_offset,z_offset")
    def update_plot(self):
        """Shows the chosen offset when an offset slider is moved.
        """
        print("x: {}, y: {}, z: {}".format(self.x_offset, self.y_offset, 
                                           self.z_offset))
    
    '''
    @on_trait_change("roi_array")
    def _update_roi_array(self):
    '''
    
    @on_trait_change("_structure_scale")
    def _update_structure_scale(self):
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        self._update_structure_level(curr_offset, curr_roi_size)
    
    def _btn_redraw_trait_fired(self):
        self.show_3d()
        if self._scene_3d_shown:
            self.scene.mlab.orientation_axes()
        # updates the GUI here even though it doesn't elsewhere for some reason
        self.rois_check_list = self._roi_default
        #print("reset selected ROI to {}".format(self.rois_check_list))
        #print("view: {}\nroll: {}".format(
        #    self.scene.mlab.view(), self.scene.mlab.roll()))
    
    @on_trait_change("scene.activated")
    def _orient_camera(self):
        # default camera position after initiation, with distance based on 
        # ROI size
        view = self.scene.mlab.view(75, 140, np.max(self.roi_array[0]) * 3)
        roll = self.scene.mlab.roll(-175)
        if self._scene_3d_shown:
            self.scene.mlab.orientation_axes()
        #self.scene.mlab.outline() # affects zoom after segmenting
        #self.scene.mlab.axes() # need to adjust units to microns
        print("view: {}\nroll: {}".format(
            self.scene.mlab.view(), self.scene.mlab.roll()))
    
    def _is_segs_none(self, segs):
        """Checks if segs is equivalent to None.
        """
        # segs is 0 for some reason if no parameter given in fired trait
        return segs is None or not isinstance(segs, np.ndarray)
    
    def _btn_segment_trait_fired(self, segs=None):
        if plot_3d.mlab_3d == plot_3d.MLAB_3D_TYPES[0]:
            # segments using the Random-Walker algorithm
            # TODO: also check ProcessSettings and/or do away with mlab_3d flag
            self.labels, self.walker = detector.segment_rw(self.roi)
            self.segs_cmap = plot_3d.show_surface_labels(self.labels, self)
        else:
            if self._DEFAULTS_2D[2] in self._check_list_2d:
                # shows labels around segments with Random-Walker
                self.labels, _ = detector.segment_rw(self.roi)
            
            # collect segments in ROI and padding region, ensureing coordinates 
            # are relative to offset
            segs_all = None
            offset = self._curr_offset()
            roi_size = self.roi_array[0].astype(int)
            if cli.segments_proc is None:
                # on-the-fly blob detection, which includes border but not 
                # padding region; already in relative coordinates
                roi = self.roi
                if config.process_settings["thresholding"]:
                    # thresholds prior to blob detection
                    roi = plot_3d.threshold(roi)
                segs_all = detector.detect_blobs(roi, config.channel)
            else:
                # get all previously processed blobs in ROI plus additional 
                # padding region to show surrounding blobs
                segs_all, _ = detector.get_blobs_in_roi(
                    cli.segments_proc, offset, 
                    roi_size, plot_2d.padding)
                # shift coordinates to be relative to offset
                segs_all[:, :3] = np.subtract(segs_all[:, :3], offset[::-1])
            print("segs_all:\n{}".format(segs_all))
            
            if not self._is_segs_none(segs):
                # if segs provided (eg from ROI), use only these segs within 
                # the ROI and add segs from the padding area outside the ROI
                _, segs_in_mask = detector.get_blobs_in_roi(
                    segs_all, np.zeros(3), 
                    roi_size, np.multiply(self.border, -1))
                segs_outside = segs_all[np.logical_not(segs_in_mask)]
                print("segs_outside:\n{}".format(segs_outside))
                segs[:, :3] = np.subtract(segs[:, :3], offset[::-1])
                segs_all = np.concatenate((segs, segs_outside), axis=0)
                
            # convert segments to visualizer table format and plot
            self.segments = detector.shift_blob_abs_coords(
                segs_all, offset[::-1])
            show_shadows = self._DEFAULTS_3D[1] in self._check_list_3d
            _, self.segs_in_mask = detector.get_blobs_in_roi(
                self.segments, np.zeros(3), 
                roi_size, np.multiply(self.border, -1))
            self.segs_pts, self.segs_cmap, scale = plot_3d.show_blobs(
                self.segments, self.scene.mlab, self.segs_in_mask, show_shadows)
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
        # prevent showing duplicate windows with selectable circles
        circles = self._circles_2d[0].lower()
        if circles != plot_2d.CIRCLES[2].lower():
            if self._circles_window_opened:
                self.segs_feedback = (
                    "Cannot show 2D plots while another plot "
                    "with circles in showing. Please redraw.")
                return
            else:
                self._circles_window_opened = True
        self.segs_feedback = ""
        
        # shows 2D plots
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        # update verify flag
        plot_2d.verify = self._DEFAULTS_2D[1] in self._check_list_2d
        img = cli.image5d
        roi = None
        if self._DEFAULTS_2D[0] in self._check_list_2d:
            print("showing processed 2D images")
            # denoised ROI processed during 3D display
            roi = self.roi
            if config.process_settings["thresholding"]:
                # thresholds prior to blob detection
                roi = plot_3d.threshold(roi)
        elif cli.image5d is None:
            print("loading original image stack from file")
            cli.image5d = importer.read_file(config.filename, config.series)
            img = cli.image5d
        
        blobs_truth_roi = None
        if config.truth_db is not None:
            # collect truth blobs from the truth DB if available
            blobs_truth_roi, _ = detector.get_blobs_in_roi(
                config.truth_db.blobs_truth, curr_offset, curr_roi_size, 
                plot_2d.padding)
            transpose = np.zeros(blobs_truth_roi.shape[1])
            transpose[0:3] = curr_offset[::-1]
            blobs_truth_roi = np.subtract(blobs_truth_roi, transpose)
            blobs_truth_roi[:, 5] = blobs_truth_roi[:, 4]
            #print("blobs_truth_roi:\n{}".format(blobs_truth_roi))
        title = _fig_title(register.get_label_name(self._atlas_label), 
                           curr_offset, curr_roi_size)
        filename_base = importer.filename_to_base(config.filename, config.series)
        grid = self._DEFAULTS_2D[3] in self._check_list_2d
        screenshot = self.scene.mlab.screenshot(antialiased=True)
        stack_args = (
            self.update_segment, title, filename_base, img, config.channel, curr_roi_size, 
            curr_offset, self.segments, self.segs_in_mask, self.segs_cmap, self._full_border(self.border), 
            self._planes_2d[0].lower())
        stack_args_named = {
            "roi": roi, "labels": self.labels, "blobs_truth": blobs_truth_roi, 
            "circles": circles, "grid": grid}
        if self._styles_2d[0] == self._DEFAULTS_STYLES_2D[1]:
            # Square style with oblique view
            plot_2d.plot_2d_stack(*stack_args, **stack_args_named, mlab_screenshot=screenshot)
        elif self._styles_2d[0] == self._DEFAULTS_STYLES_2D[2]:
            # single row
            plot_2d.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=3, single_zoom_row=True, 
                z_level=plot_2d.Z_LEVELS[1], mlab_screenshot=screenshot)
        elif self._styles_2d[0] == self._DEFAULTS_STYLES_2D[3]:
            # wide ROI
            plot_2d.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=2, mlab_screenshot=None, zoom_cols=7)
        elif self._styles_2d[0] == self._DEFAULTS_STYLES_2D[4]:
            # rulti-zoom style
            plot_2d.plot_2d_stack(*stack_args, **stack_args_named, zoom_levels=5, mlab_screenshot=None)
        else:
            # defaults to Square style without oblique view
            plot_2d.plot_2d_stack(*stack_args, **stack_args_named, zoom_levels=3, mlab_screenshot=None)
    
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
            self.roi_array = [cli.roi_size]
            cli.offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
            cli.offset = tuple(np.subtract(cli.offset, self.border).astype(int).tolist())
            self.x_offset, self.y_offset, self.z_offset = cli.offset
            
            # redraw the original ROI and prepare verify mode
            self.show_3d()
            if self._scene_3d_shown:
                self.scene.mlab.orientation_axes()
            blobs = sqlite.select_blobs(config.db.cur, roi["id"])
            self._btn_segment_trait_fired(segs=blobs)
            plot_2d.verify = True
        else:
            print("no roi found")
            plot_2d.verify = False
    
    @on_trait_change('_check_list_2d')
    def update_2d_options(self):
        border_checked = self._DEFAULTS_2D[1] in self._check_list_2d
        if border_checked != self._border_on:
            # change the border dimensions
            if border_checked:
                self._set_border()
            else:
                self._set_border(reset=True)
            # any change to border flag resets ROI selection and segments
            print("changed Border flag")
            self._border_on = border_checked
            self.rois_check_list = self._roi_default
            self._reset_segments()
    
    def _curr_offset(self):
        return (self.x_offset, self.y_offset, self.z_offset)
    
    def _full_border(self, border=None):
        """Gets the full border array, typically based on 
            :meth:`chunking.cal_overlap`.
        
        Args:
            border: If equal to :const:`_DEFAULT_BORDER`, returns the border. 
                Defaults to None.
        Returns:
            The full boundary in (x, y, z) order.
        """
        if border is not None and np.array_equal(border, self._DEFAULT_BORDER):
            return border
        return chunking.calc_overlap()[::-1]
    
    def _set_border(self, reset=False):
        """Sets the border as an (x, y, z) Numpy array, changing the final
            (z) dimension to be 0 since additional border planes will be shown 
            separately.
        
        Args:
            reset: If true, resets the border to :const:`_DEFAULT_BORDER`.
        """
        # TODO: change from (x, y, z) order?
        if reset:
            self.border = np.copy(self._DEFAULT_BORDER)
            print("set border to zeros")
        else:
            self.border = self._full_border()
            self.border[2] = 0 # ignore z
        print("set border to {}".format(self.border))
    
    def _add_segment(self, seg, offset):
        """Formats a segment to the specification used within this module and 
        adds the segment to this class object's segments list.
        
        Args:
            seg: Segment in (z, y, x, rad, confirmed, truth) format.
            offset: Offset in (x, y, z) format for consistency with 
                offset values given as user input.
        
        Returns:
            Segment in (z, y, x, rad, confirmed, truth, abs_z, abs_y, abs_x) 
            format.
        """
        print(seg)
        seg = np.array([detector.shift_blob_abs_coords(seg, offset[::-1])])
        print("added segment: {}".format(seg))
        # concatenate for in-place array update, though append
        # and re-assigning also probably works
        print(self.segments)
        if self.segments is None or len(self.segments) == 0:
            # copy since the object may be changed elsewhere; cast to float64 
            # since original type causes an incorrect database insertion 
            # for some reason
            self.segments = np.copy(seg).astype(np.float64)
        else:
            self.segments = np.concatenate((self.segments, seg))
        #print("segs:\n{}".format(self.segments))
        return seg
    
    def _seg_for_db(self, seg):
        """Convert segment output from the format used within this module 
        to that used in :module:`sqlite`, where coordinates are absolute 
        rather than relative to the offset.
        
        Args:
            seg: Segment in 
                (z, y, x, rad, confirmed, truth, abs_z, abs_y, abs_x) format.
        
        Returns:
            Segment in (abs_z, abs_y, abs_x, rad, confirmed, truth) format.
        """
        return np.array([*seg[6:9], *seg[3:6]])
    
    def _get_vis_segments_index(self, segment):
        # must take from vis rather than saved copy in case user 
        # manually updates the table
        #print("segs:\n{}".format(self.segments))
        #print("seg: {}".format(segment))
        #print(self.segments == segment)
        segi = np.where((self.segments == segment).all(axis=1))
        if len(segi) > 0:
            return segi[0][0]
        return -1
    
    def _force_seg_refresh(self, i, show=False):
       """Trigger table update by either selecting and reselected the segment
       or vice versa.
       
       Args:
           i: The element in vis.segs_selected, which is simply an index to
              the segment in vis.segments.
       """
       if i in self.segs_selected:
           self.segs_selected.remove(i)
           self.segs_selected.append(i)
       else:
           self.segs_selected.append(i)
           if not show:
               self.segs_selected.remove(i)
    
    def update_segment(self, segment_new, segment_old=None, offset=None, 
                       remove=False):
        """Update this class object's segments list with a new or updated 
        segment.
        
        Args:
            segment_new: Segment that was either added or updated, including 
                changes to coordinates or radius. Segments are generally 
                given as an (z, y, x, radius, confirmed, truth, ...) array, 
                where any elements after these ones are ignored.
            segment_old: Previous version of the segment; defaults to None, 
                in which case ``segment_new`` will only be added rather than 
                any previously segment updated.
            offset: Offset for ``segment_new``'s coordinates, used only 
                when adding a completely new segment. This segment's 
                coordinates are given in relative coordinates, and extra 
                fields will be appended to the segment to store its 
                absolute coordinates.
            remove: True if the segment should be removed, in which case 
                ``segment_old`` and ``offset`` will be ignored. Defaults to 
                False.
        
        Returns:
            The updated segment in 
            (z, y, x, rad, confirmed, truth, abs_z, abs_y, abs_x) format.
        """
        seg = segment_new
        # remove all row selections to ensure that no more than one 
        # row is selected by the end
        while len(self.segs_selected) > 0:
            self.segs_selected.pop()
        #print("updating: ", segment_new, offset)
        if remove:
            # remove segments, changing radius and confirmation values to 
            # flag for deletion from database while saving the ROI
            segi = self._get_vis_segments_index(segment_new)
            seg = self.segments[segi]
            seg[3] = -1 * abs(seg[3])
            detector.update_blob_confirmed(seg, -1)
            self._force_seg_refresh(segi, show=True)
            #self.segments = np.delete(self.segments, segi, 0)
            #seg = None
        elif segment_old is not None:
            # updates an existing segment
            self._segs_moved.append(segment_old)
            diff = np.subtract(segment_new[:3], segment_old[:3])
            detector.shift_blob_abs_coords(segment_new, diff)
            segi = self._get_vis_segments_index(segment_old)
            if segi != -1:
                self.segments[segi] = segment_new
                print("updated seg: {}".format(segment_new))
                self._force_seg_refresh(segi, show=True)
        elif offset is not None:
            # adds a new segment with the given offset
            seg = self._add_segment(segment_new, offset)[0]
            self.segs_selected.append(len(self.segments) - 1)
        return seg
    
    @property
    def segments(self):
        return self._segments
    
    @segments.setter
    def segments(self, val):
        """Sets segments.
        
        Args:
            val: Numpy array of (n, 9) shape with segments. The columns
                correspond to (z, y, x, radius, confirmed, truth, abs_z,
                abs_y, abs_x). Note that the "abs" values are different from 
                those used for duplicate shifting.
        """
        # no longer need to give default value if None, presumably from 
        # update of TraitsUI from 5.1.0 to 5.2.0pre
        if val is not None:
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
                    HGroup(
                        Item("_filename", 
                             editor=FileEditor(entries=10, allow_dir=False), 
                             label="File", style="simple"),
                        Item(
                            "_channel",
                            editor=RangeEditor(
                                low_name="_channel_low",
                                high_name="_channel_high",
                                mode="spinner")
                        ),
                    ),
                    Item("rois_check_list", 
                         editor=CheckListEditor(
                             name="object.rois_selections_class.selections"),
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
                         editor=CheckListEditor(values=_DEFAULTS_2D, cols=1), 
                         style="custom",
                         label="2D options"
                    ),
                    VGroup(
                        Item(
                             "_circles_2d", 
                             editor=CheckListEditor(values=plot_2d.CIRCLES), 
                             style="simple",
                             label="Circles"
                        ),
                        Item(
                             "_planes_2d", 
                             editor=CheckListEditor(values=_DEFAULTS_PLANES_2D), 
                             style="simple",
                             label="Plane"
                        ),
                        Item(
                             "_styles_2d", 
                             editor=CheckListEditor(values=_DEFAULTS_STYLES_2D), 
                             style="simple",
                             label="2D styles"
                        ),
                    ),
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
                Item(
                    "_structure_scale",
                    editor=RangeEditor(
                        low_name="_structure_scale_low",
                        high_name="_structure_scale_high",
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
    