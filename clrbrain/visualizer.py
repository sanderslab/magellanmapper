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
                          FileEditor, TextEditor)
from traitsui.tabular_adapter import TabularAdapter
from tvtk.pyface.scene_editor import SceneEditor
from tvtk.pyface.scene_model import SceneModelError
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene

from clrbrain import atlas_editor
from clrbrain import chunking
from clrbrain import cli
from clrbrain import config
from clrbrain import detector
from clrbrain import importer
from clrbrain import lib_clrbrain
from clrbrain import plot_3d
from clrbrain import plot_2d
from clrbrain import register
from clrbrain import sqlite


_ROI_DEFAULT = "None selected"

def main():
    """Starts the visualization GUI.
    
    Processes command-line arguments.
    """
    cli.main()
    plot_2d.setup_style()
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
    selections = List([_ROI_DEFAULT])

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
    btn_save_segments = Button("Save")
    roi = None # combine with roi_array?
    rois_selections_class = Instance(ListSelections)
    rois_check_list = Str
    _rois_dict = None
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
    atlas_ed = None # atlas editor
    _check_list_3d = List
    _DEFAULTS_3D = ["Side panes", "Side circles", "Raw", "Surface"]
    _check_list_2d = List
    _DEFAULTS_2D = [
        "Filtered", "Border zone", "Segmentation", "Grid", "Max inten proj"]
    _planes_2d = List
    _border_on = False # remembers last border selection
    _DEFAULT_BORDER = np.zeros(3) # default ROI border size
    _DEFAULTS_PLANES_2D = ["xy", "xz", "yz"]
    _circles_2d = List
    _styles_2d = List
    _DEFAULTS_STYLES_2D = [
        "Square ROI", "Square ROI with 3D", "Single row", "Wide ROI", 
        "Multi-zoom", "Thin rows", "Atlas editor"]
    _atlas_label = None
    _structure_scale = Int # ontology structure levels
    _structure_scale_low = -1
    _structure_scale_high = 20
    _region_id = Int
    _mlab_title = None
    _scene_3d_shown = False # 3D Mayavi display shown
    _circles_opened_type = None # type of 2D plots windows curr open
    _opened_window_style = None # 2D plots window style curr open
    _filename = File # file browser
    _channel = Int # channel number, 0-based
    _channel_low = -1 # -1 used for None, which translates to "all"
    _channel_high = 0
    _img_region = None
    
    def _format_seg(self, seg):
        """Formats the segment as a strong for feedback.
        
        Args:
            seg: The segment as an array given by 
                :func:``detector.blob_for_db``.
        
        Returns:
            Segment formatted as a string.
        """
        seg_str = seg[0:3].astype(int).astype(str).tolist()
        seg_str.append(str(round(seg[3], 3)))
        seg_str.append(str(int(detector.get_blob_confirmed(seg))))
        seg_str.append(str(int(detector.get_blob_channel(seg))))
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
        print("segments", self.segments)
        if self.segments is None or self.segments.size < 1:
            feedback_str = "No segments found to save"
            print(feedback_str)
            self.segs_feedback = feedback_str
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
            seg_db = detector.blob_for_db(seg)
            if seg[4] == -1 and seg[3] < config.POS_THRESH:
                # attempts to delete user added segments, where radius assumed 
                # to be 0,that are no longer selected
                feedback.append(
                    "{} to delete (unselected user added or explicitly deleted)"
                    .format(seg_db))
                segs_to_delete.append(seg_db)
            else:
                if (seg[0] >= self.border[2] 
                    and seg[0] < (curr_roi_size[2] - self.border[2])
                    and seg[1] >= self.border[1] 
                    and seg[1] < (curr_roi_size[1] - self.border[1])
                    and seg[2] >= self.border[0] 
                    and seg[2] < (curr_roi_size[0] - self.border[0])):
                    # transposes segments within inner ROI to absolute coords
                    feedback.append(
                        "{} to insert".format(self._format_seg(seg_db)))
                    segs_transposed.append(seg_db)
                else:
                    feedback.append(
                        "{} outside, ignored".format(self._format_seg(seg_db)))
        
        segs_transposed_np = np.array(segs_transposed)
        unverified = None
        if (len(segs_transposed_np) > 0):
            # unverified blobs are those with default confirmation setting 
            # and radius > 0, where radii < 0 would indicate a user-added circle
            unverified = np.logical_and(
                detector.get_blob_confirmed(segs_transposed_np) == -1, 
                np.logical_not(segs_transposed_np[:, 3] < config.POS_THRESH))
        if np.any(unverified):
            # show missing verifications
            feedback.insert(0, "**WARNING** Please check these "
                               "blobs' missing verifications:\n{}\n"
                               .format(segs_transposed_np[unverified, :4]))
        # inserts experiment if not already added, then segments
        feedback.append("\nInserting segments:")
        for seg in segs_transposed:
            feedback.append(self._format_seg(seg))
        if len(segs_to_delete) > 0:
            feedback.append("\nDeleting segments:")
            for seg in segs_to_delete:
                feedback.append(self._format_seg(seg))
        exp_id = sqlite.select_or_insert_experiment(
            config.db.conn, config.db.cur, os.path.basename(config.filename),
            None)
        roi_id, out = sqlite.select_or_insert_roi(
            config.db.conn, config.db.cur, exp_id, config.series, 
            np.add(self._curr_offset(), self.border).tolist(), 
            np.subtract(curr_roi_size, np.multiply(self.border, 2)).tolist())
        sqlite.delete_blobs(
            config.db.conn, config.db.cur, roi_id, segs_to_delete)
        
        # delete the original entry of blobs that moved since replacement
        # is based on coordinates, so moved blobs wouldn't be replaced
        for i in range(len(self._segs_moved)):
            self._segs_moved[i] = detector.blob_for_db(self._segs_moved[i])
        sqlite.delete_blobs(
            config.db.conn, config.db.cur, roi_id, self._segs_moved)
        self._segs_moved = []
        
        # insert blobs into DB and save ROI in GUI
        sqlite.insert_blobs(
            config.db.conn, config.db.cur, roi_id, segs_transposed)
        roi = sqlite.select_roi(config.db.cur, roi_id)
        self._append_roi(roi, self._rois_dict)
        self.rois_selections_class.selections = list(self._rois_dict.keys())
        feedback.append(out)
        feedback_str = "\n".join(feedback)
        print(feedback_str)
        self.segs_feedback = feedback_str
    
    def save_atlas(self):
        register.load_registered_img(
            config.filename, reg_name=register.IMG_LABELS, 
            replace=config.labels_img)
        self.segs_feedback = "Saved labels image at {}".format(
            datetime.datetime.now())
    
    def _reset_segments(self):
        """Resets the saved segments.
        """
        self.segments = None
        self.segs_pts = None
        self.segs_in_mask = None
        self.labels = None
        # window with circles may still be open but would lose segments 
        # table and be unsavable anyway; TODO: warn before resetting segments
        self._circles_opened_type = None
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
                config.labels_scaling, level, rounding=True)
            if self._atlas_label is not None:
                title = register.get_label_name(self._atlas_label)
                if title is not None:
                    self._mlab_title = self.scene.mlab.title(title)
    
    def _post_3d_display(self, title="clrbrain3d"):
        """Show axes and saved ROI parameters after 3D display.
        
        Args:
            title: Path without extension to save file if 
                :attr:``config.savefig`` is set to an extension. Defaults to 
                "clrbrain3d".
        """
        if self._scene_3d_shown:
            if config.savefig:
                path = "{}.{}".format(title, config.savefig)
                lib_clrbrain.backup_file(path)
                try:
                    # save before setting any other objects to avoid VTK 
                    # render error
                    self.scene.mlab.savefig(path)
                except SceneModelError as e:
                    # the scene may not have been activated yet
                    print("unable to save 3D surface")
            self.scene.mlab.orientation_axes()
        # updates the GUI here even though it doesn't elsewhere for some reason
        self.rois_check_list = _ROI_DEFAULT
        self._img_region = None
        #print("reset selected ROI to {}".format(self.rois_check_list))
        #print("view: {}\nroll: {}".format(
        #    self.scene.mlab.view(), self.scene.mlab.roll()))
    
    def show_3d(self):
        """Show the 3D plot and prepare for detections.
        
        Type of 3D display depends on configuration settings. A lightly 
        preprocessed image will be displayed in 3D, and afterward the 
        ROI will undergo full preprocessing in preparation for detection 
        and 2D filtered displays steps.
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
            
            if self._DEFAULTS_3D[3] in self._check_list_3d:
                # surface rendering
                plot_3d.plot_3d_surface(
                    self.roi, self.scene.mlab, config.channel)
                self._scene_3d_shown = True
            else:
                # 3D point rendering
                self._scene_3d_shown = plot_3d.plot_3d_points(
                    self.roi, self.scene.mlab, config.channel)
            
            # process ROI in prep for showing filtered 2D view and segmenting
            if not lib_clrbrain.is_binary(self.roi):
                self.roi = plot_3d.saturate_roi(
                    self.roi, channel=config.channel)
                self.roi = plot_3d.denoise_roi(self.roi, config.channel)
            else:
                lib_clrbrain.printv(
                    "binary image detected, will not preprocess")
        
        else:
            self.scene.mlab.clf()
        
        # show shadow images around the points if selected
        if self._DEFAULTS_3D[0] in self._check_list_3d:
            plot_3d.plot_2d_shadows(self.roi, self)
        
        # show title from labels reference if available
        self._update_structure_level(curr_offset, curr_roi_size)
        
        self._reset_segments()
    
    def show_label_3d(self, label_id):
        """Show 3D region of main image corresponding to label ID.
        
        Args:
            label_id: ID of label to display.
        """
        # get bounding box for label region
        bbox = plot_3d.get_label_bbox(config.labels_img, label_id)
        if bbox is None: return
        shape, slices = plot_3d.get_bbox_region(
            bbox, 10, config.labels_img.shape)
        
        # update GUI dimensions
        self.roi_array = [shape[::-1]] # TODO: avoid decimal point
        self.z_offset, self.y_offset, self.x_offset = [
            slices[i].start for i in range(len(slices))]
        self._scene_3d_shown = True
        
        # show main image corresponding to label region
        label_mask = config.labels_img[tuple(slices)] == label_id
        self.roi = np.copy(cli.image5d[0][slices])
        self.roi[~label_mask] = 0
        plot_3d.plot_3d_surface(self.roi, self.scene.mlab, config.channel)
        #plot_3d.plot_3d_points(self.roi, self.scene.mlab, config.channel)
        self._post_3d_display(title="label3d_{}".format(label_id))
    
    def _setup_for_image(self):
        """Setup GUI parameters for the loaded image5d.
        """
        # set up channel spinner based on number of channels available
        if cli.image5d.ndim >= 5:
            # increase max channels based on channel dimension
            self._channel_high = cli.image5d.shape[4] - 1
        else:
            # only one channel available
            self._channel_low = 0
        # None channel defaults to all channels, represented in the channel 
        # spinner here by -1 
        self._channel = -1 if config.channel is None else config.channel
        
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
        self._rois_dict = {_ROI_DEFAULT: None}
        self._rois = config.db.get_rois(os.path.basename(config.filename))
        self.rois_selections_class = ListSelections()
        if self._rois is not None and len(self._rois) > 0:
            for roi in self._rois:
                self._append_roi(roi, self._rois_dict)
        self.rois_selections_class.selections = list(self._rois_dict.keys())
        self.rois_check_list = _ROI_DEFAULT
        
        # show the default ROI
        settings = config.process_settings
        self.show_3d()
    
    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)
        
        # default options setup
        self._set_border()
        self._circles_2d = [plot_2d.CIRCLES[0]]
        self._planes_2d = [self._DEFAULTS_PLANES_2D[0]]
        self._styles_2d = [self._DEFAULTS_STYLES_2D[0]]
        self._check_list_2d = [self._DEFAULTS_2D[1]]
        self._check_list_3d = [self._DEFAULTS_3D[2]]
        if (config.process_settings["vis_3d"].lower() 
            == self._DEFAULTS_3D[3].lower()):
            # check "surface" if set in profile
            self._check_list_3d.append(self._DEFAULTS_3D[3])
        #self._structure_scale = self._structure_scale_high
        
        # setup interface for image
        # TODO: show the currently loaded Numpy image file without triggering 
        # update
        #self._filename = config.filename
        self._setup_for_image()
    
    @on_trait_change("_filename")
    def update_filename(self):
        """Update the selected filename and load the corresponding image.
        
        Since an original (eg .czi) image can be processed in so many 
        different ways, assume that the user will select the Numpy image 
        file instead of the raw image. Image settings will be constructed 
        from the Numpy image filename. Processed files (eg ROIs, blobs) 
        will not be loaded for now.
        """
        filename, series = importer.deconstruct_np_filename(self._filename)
        if filename is not None and series is not None:
            config.filename = filename
            config.series = series
            print("Changed filename to {}, series to {}"
                  .format(config.filename, config.series))
            # TODO: consider loading processed images, blobs, etc
            cli.image5d = importer.read_file(config.filename, config.series)
            self._setup_for_image()
        else:
            print("Could not parse filename {} and series {}"
                  .format(filename, series))
    
    @on_trait_change("_channel")
    def update_channel(self):
        """Update the selected channel, resetting the current state to 
        prevent displaying the old channel.
        """
        config.channel = self._channel
        if config.channel == -1:
            config.channel = None
        self.rois_check_list = _ROI_DEFAULT
        self._reset_segments()
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
        self._post_3d_display()
    
    @on_trait_change("scene.activated")
    def _orient_camera(self):
        # default camera position after initiation, with distance based on 
        # ROI size and further zoomed out based on any isotropic factor resizing
        zoom_out = 3
        isotropic_factor = config.process_settings["isotropic_vis"]
        if isotropic_factor is not None:
            # only use max dimension since this factor seems to influence the 
            # overall zoom the most
            zoom_out *= np.amax(isotropic_factor)
        view = self.scene.mlab.view(
            75, 140, np.max(self.roi_array[0]) * zoom_out)
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
                    cli.segments_proc, offset, roi_size, plot_2d.padding)
                # shift coordinates to be relative to offset
                segs_all[:, :3] = np.subtract(segs_all[:, :3], offset[::-1])
                segs_all = detector.format_blobs(segs_all)
                segs_all = detector.blobs_in_channel(
                    segs_all, config.channel)
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
                segs = detector.format_blobs(segs)
                segs = detector.blobs_in_channel(segs, config.channel)
                segs_all = np.concatenate((segs, segs_outside), axis=0)
                
            if segs_all is not None:
                # convert segments to visualizer table format and plot
                self.segments = detector.shift_blob_abs_coords(
                    segs_all, offset[::-1])
                show_shadows = self._DEFAULTS_3D[1] in self._check_list_3d
                _, self.segs_in_mask = detector.get_blobs_in_roi(
                    self.segments, np.zeros(3), 
                    roi_size, np.multiply(self.border, -1))
                self.segs_pts, self.segs_cmap, scale = plot_3d.show_blobs(
                    self.segments, self.scene.mlab, self.segs_in_mask, 
                    show_shadows)
                self._segs_scale_high = scale * 2
                self.segs_scale = scale
            
            if self._DEFAULTS_2D[2] in self._check_list_2d:
                blobs = self.segments[self.segs_in_mask]
                '''
                # 3D-seeded watershed segmentation using detection blobs
                self.labels, walker = detector.segment_rw(
                    self.roi, config.channel, erosion=1)
                self.labels = detector.segment_ws(self.roi, walker, blobs)
                '''
                # 3D-seeded random-walker with high beta to limit walking 
                # into background, also removing objects smaller than the 
                # smallest blob, roughly normalized for anisotropy and 
                # reduced by not including the 4/3 factor
                min_size = int(
                    np.pi * np.power(np.amin(np.abs(blobs[:, 3])), 3) 
                    / np.mean(plot_3d.calc_isotropic_factor(1)))
                print("min size threshold for r-w: {}".format(min_size))
                self.labels, walker = detector.segment_rw(
                    self.roi, config.channel, beta=5000, 
                    blobs=blobs, remove_small=min_size)
            #detector.show_blob_surroundings(self.segments, self.roi)
        self.scene.mlab.outline()
    
    @on_trait_change('segs_scale')
    def update_segs_scale(self):
        """Updates the glyph scale factor.
        """
        if self.segs_pts is not None:
            self.segs_pts.glyph.glyph.scale_factor = self.segs_scale
    
    def _fig_close_listener(self, evt):
        """Handle figure close events.
        """
        self._circles_opened_type = None
        self._opened_window_style = None
        circles = self._circles_2d[0].lower()
        if circles == plot_2d.CIRCLES[3].lower():
            # reset if in full annotation mode to avoid further duplicating 
            # circles, saving beforehand to prevent loss from premature  
            # window closure
            self.save_segs()
            self._reset_segments()
            self._circles_2d = [plot_2d.CIRCLES[0]]
            self.segs_feedback = "Reset circles after saving full annotations"
    
    def _btn_2d_trait_fired(self):
        if (self._circles_opened_type 
            and self._circles_opened_type != plot_2d.CIRCLES[2].lower()
            or self._opened_window_style == self._DEFAULTS_STYLES_2D[6]):
            # prevent multiple editable windows from being opened 
            # simultaneously to avoid unsynchronized state
            self.segs_feedback = (
                "Cannot show 2D plots while another editable "
                "plot is showing. Please redraw.")
            return
        circles = self._circles_2d[0].lower()
        if (not self._circles_opened_type 
            or self._circles_opened_type == plot_2d.CIRCLES[2].lower()):
            # set opened window type if not already set or non-editable window
            self._circles_opened_type = circles
        self._opened_window_style = self._styles_2d[0]
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
        filename_base = importer.filename_to_base(
            config.filename, config.series)
        grid = self._DEFAULTS_2D[3] in self._check_list_2d
        max_intens_proj = self._DEFAULTS_2D[4] in self._check_list_2d
        stack_args = (
            self.update_segment, title, filename_base, img, config.channel, 
            curr_roi_size, curr_offset, self.segments, self.segs_in_mask, 
            self.segs_cmap, self._fig_close_listener, 
            # additional args with defaults
            self._full_border(self.border), self._planes_2d[0].lower())
        stack_args_named = {
            "roi": roi, "labels": self.labels, "blobs_truth": blobs_truth_roi, 
            "circles": circles, "grid": grid, "img_region": self._img_region, 
            "max_intens_proj": max_intens_proj}
        if self._styles_2d[0] == self._DEFAULTS_STYLES_2D[1]:
            # layout for square ROIs with 3D screenshot for square-ish fig
            screenshot = self.scene.mlab.screenshot(antialiased=True)
            plot_2d.plot_2d_stack(
                *stack_args, **stack_args_named, mlab_screenshot=screenshot)
        elif self._styles_2d[0] == self._DEFAULTS_STYLES_2D[2]:
            # single row
            screenshot = self.scene.mlab.screenshot(antialiased=True)
            plot_2d.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=3, 
                single_zoom_row=True, 
                z_level=plot_2d.Z_LEVELS[1], mlab_screenshot=screenshot)
        elif self._styles_2d[0] == self._DEFAULTS_STYLES_2D[3]:
            # layout for wide ROIs to maximize real estate on widescreen
            plot_2d.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=2, zoom_cols=7)
        elif self._styles_2d[0] == self._DEFAULTS_STYLES_2D[4]:
            # multi-zoom overview plots
            plot_2d.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=5)
        elif self._styles_2d[0] == self._DEFAULTS_STYLES_2D[5]:
            # layout for square ROIs with thin rows to create a tall fig
            plot_2d.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=3, zoom_cols=6)
        elif self._styles_2d[0] == self._DEFAULTS_STYLES_2D[6]:
            # atlas editor; need to retain ref or else instance callbacks 
            # created within AtlasEditor will be garbage collected
            self.atlas_ed = atlas_editor.AtlasEditor(
                cli.image5d, config.labels_img, config.channel, curr_offset, 
                self._fig_close_listener, borders_img=config.borders_img, 
                fn_show_label_3d=self.show_label_3d)
            self.atlas_ed.show_atlas()
        else:
            # defaults to Square style without oblique view
            plot_2d.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=3)
    
    def _btn_save_segments_fired(self):
        if self._opened_window_style == self._DEFAULTS_STYLES_2D[6]:
            self.save_atlas()
        else:
            self.save_segs()
    
    @on_trait_change('rois_check_list')
    def update_roi(self):
        print("got {}".format(self.rois_check_list))
        if self.rois_check_list not in ("", _ROI_DEFAULT):
            # get chosen ROI to reconstruct original ROI size and offset 
            # including border
            roi = self._rois_dict[self.rois_check_list]
            cli.roi_size = (roi["size_x"], roi["size_y"], roi["size_z"])
            cli.roi_size = tuple(
                np.add(
                    cli.roi_size, 
                    np.multiply(self.border, 2)).astype(int).tolist())
            self.roi_array = [cli.roi_size]
            cli.offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
            cli.offset = tuple(
                np.subtract(cli.offset, self.border).astype(int).tolist())
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
            self.rois_check_list = _ROI_DEFAULT
            self._reset_segments()
    
    @on_trait_change("_region_id")
    def _region_id_changed(self):
        print("region ID: {}".format(self._region_id))
        centroid, self._img_region = register.get_region_middle(
            config. labels_ref_lookup, self._region_id, config.labels_img, 
            config.labels_scaling)
        if centroid is None:
            self.segs_feedback = (
                "Could not find the region corresponding to ID {}"
                .format(self._region_id))
            return
        curr_roi_size = self.roi_array[0].astype(int)
        corner = np.subtract(
            centroid, 
            np.around(np.divide(curr_roi_size[::-1], 2)).astype(np.int))
        self.z_offset, self.y_offset, self.x_offset = corner
        self.show_3d()
        if self.atlas_ed is not None:
            # sync with atlas editor to point at center of region
            self.atlas_ed.update_coords(centroid)
        self.segs_feedback = (
            "Found region ID {}".format(self._region_id))
    
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
    
    def _get_vis_segments_index(self, segment):
        # must take from vis rather than saved copy in case user 
        # manually updates the table
        #print("segs:\n{}".format(self.segments))
        #print("seg: {}".format(segment))
        #print(self.segments == segment)
        segi = np.where((self.segments == segment).all(axis=1))
        if len(segi) > 0 and len(segi[0]) > 0:
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
    
    def _flag_seg_for_deletion(self, seg):
        seg[3] = -1 * abs(seg[3])
        detector.update_blob_confirmed(seg, -1)
    
    def update_segment(self, segment_new, segment_old=None, remove=False):
        """Update this class object's segments list with a new or updated 
        segment.
        
        Args:
            segment_new: Segment to either add or update, including 
                changes to relative coordinates or radius. Segments are 
                generally given as an array in :func:``detector.format_blob`` 
                format. 
            segment_old: Previous version of the segment, which if found will 
                be replaced by ``segment_new``. The absolute coordinates of 
                ``segment_new`` will also be updated based on the relative 
                coordinates' difference between ``segment_new`` and 
                ``segments_old`` as a convenience. Defaults to None.
            remove: True if the segment should be removed instead of added, 
                in which case ``segment_old`` will be ignored. Defaults to 
                False.
        
        Returns:
            The updated segment.
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
            segi = self._get_vis_segments_index(seg)
            seg = self.segments[segi]
            self._flag_seg_for_deletion(seg)
            self._force_seg_refresh(segi, show=True)
        elif segment_old is not None:
            # update abs coordinates of new segment based on relative coords 
            # since the new segment will typically only update relative coords
            # TODO: consider requiring new seg to already have abs coord updated
            self._segs_moved.append(segment_old)
            diff = np.subtract(seg[:3], segment_old[:3])
            detector.shift_blob_abs_coords(seg, diff)
            segi = self._get_vis_segments_index(segment_old)
            if segi == -1:
                # check if deleted segment if not found
                self._flag_seg_for_deletion(segment_old)
                segi = self._get_vis_segments_index(segment_old)
            if segi != -1:
                # update an existing segment if found
                self.segments[segi] = seg
                print("updated seg: {}".format(seg))
                self._force_seg_refresh(segi, show=True)
        else:
            # add a new segment to the visualizer table
            segs = [seg] # for concatenation
            if self.segments is None or len(self.segments) == 0:
                # copy since the object may be changed elsewhere; cast to 
                # float64 since original type causes an incorrect database 
                # insertion for some reason
                self.segments = np.copy(segs).astype(np.float64)
            else:
                self.segments = np.concatenate((self.segments, segs))
            self.segs_selected.append(len(self.segments) - 1)
            print("added segment to table: {}".format(seg))
        return seg
    
    @property
    def segments(self):
        return self._segments
    
    @segments.setter
    def segments(self, val):
        """Sets segments.
        
        Args:
            val: Numpy array of (n, 10) shape with segments. The columns
                correspond to (z, y, x, radius, confirmed, truth, channel, 
                abs_z,abs_y, abs_x). Note that the "abs" values are different 
                from those used for duplicate shifting. If None, the 
                segments will be reset to an empty list.
        """
        # no longer need to give default value if None, presumably from 
        # update of TraitsUI from 5.1.0 to 5.2.0pre
        if val is None:
            self._segments = []
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
                     editor=CheckListEditor(values=_DEFAULTS_3D, cols=4), 
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
                HGroup(
                    Item(
                        "_structure_scale",
                        editor=RangeEditor(
                            low_name="_structure_scale_low",
                            high_name="_structure_scale_high",
                            mode="slider"),
                        label="Level"
                    ),
                    Item(
                        "_region_id",
                        editor=TextEditor(
                            auto_set=False, enter_set=True, evaluate=int),
                        label="Region"
                    ),
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
    