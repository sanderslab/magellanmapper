#!/usr/bin/env python
# 3D image visualization
# Author: David Young, 2017, 2020
"""3D Visualization GUI.

This module is the main GUI for visualizing 3D objects from imaging stacks. 
The module can be run either as a script or loaded and initialized by 
calling main().

Examples:
    Launch the GUI with the given file at a particular size and offset::
        
        $ ./run.py --img /path/to/file.czi --offset 30,50,205 \
            --size 150,150,10
    
    Alternatively, this module can be run as a script::
        
        $ python -m magmap.visualizer --img /path/to/file.czi
"""

from enum import Enum
import os

import numpy as np
from traits.api import (HasTraits, Instance, on_trait_change, Button, Float,
                        Int, List, Array, Str, push_exception_handler,
                        Property, File)
from traitsui.api import (View, Item, HGroup, VGroup, Handler, 
                          RangeEditor, HSplit, TabularEditor, CheckListEditor, 
                          FileEditor, TextEditor)
from traitsui.tabular_adapter import TabularAdapter
from tvtk.pyface.scene_editor import SceneEditor
from tvtk.pyface.scene_model import SceneModelError
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
import vtk

from magmap.gui import atlas_editor
from magmap.cv import chunking
from magmap.io import cli
from magmap.settings import config
from magmap.cv import cv_nd
from magmap.cv import detector
from magmap.cv import stack_detect
from magmap.io import importer
from magmap.io import libmag
from magmap.io import np_io
from magmap.atlas import ontology
from magmap.plot import plot_3d
from magmap.plot import plot_2d
from magmap.plot import plot_support
from magmap.gui import roi_editor
from magmap.cv import segmenter
from magmap.io import sqlite

# default ROI name
_ROI_DEFAULT = "None selected"


def main():
    """Starts the visualization GUI.
    
    Also processes command-line arguments, sets up Matplotlib style, and
    sets up exception handling.
    """
    # set up command-line arguments, Matplotlib style in which the GUI
    # has been tested, and show complete stacktraces for debugging.
    cli.main()
    plot_2d.setup_style()
    push_exception_handler(reraise_exceptions=True)

    # suppress output window on Windows but print errors to console
    vtk_out = vtk.vtkOutputWindow()
    vtk_out.SetInstance(vtk_out)

    # create Trait-enabled GUI
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
        np.multiply(roi_size, config.resolutions[0][::-1]))
    return "{}{} (series {})\noffset {}, ROI size {} [{}{}]".format(
        region, os.path.basename(config.filename), config.series, offset, 
        tuple(roi_size), tuple(roi_size_um), u'\u00b5m')


class VisHandler(Handler):
    """Simple handler for Visualization object events.
    
    Closes the JVM when the window is closed.
    """
    def closed(self, info, is_ok):
        """Shuts down the application when the GUI is closed."""
        cli.shutdown()


class ListSelections(HasTraits):
    """Traits-enabled list of ROIs."""
    selections = List([_ROI_DEFAULT])


class SegmentsArrayAdapter(TabularAdapter):
    """Blobs TraitsUI table adapter."""
    columns = [("i", "index"), ("z", 0), ("row", 1), ("col", 2),
               ("radius", 3), ("confirmed", 4), ("channel", 6), ("abs_z", 7), 
               ("abs_y", 8), ("abs_x", 9)]
    index_text = Property
    
    def _get_index_text(self):
        return str(self.row)


class Styles2D(Enum):
    """Enumerations for 2D ROI GUI styles."""
    SQUARE = "Square"
    SQUARE_3D = "Square with 3D"
    SINGLE_ROW = "Single row"
    WIDE = "Wide region"
    ZOOM3 = "3 level zoom"
    ZOOM4 = "4 level zoom"
    THIN_ROWS = "Thin rows"


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
        btn_detect_trait: Button editor for segmenting the ROI.
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
    btn_detect_trait = Button("Detect")
    btn_2d_trait = Button("ROI Editor")
    btn_atlas_editor_trait = Button("Atlas Editor")
    btn_save_3d = Button("Save 3D Screenshot")
    btn_save_segments = Button("Save Blobs")
    roi = None # combine with roi_array?
    rois_selections_class = Instance(ListSelections)
    rois_check_list = Str
    _rois_dict = None
    _rois = None
    _segments = Array
    _segs_moved = []  # orig seg of moved blobs to track for deletion
    _scale_detections_low = 0.0
    _scale_detections_high = Float  # needs to be trait to dynamically update
    scale_detections = Float
    segs_pts = None
    segs_selected = List  # indices
    # multi-select to allow updating with a list, but segment updater keeps
    # selections to single when updating them
    segs_table = TabularEditor(
        adapter=SegmentsArrayAdapter(), multi_select=True, 
        selected_row="segs_selected")
    segs_in_mask = None  # boolean mask for segments in the ROI
    segs_cmap = None
    segs_feedback = Str("Segments output")
    labels = None  # segmentation labels
    atlas_eds = []  # open atlas editors
    flipz = True  # True to invert 3D vis along z-axis
    _check_list_3d = List
    _DEFAULTS_3D = ["Side panes", "Side circles", "Raw", "Surface"]
    _check_list_2d = List
    _DEFAULTS_2D = [
        "Filtered", "Border zone", "Segmentation", "Grid", "Max inten proj"]
    _planes_2d = List
    _border_on = False  # remembers last border selection
    _DEFAULT_BORDER = np.zeros(3) # default ROI border size
    _DEFAULTS_PLANES_2D = ["xy", "xz", "yz"]
    _circles_2d = List  # ROI editor circle styles
    _styles_2d = List
    _atlas_label = None
    _structure_scale = Int  # ontology structure levels
    _structure_scale_low = -1
    _structure_scale_high = 20
    _region_id = Str
    _mlab_title = None
    _scene_3d_shown = False  # 3D Mayavi display shown
    _circles_opened_type = None  # enum of circle style for opened 2D plots
    _opened_window_style = None  # 2D plots window style curr open
    _filename = File  # file browser
    _ignore_filename = False  # ignore file update trigger
    _channel = Int  # channel number, 0-based
    _channel_low = -1  # -1 used for None, which translates to "all"
    _channel_high = 0
    _img_region = None
    _PREFIX_BOTH_SIDES = "+/-"
    _camera_pos = None

    def __init__(self):
        # Do not forget to call the parent's __init__
        HasTraits.__init__(self)

        # default options setup
        self._set_border(True)
        self._circles_2d = [
            roi_editor.ROIEditor.CircleStyles.CIRCLES.value]
        self._planes_2d = [self._DEFAULTS_PLANES_2D[0]]
        self._styles_2d = [Styles2D.SQUARE.value]
        # self._check_list_2d = [self._DEFAULTS_2D[1]]
        self._check_list_3d = [self._DEFAULTS_3D[2]]
        if (config.process_settings["vis_3d"].lower()
                == self._DEFAULTS_3D[3].lower()):
            # check "surface" if set in profile
            self._check_list_3d.append(self._DEFAULTS_3D[3])
        # self._structure_scale = self._structure_scale_high

        # setup interface for image
        if config.filename:
            # show image filename in file selector without triggering update
            self._ignore_filename = True
            self._filename = config.filename
        self._setup_for_image()

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

    @staticmethod
    def _get_exp_name(path):
        # get the experiment name as the basename, adding any sub-image
        # offset/size parameters if currently set
        exp_name = os.path.basename(path)
        if config.subimg_offsets and config.subimg_sizes:
            exp_name = stack_detect.make_subimage_name(
                exp_name, config.subimg_offsets[0], config.subimg_sizes[0])
        return exp_name

    def save_segs(self):
        """Saves segments to database.
        
        Segments are selected from a table, and positions are transposed
        based on the current offset. Also inserts a new experiment based 
        on the filename if not already added.
        """
        print("segments", self.segments)
        segs_transposed = []
        segs_to_delete = []
        curr_roi_size = self.roi_array[0].astype(int)
        print("Preparing to insert segments to database with border widths {}"
              .format(self.border))
        feedback = ["Preparing segments:"]
        if self.segments is not None:
            for i in range(len(self.segments)):
                seg = self.segments[i]
                # uses absolute coordinates from end of seg
                seg_db = detector.blob_for_db(seg)
                if seg[4] == -1 and seg[3] < config.POS_THRESH:
                    # attempts to delete user added segments, where radius
                    # assumed to be 0,that are no longer selected
                    feedback.append(
                        "{} to delete (unselected user added or explicitly "
                        "deleted)".format(seg_db))
                    segs_to_delete.append(seg_db)
                else:
                    if (self.border[2] <= seg[0]
                            < (curr_roi_size[2] - self.border[2])
                            and self.border[1] <= seg[1]
                            < (curr_roi_size[1] - self.border[1])
                            and self.border[0] <= seg[2]
                            < (curr_roi_size[0] - self.border[0])):
                        # transpose segments within inner ROI to absolute coords
                        feedback.append(
                            "{} to insert".format(self._format_seg(seg_db)))
                        segs_transposed.append(seg_db)
                    else:
                        feedback.append("{} outside, ignored".format(
                            self._format_seg(seg_db)))
        
        segs_transposed_np = np.array(segs_transposed)
        unverified = None
        if len(segs_transposed_np) > 0:
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
        exp_name = self._get_exp_name(config.filename)
        exp_id = sqlite.select_or_insert_experiment(
            config.db.conn, config.db.cur, exp_name, None)
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
            self._atlas_label = ontology.get_label(
                center[::-1], config.labels_img, config.labels_ref_lookup, 
                config.labels_scaling, level, rounding=True)
            if self._atlas_label is not None:
                title = ontology.get_label_name(self._atlas_label)
                if title is not None:
                    self._mlab_title = self.scene.mlab.title(title)
    
    def _post_3d_display(self, title="clrbrain3d", show_orientation=True):
        """Show axes and saved ROI parameters after 3D display.
        
        Args:
            title: Path without extension to save file if 
                :attr:``config.savefig`` is set to an extension. Defaults to 
                "clrbrain3d".
            show_orientation: True to show orientation axes; defaults to True.
        """
        if self._scene_3d_shown:
            if config.savefig in config.FORMATS_3D:
                path = "{}.{}".format(title, config.savefig)
                libmag.backup_file(path)
                try:
                    # save before setting any other objects to avoid VTK 
                    # render error
                    print("saving 3D scene to {}".format(path))
                    self.scene.mlab.savefig(path)
                except SceneModelError as e:
                    # the scene may not have been activated yet
                    print("unable to save 3D surface")
            if show_orientation:
                # TODO: cannot save file manually once orientation axes are on 
                # and have not found a way to turn them off easily, so 
                # consider turning them off by default and deferring to the 
                # GUI to turn them back on
                self.show_orientation_axes(self.flipz)
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
        curr_offset = list(self._curr_offset())
        offset_orig = np.copy(curr_offset)
        max_offset = (self.x_high, self.y_high, self.z_high)
        # keep offset within bounds
        for i, offset in enumerate(curr_offset):
            if offset >= max_offset[i]:
                curr_offset[i] = max_offset[i] - 1
        feedback = []
        if not np.all(np.equal(curr_offset, offset_orig)):
            self.x_offset, self.y_offset, self.z_offset = curr_offset
            feedback.append(
                "Repositioned ROI from {} to {} to fit within max bounds of {}"
                .format(offset_orig, curr_offset, max_offset))
        # keep size + offset within bounds
        for i, offset in enumerate(curr_offset):
            if offset + curr_roi_size[i] > max_offset[i]:
                curr_roi_size[i] = max_offset[i] - offset
        if not np.all(np.equal(curr_roi_size, roi_size_orig)):
            self.roi_array = [curr_roi_size]
            feedback.append(
                "Resized ROI from {} to {} to fit within max bounds of {}"
                .format(roi_size_orig, curr_roi_size, max_offset))
        print("using ROI offset {}, size of {} (x,y,z)"
              .format(curr_offset, curr_roi_size))
        
        # show raw 3D image unless selected not to
        if self._DEFAULTS_3D[2] in self._check_list_3d:
            # show region of interest based on raw image
            self.roi = plot_3d.prepare_roi(
                config.image5d, curr_roi_size, curr_offset)
            
            if self._DEFAULTS_3D[3] in self._check_list_3d:
                # surface rendering, segmenting to clean up image 
                # if 2D segmentation option checked
                segment = self._DEFAULTS_2D[2] in self._check_list_2d
                plot_3d.plot_3d_surface(
                    self.roi, self.scene.mlab, config.channel, segment, 
                    self.flipz)
                self._scene_3d_shown = True
            else:
                # 3D point rendering
                self._scene_3d_shown = plot_3d.plot_3d_points(
                    self.roi, self.scene.mlab, config.channel, self.flipz)
            
            # process ROI in prep for showing filtered 2D view and segmenting
            if not libmag.is_binary(self.roi):
                self.roi = plot_3d.saturate_roi(
                    self.roi, channel=config.channel)
                self.roi = plot_3d.denoise_roi(self.roi, config.channel)
            else:
                libmag.printv(
                    "binary image detected, will not preprocess")
        
        else:
            self.scene.mlab.clf()
        
        # show shadow images around the points if selected
        if self._DEFAULTS_3D[0] in self._check_list_3d:
            plot_3d.plot_2d_shadows(self.roi, self)
        
        # show title from labels reference if available
        self._update_structure_level(curr_offset, curr_roi_size)
        
        self._reset_segments()
        if feedback:
            self.segs_feedback = " ".join(feedback)
            print(self.segs_feedback)
    
    def show_label_3d(self, label_id):
        """Show 3D region of main image corresponding to label ID.
        
        Args:
            label_id: ID of label to display.
        """
        # get bounding box for label region
        bbox = cv_nd.get_label_bbox(config.labels_img, label_id)
        if bbox is None: return
        shape, slices = cv_nd.get_bbox_region(
            bbox, 10, config.labels_img.shape)
        
        # update GUI dimensions
        self.roi_array = [shape[::-1]] # TODO: avoid decimal point
        self.z_offset, self.y_offset, self.x_offset = [
            slices[i].start for i in range(len(slices))]
        self._scene_3d_shown = True
        
        # show main image corresponding to label region
        if isinstance(label_id, (tuple, list)):
            label_mask = np.isin(config.labels_img[tuple(slices)], label_id)
        else:
            label_mask = config.labels_img[tuple(slices)] == label_id
        self.roi = np.copy(config.image5d[0][slices])
        self.roi[~label_mask] = 0
        plot_3d.plot_3d_surface(
            self.roi, self.scene.mlab, config.channel, flipud=self.flipz)
        #plot_3d.plot_3d_points(self.roi, self.scene.mlab, config.channel)
        name = os.path.splitext(os.path.basename(config.filename))[0]
        self._post_3d_display(
            title="label3d_{}".format(name), show_orientation=False)
    
    def _setup_for_image(self):
        """Setup GUI parameters for the loaded image5d.
        """
        if config.image5d is not None:
            # set up channel spinner based on number of channels available
            if config.image5d.ndim >= 5:
                # increase max channels based on channel dimension
                self._channel_high = config.image5d.shape[4] - 1
            else:
                # only one channel available
                self._channel_low = 0
            # None channel defaults to all channels, represented in the channel
            # spinner here by -1
            self._channel = -1 if config.channel is None else config.channel

            # dimension max values in pixels
            size = config.image5d.shape[1:4]
            # TODO: consider subtracting 1 to avoid max offset being 1 above
            # true max, but currently convenient to display size and checked
            # elsewhere
            # TODO: does not appear to update on subsequent image loading
            self.z_high, self.y_high, self.x_high = size
            if config.roi_offset is not None:
                # apply user-defined offsets
                self.x_offset = config.roi_offset[0]
                self.y_offset = config.roi_offset[1]
                self.z_offset = config.roi_offset[2]
            self.roi_array[0] = ([100, 100, 15] if config.roi_size is None
                                 else config.roi_size)
            # show the default ROI
            self.show_3d()
        
        # set up selector for loading past saved ROIs
        self._rois_dict = {_ROI_DEFAULT: None}
        if config.db is not None:
            self._rois = config.db.get_rois(self._get_exp_name(config.filename))
        self.rois_selections_class = ListSelections()
        if self._rois is not None and len(self._rois) > 0:
            for roi in self._rois:
                self._append_roi(roi, self._rois_dict)
        self.rois_selections_class.selections = list(self._rois_dict.keys())
        self.rois_check_list = _ROI_DEFAULT

    @on_trait_change("_filename")
    def update_filename(self):
        """Update the selected filename and load the corresponding image.
        
        Since an original (eg .czi) image can be processed in so many 
        different ways, assume that the user will select the Numpy image 
        file instead of the raw image. Image settings will be constructed 
        from the Numpy image filename. Processed files (eg ROIs, blobs) 
        will not be loaded for now.
        """
        if self._ignore_filename:
            # may ignore if only updating widget value, without triggering load
            self._ignore_filename = False
            return
        filename, series = importer.deconstruct_np_filename(self._filename)
        if filename is not None and series is not None:
            config.filename = filename
            config.series = series
            print("Changed filename to {}, series to {}"
                  .format(config.filename, config.series))
            # TODO: consider loading processed images, blobs, etc
            np_io.setup_images(config.filename)
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
            self.show_orientation_axes(self.flipz)
        #self.scene.mlab.outline() # affects zoom after segmenting
        #self.scene.mlab.axes() # need to adjust units to microns
        print("view: {}\nroll: {}".format(
            self.scene.mlab.view(), self.scene.mlab.roll()))
    
    def show_orientation_axes(self, flipud=False):
        """Show orientation axes with option to flip z-axis to match 
        handedness in Matplotlib images with z increasing upward.
        
        Args:
            flipud: True to invert z-axis, which also turns off arrowheads; 
                defaults to True.
        """
        orient = self.scene.mlab.orientation_axes()
        if flipud:
            # flip z-axis and turn off now upside-down arrowheads
            orient.axes.total_length = [1, 1, -1]
            orient.axes.cone_radius = 0
    
    @on_trait_change("scene.busy")
    def _scene_changed(self):
        # show camera position after roll changes; only use roll for 
        # simplification since almost any movement involves a roll change
        roll = self.scene.mlab.roll()
        if self._camera_pos is None or self._camera_pos["roll"] != roll:
            self._camera_pos = {"view": self.scene.mlab.view(), "roll": roll}
            print("camera:", self._camera_pos)
        
    def _is_segs_none(self, segs):
        """Checks if segs is equivalent to None.
        """
        # segs is 0 for some reason if no parameter given in fired trait
        return segs is None or not isinstance(segs, np.ndarray)
    
    def _btn_detect_trait_fired(self, segs=None):
        # collect segments in ROI and padding region, ensuring coordinates
        # are relative to offset
        offset = self._curr_offset()
        roi_size = self.roi_array[0].astype(int)
        if config.blobs is None:
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
                config.blobs, offset, roi_size, roi_editor.padding)
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
                show_shadows, self.flipz)
            # reduce number of digits to make the slider more compact
            scale = float(libmag.format_num(scale, 4))
            self._scale_detections_high = scale * 2
            self.scale_detections = scale
        
        if self._DEFAULTS_2D[2] in self._check_list_2d:
            blobs = self.segments[self.segs_in_mask]
            # 3D-seeded watershed segmentation using detection blobs
            '''
            # could get initial segmentation from r-w
            walker = segmenter.segment_rw(
                self.roi, config.channel, erosion=1)
            '''
            self.labels = segmenter.segment_ws(
                self.roi, config.channel, None, blobs)
            '''
            # 3D-seeded random-walker with high beta to limit walking 
            # into background, also removing objects smaller than the 
            # smallest blob, roughly normalized for anisotropy and 
            # reduced by not including the 4/3 factor
            min_size = int(
                np.pi * np.power(np.amin(np.abs(blobs[:, 3])), 3) 
                / np.mean(plot_3d.calc_isotropic_factor(1)))
            print("min size threshold for r-w: {}".format(min_size))
            self.labels = segmenter.segment_rw(
                self.roi, config.channel, beta=100000, 
                blobs=blobs, remove_small=min_size)
            '''
        #detector.show_blob_surroundings(self.segments, self.roi)
        self.scene.mlab.outline()
    
    @on_trait_change('scale_detections')
    def update_scale_detections(self):
        """Updates the glyph scale factor.
        """
        if self.segs_pts is not None:
            self.segs_pts.glyph.glyph.scale_factor = self.scale_detections
    
    def _roi_ed_close_listener(self, evt):
        """Handle ROI Editor close events.

        Args:
            evt (:obj:`matplotlib.backend_bases.Event`): Event.

        """
        self._circles_opened_type = None
        self._opened_window_style = None
        if (self._circles_2d[0] ==
                roi_editor.ROIEditor.CircleStyles.FULL_ANNOTATION):
            # reset if in full annotation mode to avoid further duplicating 
            # circles, saving beforehand to prevent loss from premature  
            # window closure
            self.save_segs()
            self._reset_segments()
            self._circles_2d = [roi_editor.ROIEditor.CircleStyles.CIRCLES.value]
            self.segs_feedback = "Reset circles after saving full annotations"

    def _atlas_ed_close_listener(self, evt, atlas_ed):
        """Handle Atlas Editor close events.

        Args:
            evt (:obj:`matplotlib.backend_bases.Event`): Event.
            atlas_ed (:obj:`gui.atlas_editor.AtlasEdtor`): Atlas editor
                that was closed.

        """
        self.atlas_eds[self.atlas_eds.index(atlas_ed)] = None

    def _btn_2d_trait_fired(self):
        """Handle ROI Editor button events."""
        if (roi_editor.ROIEditor.CircleStyles(self._circles_2d[0])
                != roi_editor.ROIEditor.CircleStyles.NO_CIRCLES
                and self._circles_opened_type
                and self._circles_opened_type !=
                roi_editor.ROIEditor.CircleStyles.NO_CIRCLES):
            # prevent multiple editable windows from being opened 
            # simultaneously to avoid unsynchronized state
            self.segs_feedback = (
                "Cannot show ROI Editor while another editable "
                "plot is showing. Please choose \"No circles\" or redraw.")
            return
        if (not self._circles_opened_type 
                or self._circles_opened_type ==
                roi_editor.ROIEditor.CircleStyles.NO_CIRCLES):
            # set opened window type if not already set or non-editable window
            self._circles_opened_type = roi_editor.ROIEditor.CircleStyles(
                self._circles_2d[0])
        self._opened_window_style = self._styles_2d[0]
        self.segs_feedback = ""
        
        # shows 2D plots
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        # update verify flag
        roi_editor.verify = self._DEFAULTS_2D[1] in self._check_list_2d
        img = config.image5d
        roi = None
        if self._DEFAULTS_2D[0] in self._check_list_2d:
            print("showing processed 2D images")
            # denoised ROI processed during 3D display
            roi = self.roi
            if config.process_settings["thresholding"]:
                # thresholds prior to blob detection
                roi = plot_3d.threshold(roi)
        elif config.image5d is None:
            print("loading original image stack from file")
            config.image5d = importer.read_file(config.filename, config.series)
            img = config.image5d
        
        blobs_truth_roi = None
        if config.truth_db is not None:
            # collect truth blobs from the truth DB if available
            blobs_truth_roi, _ = detector.get_blobs_in_roi(
                config.truth_db.blobs_truth, curr_offset, curr_roi_size, 
                roi_editor.padding)
            transpose = np.zeros(blobs_truth_roi.shape[1])
            transpose[0:3] = curr_offset[::-1]
            blobs_truth_roi = np.subtract(blobs_truth_roi, transpose)
            blobs_truth_roi[:, 5] = blobs_truth_roi[:, 4]
            #print("blobs_truth_roi:\n{}".format(blobs_truth_roi))
        title = _fig_title(ontology.get_label_name(self._atlas_label),
                           curr_offset, curr_roi_size)
        filename_base = importer.filename_to_base(
            config.filename, config.series)
        grid = self._DEFAULTS_2D[3] in self._check_list_2d
        max_intens_proj = self._DEFAULTS_2D[4] in self._check_list_2d
        stack_args = (
            self.update_segment, title, filename_base, img, config.channel,
            curr_roi_size, curr_offset, self.segments, self.segs_in_mask,
            self.segs_cmap, self._roi_ed_close_listener,
            # additional args with defaults
            self._full_border(self.border), self._planes_2d[0].lower())
        roi_ed = roi_editor.ROIEditor()
        roi_cols = libmag.get_if_within(
            config.plot_labels[config.PlotLabels.LAYOUT], 0)
        stack_args_named = {
            "roi": roi, "labels": self.labels, "blobs_truth": blobs_truth_roi, 
            "circles": roi_editor.ROIEditor.CircleStyles(self._circles_2d[0]),
            "grid": grid, "img_region": self._img_region,
            "max_intens_proj": max_intens_proj, 
            "labels_img": config.labels_img,
            "zoom_shift": config.plot_labels[config.PlotLabels.ZOOM_SHIFT],
            "roi_cols": roi_cols,
        }
        if self._styles_2d[0] == Styles2D.SQUARE_3D.value:
            # layout for square ROIs with 3D screenshot for square-ish fig
            screenshot = self.scene.mlab.screenshot(
                mode="rgba", antialiased=True)
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, mlab_screenshot=screenshot)
        elif self._styles_2d[0] == Styles2D.SINGLE_ROW.value:
            # single row
            screenshot = self.scene.mlab.screenshot(
                mode="rgba", antialiased=True)
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=2, 
                single_roi_row=True, 
                z_level=roi_ed.ZLevels.MIDDLE, mlab_screenshot=screenshot)
        elif self._styles_2d[0] == Styles2D.WIDE.value:
            # layout for wide ROIs, which shows only one overview plot
            stack_args_named["roi_cols"] = 7
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named)
        elif self._styles_2d[0] == Styles2D.ZOOM3.value:
            # 3 level zoom overview plots with specific multipliers
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=(0, 4, 20))
        elif self._styles_2d[0] == Styles2D.ZOOM4.value:
            # 4 level zoom overview plots with default zoom multipliers
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=4)
        elif self._styles_2d[0] == Styles2D.THIN_ROWS.value:
            # layout for fewer columns to create a thinner, taller fig
            stack_args_named["roi_cols"] = 6
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=2)
        else:
            # defaults to Square style with another overview plot in place
            # of 3D screenshot
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=2)

    def _btn_atlas_editor_trait_fired(self):
        # atlas editor; need to retain ref or else instance callbacks 
        # created within AtlasEditor will be garbage collected
        title = config.filename
        if self.atlas_eds:
            # distinguish multiple Atlas Editor windows with number since
            # using the same title causes the windows to overlap
            title += " ({})".format(len(self.atlas_eds) + 1)
        atlas_ed = atlas_editor.AtlasEditor(
            config.image5d, config.labels_img, config.channel, 
            self._curr_offset(), self._atlas_ed_close_listener,
            config.borders_img, self.show_label_3d, title,
            self._refresh_atlas_eds)
        self.atlas_eds.append(atlas_ed)
        atlas_ed.show_atlas()

    def _refresh_atlas_eds(self, ed_ignore):
        """Callback handler to refresh all other Atlas Editors

        Args:
            ed_ignore (:obj:`gui.atlas_editor.AtlasEditor`): Atlas Editor
                to not refresh, typically the calling editor.

        """
        for ed in self.atlas_eds:
            if ed is None or ed is ed_ignore: continue
            ed.refresh_images()

    def _btn_save_3d_fired(self):
        # save 3D image with the currently set extension in config
        screenshot = self.scene.mlab.screenshot(mode="rgba", antialiased=True)
        path = plot_support.get_roi_path(
            config.filename, self._curr_offset(), self.roi_array[0].astype(int))
        plot_2d.plot_image(screenshot, path)
    
    def _btn_save_segments_fired(self):
        # save blobs to database
        self.save_segs()
    
    @on_trait_change('rois_check_list')
    def update_roi(self):
        print("got {}".format(self.rois_check_list))
        if self.rois_check_list not in ("", _ROI_DEFAULT):
            # get chosen ROI to reconstruct original ROI size and offset 
            # including border
            roi = self._rois_dict[self.rois_check_list]
            config.roi_size = (roi["size_x"], roi["size_y"], roi["size_z"])
            config.roi_size = tuple(
                np.add(
                    config.roi_size, 
                    np.multiply(self.border, 2)).astype(int).tolist())
            self.roi_array = [config.roi_size]
            config.roi_offset = (roi["offset_x"], roi["offset_y"], roi["offset_z"])
            config.roi_offset = tuple(
                np.subtract(config.roi_offset, self.border).astype(int).tolist())
            self.x_offset, self.y_offset, self.z_offset = config.roi_offset
            
            # redraw the original ROI and prepare verify mode
            self.show_3d()
            if self._scene_3d_shown:
                self.show_orientation_axes(self.flipz)
            blobs = sqlite.select_blobs(config.db.cur, roi["id"])
            self._btn_detect_trait_fired(segs=blobs)
            roi_editor.verify = True
        else:
            print("no roi found")
            roi_editor.verify = False
    
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
        """Center the viewer on the region specified in the corresponding 
        text box.
        
        :const:``_PREFIX_BOTH_SIDES`` can be given to specify IDs with 
        both positive and negative values. All other non-integers will 
        be ignored.
        """
        print("region ID: {}".format(self._region_id))
        if config.labels_img is None:
            self.segs_feedback = "No labels image loaded to find region"
            return
        
        both_sides = self._region_id.startswith(self._PREFIX_BOTH_SIDES)
        region_id = self._region_id
        if both_sides:
            # user can specify both sides to get corresponding pos and neg IDs
            region_id = self._region_id[len(self._PREFIX_BOTH_SIDES):]
        try:
            region_id = int(region_id)
        except ValueError:
            # return if cannot convert to an integer
            self.segs_feedback = (
                "Region ID must be an integer n, or \"+/-n\" to get both sides"
            )
            return
        centroid, self._img_region, region_ids = ontology.get_region_middle(
            config. labels_ref_lookup, region_id, config.labels_img, 
            config.labels_scaling, both_sides=both_sides)
        if centroid is None:
            self.segs_feedback = (
                "Could not find the region corresponding to ID {}"
                .format(self._region_id))
            return
        if self._DEFAULTS_3D[2] in self._check_list_3d:
            # in "raw" mode, simply center the current ROI on centroid, 
            # which may within a sub-label
            curr_roi_size = self.roi_array[0].astype(int)
            corner = np.subtract(
                centroid, 
                np.around(np.divide(curr_roi_size[::-1], 2)).astype(np.int))
            self.z_offset, self.y_offset, self.x_offset = corner
            self.show_3d()
        else:
            # in non-"raw" mode, show the full label including sub-labels 
            # without non-label areas; TODO: consider making default or 
            # only option
            self.show_label_3d(region_ids)
        for ed in self.atlas_eds:
            if ed is None: continue
            # sync with atlas editor to point at center of region
            ed.update_coords(centroid)
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
                        Item("_channel", label="Channel",
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
                             editor=CheckListEditor(values=[
                                 e.value
                                 for e in roi_editor.ROIEditor.CircleStyles]),
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
                             editor=CheckListEditor(
                                 values=[e.value for e in Styles2D]), 
                             style="simple",
                             label="2D styles"
                        ),
                    ),
                ),
                HGroup(
                    Item("btn_redraw_trait", show_label=False), 
                    Item("btn_detect_trait", show_label=False),
                    Item("btn_2d_trait", show_label=False), 
                    Item("btn_atlas_editor_trait", show_label=False)
                ),
                Item(
                    "scale_detections",
                    editor=RangeEditor(
                        low_name="_scale_detections_low",
                        high_name="_scale_detections_high",
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
                            auto_set=False, enter_set=True, evaluate=str),
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
                ),
                HGroup(
                    Item("btn_save_3d", show_label=False), 
                    Item("btn_save_segments", show_label=False)
                ),
            )
        ),
        handler=VisHandler(),
        title="magmap",
        resizable=True
    )


if __name__ == "__main__":
    print("Starting visualizer...")
    main()
