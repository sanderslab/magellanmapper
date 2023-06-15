#!/usr/bin/env python
# 3D image visualization
# Author: David Young, 2017, 2020
"""3D Visualization GUI.

This module is the main GUI for visualizing 3D objects from imaging stacks. 
The module can be run either as a script or loaded and initialized by 
calling main().

Examples:
    Launch the GUI with the given file at a particular size and offset::
        
        ./run.py --img /path/to/file.czi --offset 30,50,205 --size 150,150,10
"""
import glob
import pprint
from collections import OrderedDict
from enum import auto, Enum
import os
import subprocess
import sys
from typing import Dict, List as TypeList, Optional, Sequence, TYPE_CHECKING, \
    Tuple, Type, Union

import matplotlib
matplotlib.use("Qt5Agg")  # explicitly use PyQt5 for custom GUI events
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import figure
import numpy as np
from PyQt5 import Qt, QtWidgets, QtCore

# adjust for HiDPI screens before QGuiApplication is created, necessary
# on Windows and Linux (not needed but no apparent affect on MacOS)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
try:
    # use policy introduced in Qt 5.14 to account for non-integer factor
    # scaling, eg 150%, which avoids excessive window size upscaling
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor)
except AttributeError:
    pass

# import PyFace components after HiDPI adjustment
from pyface import confirmation_dialog
from pyface.api import DirectoryDialog, FileDialog, OK, YES, CANCEL
from pyface.image_resource import ImageResource
from traits.api import Any, Array, Bool, Button, Event, Float, Instance, \
    Int, List, observe, on_trait_change, Property, push_exception_handler, \
    Str, HasTraits
from traitsui.api import ArrayEditor, BooleanEditor, CheckListEditor, \
    EnumEditor, HGroup, HSplit, Item, NullEditor, ProgressEditor, RangeEditor, \
    StatusItem, TextEditor, VGroup, View, Tabbed, TabularEditor
from traitsui.basic_editor_factory import BasicEditorFactory
try:
    from traitsui.qt.editor import Editor
except ModuleNotFoundError:
    from traitsui.qt4.editor import Editor
from traitsui.tabular_adapter import TabularAdapter

try:
    from tvtk.pyface.scene_editor import SceneEditor
    from tvtk.pyface.scene_model import SceneModelError
    from mayavi.tools.mlab_scene_model import MlabSceneModel
    from mayavi.core.ui.mayavi_scene import MayaviScene
    import vtk
except ImportError:
    SceneEditor = None
    SceneModelError = None
    MlabSceneModel = None
    MayaviScene = None
    vtk = None

from magmap.atlas import ontology
from magmap.brain_globe import bg_controller
from magmap.cv import classifier, colocalizer, cv_nd, detector, segmenter, \
    verifier
from magmap.gui import atlas_editor, atlas_threads, import_threads, \
    plot_editor, roi_editor, verifier_editor, vis_3d, vis_handler
from magmap.io import cli, importer, libmag, load_env, naming, np_io, sitk_io, \
    sqlite
from magmap.plot import colormaps, plot_2d, plot_3d
from magmap.settings import config, prefs_prof, profiles

if TYPE_CHECKING:
    from bg_atlasapi import BrainGlobeAtlas
    from magmap.plot import plot_support

# logging instance
_logger = config.logger.getChild(__name__)


def main():
    """Starts the visualization GUI.
    
    Also sets up exception handling.
    """
    # show complete stacktraces for debugging
    push_exception_handler(reraise_exceptions=True)

    if vtk is not None:
        # suppress output window on Windows but print errors to console
        vtk_out = vtk.vtkOutputWindow()
        vtk_out.SetInstance(vtk_out)

    # create Trait-enabled GUI
    visualization = Visualization()
    visualization.configure_traits()
    

class _MPLFigureEditor(Editor):
    """Custom TraitsUI editor to handle Matplotlib figures."""
    scrollable = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """Create canvas through the Matplotlib Qt backend."""
        mpl_canvas = FigureCanvasQTAgg(self.value)
        return mpl_canvas


class MPLFigureEditor(BasicEditorFactory):
    """Custom TraitsUI editor for a Matplotlib figure."""
    klass = _MPLFigureEditor


class SegmentsArrayAdapter(TabularAdapter):
    """Blobs TraitsUI table adapter."""
    columns = [("i", "index"), ("z", 0), ("row", 1), ("col", 2),
               ("radius", 3), ("confirmed", 4), ("channel", 6), ("abs_z", 7), 
               ("abs_y", 8), ("abs_x", 9)]
    index_text = Property
    
    def _get_index_text(self):
        return str(self.row)


class ProfileCats(Enum):
    """Profile categories enumeration."""
    ROI = "ROI profiles"
    ATLAS = "Atlas profiles"
    GRID = "Grid Search"


class TraitsList(HasTraits):
    """Generic Traits-enabled list."""
    selections = List([""])


class ProfilesArrayAdapter(TabularAdapter):
    """Profiles TraitsUI table adapter."""
    columns = [("Category", 0), ("Profile", 1), ("Channel", 2)]
    widths = {0: 0.3, 1: 0.6, 2: 0.1}

    def get_width(self, object, trait, column):
        """Specify column widths."""
        return self.widths[column]


class ImportFilesArrayAdapter(TabularAdapter):
    """Import files TraitsUI table adapter."""
    columns = [("File", 0), ("Channel", 1)]
    widths = {0: 0.9, 1: 0.1}

    def get_width(self, object, trait, column):
        """Specify column widths."""
        return self.widths[column]


class ImportModes(Enum):
    """Enumerations for import modes."""
    DIR = auto()
    MULTIPAGE = auto()


class Styles2D(Enum):
    """Enumerations for 2D ROI GUI styles."""
    SQUARE = "Square layout"
    SQUARE_3D = "Square with 3D"
    SINGLE_ROW = "Single row"
    WIDE = "Wide region"
    ZOOM3 = "3 level zoom"
    ZOOM4 = "4 level zoom"
    THIN_ROWS = "Thin rows"


class RegionOptions(Enum):
    """Enumerations for region options."""
    BOTH_SIDES = "Both sides"
    INCL_CHILDREN = "Children"
    APPEND = "Append"
    SHOW_ALL = "Show all"


class AtlasEditorOptions(Enum):
    """Enumerations for Atlas Editor options."""
    SHOW_LABELS = "Labels"
    SYNC_ROI = "Sync ROI"
    ZOOM_ROI = "Zoom ROI"
    CROSSHAIRS = "Crosshairs"


class Vis3dOptions(Enum):
    """Enumerations for 3D viewer options."""
    RAW = "Raw"
    SURFACE = "Surface"
    CLEAR = "Clear"
    PANES = "Panes"
    SHADOWS = "Shadows"


class BlobsVisibilityOptions(Enum):
    """Enumerations for blob visibility options."""
    VISIBLE = "Visible"


class ColocalizeOptions(Enum):
    """Enumerations for co-localization options."""
    DEFAULT = ""
    INTENSITY = "Intensity"
    MATCHES = "Matches"


class BlobColorStyles(Enum):
    """Enumerations for blob color style options."""
    ATLAS_LABELS = "Atlas label colors"
    UNIQUE = "Unique colors"
    CHANNEL = "Channel colors"


class ControlsTabs(Enum):
    """Enumerations for controls tabs."""
    ROI = auto()
    DETECT = auto()
    PROFILES = auto()
    ADJUST = auto()
    IMPORT = auto()


class BrainGlobeArrayAdapter(TabularAdapter):
    """Import files TraitsUI table adapter."""
    columns = [("Atlas", 0), ("Ver", 1), ("Installed?", 2)]
    widths = {0: 0.5, 1: 0.1, 2: 0.4}

    def get_width(self, object, trait, column):
        """Specify column widths."""
        return self.widths[column]


class Visualization(HasTraits):
    """GUI for choosing a region of interest and segmenting it.
    
    TraitUI-based graphical interface for selecting dimensions of an
    image to view and segment.
    
    Attributes:
        x_low, x_high, ... (Int): Low and high values for image coordinates.
        x_offset: Integer trait for x-offset.
        y_offset: Integer trait for y-offset.
        z_offset: Integer trait for z-offset.
        scene: The main scene
        btn_redraw: Button editor for drawing the reiong of
            interest.
        btn_detect: Button editor for segmenting the ROI.
        roi: The ROI.
        segments: Array of segments; if None, defaults to a Numpy array
            of zeros with one row.
        segs_selected: List of indices of selected segments.
        controls_created (Bool): True if the controls have been created;
            defaults to False.
        mpl_fig_active (Any): The Matplotlib figure currently shown.
        scene_3d_shown (bool): True if the Mayavi 3D plot has been shown.
        selected_viewer_tab (Enum): The Enum corresponding to the selected
            tab in the viewer panel.
        select_controls_tab (int): Enum value from :class:`ControlsTabs` to
            select the controls panel tab.
        stale_viewers (
            dict[:class:`magmap.gui.vis_handler.vis_handler.ViewerTabs`,
            :class:`magmap.gui.vis_handler.vis_handler.StaleFlags`]):
            Dictionary of viewer tab Enums to stale flag Enums, where the
            flag indicates the update that the viewer would require.
            Defaults to all viewers set to
            :class:`vis_handler.StaleFlags.IMAGE`.
    
    """
    #: default ROI name.
    _ROI_DEFAULT: str = "None selected"

    # File selection

    _filename = Str(
        tooltip="Load an image. If the file format requires import, the\n"
                "Import tab will open automatically.")
    _filename_btn = Button("Browse")
    _channel_names = Instance(TraitsList)
    _channel = List  # selected channels, 0-based
    _rgb = Bool(config.rgb, tooltip="Show image as RGB(A)")
    _rgb_enabled = Bool
    _merge_chls = Bool(
        tooltip="Merge channels using additive blending"
    )
    
    # main registered image available and selected dropdowns
    _main_img_name_avail = Str
    _main_img_names_avail = Instance(TraitsList)
    _MAIN_IMG_NAME_AVAIL_DEFAULT = "Available ({})"
    _main_img_name = Str
    _main_img_names = Instance(TraitsList)
    _MAIN_IMG_NAME_DEFAULT = "Selected ({})"
    
    # labels registered image selection dropdown
    _labels_img_name = Str
    _labels_img_names = Instance(TraitsList)
    
    _labels_ref_path = Str  # labels ontology reference path
    _labels_ref_btn = Button("Browse")  # button to select ref file
    _reload_btn = Button("Reload")  # button to reload images
    
    # ROI selection

    # image coordinate limits
    x_low = Int(0)
    x_high = Int(100)
    y_low = Int(0)
    y_high = Int(100)
    z_low = Int(0)
    z_high = Int(100)

    # ROI offset and shape
    x_offset = Int
    y_offset = Int
    z_offset = Int
    roi_array = Array(Int, shape=(1, 3), editor=ArrayEditor(format_str="%0d"))
    _roi_center = Bool(tooltip="Treat the offset as the center of the ROI")

    btn_redraw = Button("Redraw")
    _btn_save_fig = Button("Save Figure")
    _roi_btn_help = Button("Help")
    roi = None  # combine with roi_array?
    _rois_selections = Instance(TraitsList)
    rois_check_list = Str
    _rois_dict = None
    _rois = None
    _roi_feedback = Str()
    
    # Detect panel
    
    btn_detect = Button("Detect")
    btn_save_segments = Button("Save Blobs")
    _btn_verifier = Button(
        "Open Verifier",
        tooltip="Open a viewer displaying each blob in a separate plot to\n"
                "verify classifications.")
    _segs_chls_names = Instance(TraitsList)  # blob detection channels
    _segs_chls = List  # selected channels, 0-based
    _segs_labels = List(  # blob label selector
        ["-1"],
        tooltip="All blob labels will be set to this value when running Detect",
    )
    _segs_model_path = Str  # blobs classifier model path
    _segs_model_btn = Button("Browse")  # button to select model file
    _segs_visible = List  # blob visibility options
    _colocalize = List  # blob co-localization options
    _blob_color_style = List  # blob coloring
    _segments = Array  # table of blobs
    _segs_moved = []  # orig seg of moved blobs to track for deletion
    _scale_detections_low = Float(0)
    _scale_detections_high = Float  # needs to be trait to dynamically update
    scale_detections = Float(
        tooltip="Change the size of blobs in the 3D viewer"
    )
    # for blob mask slider
    _segs_mask_low = 1
    _segs_mask_high = Int(1)
    _segs_mask = Int(
        1, tooltip="Change the fraction of blobs shown in the 3D viewer"
    )
    segs_selected = List  # indices
    _segs_row_scroll = Int()  # row index to scroll the table
    # multi-select to allow updating with a list, but segment updater keeps
    # selections to single when updating them
    _segs_refresh_evt = Event()
    segs_table = TabularEditor(
        adapter=SegmentsArrayAdapter(), multi_select=True,
        selected_row="segs_selected", scroll_to_row="_segs_row_scroll",
        refresh="_segs_refresh_evt")
    segs_in_mask = None  # boolean mask for segments in the ROI
    segs_cmap = None
    segs_feedback = Str("Segments output")
    labels = None  # segmentation labels

    # Profiles panel

    _profiles_cats = List
    _profiles_names = Instance(TraitsList)
    _profiles_name = Str
    _profiles_chls = List(
        tooltip="Channels to apply the selected profile"
    )
    _profiles_table = TabularEditor(
        adapter=ProfilesArrayAdapter(), editable=True, auto_resize_rows=True,
        stretch_last_section=False, drag_move=True)
    _profiles = List  # profiles table list
    _profiles_add_btn = Button(
        "Add",
        tooltip="Click to add the selected profile for the checked channels.\n"
                "Press Backspace/Delete key to remove the selected table row.\n"
                "Drag rows to rearrange the order of loaded profiles."
    )
    _profiles_load_btn = Button(
        "Rescan",
        tooltip="Rescan the 'profiles' folder for YAML files"
    )
    _profiles_preview = Str  # preview the selected profile
    _profiles_combined = Str  # combined profiles for selected category
    _profiles_ver = Str
    _profiles_reset_prefs_btn = Button("Reset preferences")
    _profiles_btn_help = Button("Help")
    _profiles_reset_prefs = Bool  # trigger resetting prefs in handler

    # Image adjustment panel

    _imgadj_names = Instance(TraitsList)
    _imgadj_name = Str
    _imgadj_chls_names = Instance(TraitsList)
    _imgadj_chls = Str
    _imgadj_min = Float
    _imgadj_min_low = Float(0)
    _imgadj_min_high = Float
    _imgadj_min_auto = Bool
    _imgadj_max = Float
    _imgadj_max_low = Float(0)
    _imgadj_max_high = Float
    _imgadj_max_auto = Bool
    _imgadj_brightness = Float
    _imgadj_brightness_low = Float(0)
    _imgadj_brightness_high = Float
    _imgadj_contrast = Float
    _imgadj_alpha = Float
    _imgadj_alpha_blend = Float(0.5)
    _imgadj_alpha_blend_check = Bool(
        tooltip="Blend the overlapping parts of imags to shown alignment.\n"
                "Applied to the first two channels of the main image."
    )

    # Image import panel

    _import_browser = Str
    _import_file_btn = Button(
        "File(s)",
        tooltip="Select a file to import, typically a multiplane image.\n"
                "Channels are determined by names ending with '_ch_x'\n"
                "(eg 'img_ch_0.tif', 'img_ch_1.tif'). If only one file is\n"
                "selected, all files with matching names except for the\n"
                "channel will also be added.")
    _import_dir_btn = Button(
        "Dir",
        tooltip="Select a directory of single-plane images to import.\n"
                "Files will be sorted alphabetically, and files ending with\n"
                "'_ch_x' are sorted into the given channel. Any file that\n"
                "cannot be loaded will be simply ignored.")
    _import_table = TabularEditor(
        adapter=ImportFilesArrayAdapter(), editable=True, auto_resize_rows=True,
        stretch_last_section=False)
    _import_paths = List  # import paths table list
    _import_btn = Button("Import files")
    _import_btn_enabled = Bool
    _import_clear_btn = Button("Clear import")
    _import_res = Array(float, shape=(1, 3))
    _import_mag = Float(1.0)
    _import_zoom = Float(1.0)
    _import_shape = Array(int, shape=(1, 5), editor=ArrayEditor(
        width=-40, format_str="%0d"))
    # map bits to bytes for constructing Numpy data type
    _IMPORT_BITS = OrderedDict((
        ("Bit", ""), ("8", "1"), ("16", "2"), ("32", "3"), ("64", "4")))
    _import_bit = List
    # map numerical signage and precision to Numpy data type
    _IMPORT_DATA_TYPES = OrderedDict((
        ("Type", ""),
        ("Unsigned integer", "u"),
        ("Signed integer", "i"),
        ("Floating point", "f"),
    ))
    _import_data_type = List
    # map byte order name to Numpy symbol
    _IMPORT_BYTE_ORDERS = OrderedDict((
        ("Default order", "="), ("Little endian", "<"), ("Big endian", ">")))
    _import_byte_order = List
    _import_prefix = Str
    _import_feedback = Str
    
    # BrainGlobe panel
    
    _bg_atlases = List
    _bg_atlases_sel = List
    _bg_atlases_table = TabularEditor(
        adapter=BrainGlobeArrayAdapter(), editable=False, auto_resize_rows=True,
        stretch_last_section=False, selected="_bg_atlases_sel")
    _bg_access_btn = Button("Open")
    _bg_remove_btn = Button("Remove")
    _bg_feedback = Str

    # Image viewers

    atlas_eds = []  # open atlas editors
    flipz = True  # True to invert 3D vis along z-axis
    controls_created = Bool(False)
    mpl_fig_active = Any
    scene = None if MlabSceneModel is None else Instance(MlabSceneModel, ())
    scene_3d_shown = False  # 3D Mayavi display shown
    selected_viewer_tab = vis_handler.ViewerTabs.ROI_ED
    select_controls_tab = Int(-1)
    
    # Viewer options
    
    _check_list_3d = List(
        tooltip="Raw: show raw intensity image; limit region to ROI\n"
                "Surface: render as surface; uncheck to render as points\n"
                "Clear: clear scene before drawing new objects\n"
                "Panes: show back and top panes\n"
                "Shadows: show blob shadows as circles")
    _check_list_2d = List(
        tooltip="Filtered: show filtered image after detection in ROI planes\n"
                "Border: margin around ROI planes\n"
                "Seg: segment blobs during detection in ROI planes\n"
                "Grid: overlay a grid in ROI planes\n"
                "MIP: maximum intensity projection in overview images")
    _DEFAULTS_2D = [
        "Filtered", "Border", "Seg", "Grid", "MIP"]
    _planes_2d = List
    _border_on = False  # remembers last border selection
    _DEFAULT_BORDER = np.zeros(3)  # default ROI border size
    _DEFAULTS_PLANES_2D = ["xy", "xz", "yz"]
    _circles_2d = List  # ROI editor circle styles
    _styles_2d = List
    _atlas_ed_options = List(
        tooltip="Labels: show a description when hovering over an atlas label"
                "in\nboth the ROI and Atlas Editors\n"
                "Sync ROI: move to the ROI offset whenever it changes\n"
                "Zoom ROI: zoom into ROI; select Center to center ROI on "
                "crosshairs\n"
                "Crosslines: show lines for orthogonal planes")
    _atlas_ed_options_prev = []  # last atlas ed option settings
    
    #: Choices of planes to show in Plot Editors.
    _PLOT_ED_PLANES: Sequence[str] = ("", "xy", "xz", "yz")
    _atlas_ed_plot_tip = (
        "Plane to show in the {} plot. The blank selection will turn off the\n"
        "plot and expand the other plots to fill the space.")
    _atlas_ed_plot_left = List(tooltip=_atlas_ed_plot_tip.format("left"))
    _atlas_ed_plot_right_up = List(
        tooltip=_atlas_ed_plot_tip.format("upper right"))
    _atlas_ed_plot_right_down = List(
        tooltip=_atlas_ed_plot_tip.format("lower right"))
    _atlas_ed_plot_ignore = False
    
    # atlas labels
    _atlas_label = None
    _structure_scale_low = -1
    _structure_scale_high = 20
    _structure_scale = Int(_structure_scale_high)  # ontology structure levels
    _structure_remap_btn = Button(
        "Remap",
        tooltip="Remap atlas labels to this level")
    _region_name = Str(
        tooltip="Navigate to selected region. Start typing to search.")
    _region_names = Instance(TraitsList)
    # tooltip for both text box and options since options group has no main
    # label to display a tooltip
    _region_id = Str(
        tooltip="IDs: IDs of labels to navigate to. Add +/- to including both\n"
                "sides. Separate multiple IDs by commas.\n"
                "Both sides: include correspondings labels from opposite sides"
                "\nChildren: include all children of the label\n"
                "Append: append label IDs; uncheck to navigate to group\n"
                "Show all: show abbreviations for all visible labels")
    _region_options = List
    
    _mlab_title = None
    _circles_opened_type = None  # enum of circle style for opened 2D plots
    _opened_window_style = None  # 2D plots window style curr open
    _img_region = None
    _PREFIX_BOTH_SIDES = "+/-"
    _camera_pos = None
    _roi_ed_fig = Instance(figure.Figure, ())
    _atlas_ed_fig = Instance(figure.Figure, ())
    
    # Status bar
    
    _status_bar_msg = Str()  # text for status bar
    _prog_pct = Int(0)  # progress bar percentage
    _prog_pct_max = Int(100)  # progress bar max percentage
    _prog_msg = Str()  # progress bar message

    # ROI selector panel
    panel_roi_selector = VGroup(
        VGroup(
            HGroup(
                Item("_filename", label="Image path", style="simple"),
                Item("_filename_btn", show_label=False),
            ),
            HGroup(
                Item("_channel", label="Channels", style="custom",
                     enabled_when="not _rgb",
                     editor=CheckListEditor(
                         name="object._channel_names.selections", cols=8)),
                Item("_rgb", label="RGB",
                     enabled_when="_rgb_enabled and not _merge_chls"),
                Item("_merge_chls", label="Merge", enabled_when="not _rgb"),
            ),
        ),
        VGroup(
            HGroup(
                Item("_main_img_name_avail", label="Intensity", springy=True,
                     editor=CheckListEditor(
                         name="object._main_img_names_avail.selections",
                         format_func=lambda x: x)),
                Item("_main_img_name", show_label=False, springy=True,
                     editor=CheckListEditor(
                         name="object._main_img_names.selections",
                         format_func=lambda x: x)),
                Item("_reload_btn", show_label=False),
            ),
            HGroup(
                Item("_labels_img_name", label="Labels", springy=True,
                     editor=CheckListEditor(
                         name="object._labels_img_names.selections",
                         format_func=lambda x: x)),
                Item("_labels_ref_path", label="Reference", style="simple"),
                Item("_labels_ref_btn", show_label=False),
            ),
            label="Registered Images",
        ),
        VGroup(
            HGroup(
                Item("rois_check_list", show_label=False,
                     editor=CheckListEditor(
                         name="object._rois_selections.selections")),
                label="Regions",
            ),
            HGroup(
                Item("roi_array", label="Size (x,y,z)"),
                Item("_roi_center", label="Center",
                     editor=BooleanEditor()),
            ),
            Item("x_offset",
                 editor=RangeEditor(
                     low_name="x_low",
                     high_name="x_high",
                     mode="slider")),
            Item("y_offset",
                 editor=RangeEditor(
                     low_name="y_low",
                     high_name="y_high",
                     mode="slider")),
            Item("z_offset",
                 editor=RangeEditor(
                     low_name="z_low",
                     high_name="z_high",
                     mode="slider")),
        ),
        VGroup(
            VGroup(
                Item("_check_list_2d", style="custom", label="ROI Editor",
                     editor=CheckListEditor(
                         values=_DEFAULTS_2D, cols=5, format_func=lambda x: x)),
                HGroup(
                    Item("_circles_2d", style="simple", show_label=False,
                         editor=CheckListEditor(
                             values=[e.value for e in
                                     roi_editor.ROIEditor.CircleStyles],
                             format_func=lambda x: x)),
                    Item("_planes_2d", style="simple", show_label=False,
                         editor=CheckListEditor(
                             values=_DEFAULTS_PLANES_2D,
                             format_func=lambda x: "Plane {}".format(x.upper()))),
                    Item("_styles_2d", style="simple", show_label=False,
                         springy=True,
                         editor=CheckListEditor(
                             values=[e.value for e in Styles2D],
                             format_func=lambda x: x)),
                ),
                show_border=True,
            ),
            VGroup(
                HGroup(
                    Item("_atlas_ed_options", style="custom", label="Atlas Editor",
                         editor=CheckListEditor(
                             values=[e.value for e in AtlasEditorOptions],
                             cols=len(AtlasEditorOptions),
                             format_func=lambda x: x)),
                ),
                HGroup(
                    Item("_atlas_ed_plot_left", style="simple", label="Left plane",
                         editor=CheckListEditor(
                             values=_PLOT_ED_PLANES, format_func=lambda x: x)),
                    Item("_atlas_ed_plot_right_up", style="simple",
                         label="Upper right",
                         editor=CheckListEditor(
                             values=_PLOT_ED_PLANES, format_func=lambda x: x)),
                    Item("_atlas_ed_plot_right_down", style="simple",
                         label="Lower right",
                         editor=CheckListEditor(
                             values=_PLOT_ED_PLANES, format_func=lambda x: x)),
                ),
                show_border=True,
            ),
            VGroup(
                Item("_check_list_3d", style="custom", label="3D Viewer",
                     editor=CheckListEditor(
                         values=[e.value for e in Vis3dOptions],
                         cols=len(Vis3dOptions), format_func=lambda x: x)),
                show_border=True,
            ),
        ),
        # give initial focus to text editor that does not trigger any events to
        # avoid inadvertent actions by the user when the window first displays;
        # set width to any small val to get smallest size for the whole panel
        Item("_roi_feedback", style="custom", show_label=False, has_focus=True,
             width=100),
        
        # generate progress bar but give essentially no height since it will
        # be moved to the status bar in the handler
        Item("_prog_pct", show_label=False, height=-2,
             editor=ProgressEditor(min=0, max_name="_prog_pct_max")),
        
        HGroup(
            Item("btn_redraw", show_label=False),
            Item("_btn_save_fig", show_label=False),
            Item("_roi_btn_help", show_label=False),
        ),
        label="ROI",
    )
    
    # blob detections panel
    panel_detect = VGroup(
        HGroup(
            Item("btn_detect", show_label=False),
            Item("_segs_chls", label="Chl", style="custom",
                 editor=CheckListEditor(
                     name="object._segs_chls_names.selections", cols=8)),
        ),
        HGroup(
            Item("btn_save_segments", show_label=False),
            Item("_segs_labels", label="Change labels to",
                 editor=CheckListEditor(
                     values=[str(c) for c in
                             roi_editor.DraggableCircle.BLOB_COLORS.keys()],
                     format_func=lambda x: x)),
            Item("_btn_verifier", show_label=False),
        ),
        HGroup(
            Item("_segs_model_path", label="Classifier model", style="simple"),
            Item("_segs_model_btn", show_label=False),
        ),
        HGroup(
            Item("_segs_visible", style="custom", show_label=False,
                 editor=CheckListEditor(
                     values=[e.value for e in BlobsVisibilityOptions],
                     cols=len(BlobsVisibilityOptions),
                     format_func=lambda x: x)),
            Item("_colocalize", label="Co-localize by",
                 editor=CheckListEditor(
                     values=[e.value for e in ColocalizeOptions],
                     format_func=lambda x: x)),
            Item("_blob_color_style", show_label=False, springy=True,
                 editor=CheckListEditor(
                     values=[e.value for e in BlobColorStyles],
                     format_func=lambda x: x)),
        ),
        Item("scale_detections", label="Scale 3D blobs",
             editor=RangeEditor(
                 low_name="_scale_detections_low",
                 high_name="_scale_detections_high",
                 mode="slider", format_str="%.3g")),
        Item("_segs_mask", label="3D blobs mask size",
             editor=RangeEditor(
                 low_name="_segs_mask_low",
                 high_name="_segs_mask_high",
                 mode="slider")),
        VGroup(
            Item("_segments", editor=segs_table, show_label=False),
            Item("segs_feedback", style="custom", show_label=False),
        ),
        label="Detect",
    )

    # profiles panel
    panel_profiles = VGroup(
        # profile selectors
        VGroup(
            HGroup(
                Item("_profiles_cats", style="simple", show_label=False,
                     editor=CheckListEditor(
                         values=[e.value for e in ProfileCats], cols=1,
                         format_func=lambda x: x)),
                Item("_profiles_chls", label="Chls", style="custom",
                     editor=CheckListEditor(
                         name="object._channel_names.selections", cols=8)),
            ),
            HGroup(
                Item("_profiles_name", show_label=False,
                     editor=CheckListEditor(
                         name="object._profiles_names.selections",
                         format_func=lambda x: x)),
                Item("_profiles_add_btn", show_label=False),
                Item("_profiles_load_btn", show_label=False),
            ),
            label="Profile Selection",
        ),
        
        # selected profile preview and table of profiles
        VGroup(
            # neg height to ignore optimal height
            Item("_profiles_preview", style="custom", show_label=False,
                 height=-100),
            Item("_profiles", editor=_profiles_table, show_label=False),
            label="Profile Preview",
        ),
        
        # combined profiles dict
        VGroup(
            Item("_profiles_combined", style="custom", show_label=False),
            label="Combined profiles",
        ),
        
        # settings/preferences
        VGroup(
            HGroup(
                Item("_profiles_ver", style="readonly",
                     label="MagellanMapper version:"),
            ),
            HGroup(
                Item("_profiles_reset_prefs_btn", show_label=False),
                Item("_profiles_btn_help", show_label=False),
            ),
            label="Settings",
        ),
        label="Profiles",
    )

    # image adjustment panel
    panel_imgadj = VGroup(
        HGroup(
            Item("_imgadj_name", label="Image",
                 editor=CheckListEditor(
                     name="object._imgadj_names.selections")),
            # use EnumEditor for radio buttons
            Item("_imgadj_chls", label="Channel", style="custom",
                 editor=EnumEditor(
                     name="object._imgadj_chls_names.selections", cols=8)),
        ),
        HGroup(
            Item("_imgadj_min", label="Minimum", editor=RangeEditor(
                     low_name="_imgadj_min_low", high_name="_imgadj_min_high",
                     mode="slider", format_str="%.4g")),
            Item("_imgadj_min_auto", label="Auto", editor=BooleanEditor()),
        ),
        HGroup(
            Item("_imgadj_max", label="Maximum", editor=RangeEditor(
                     low_name="_imgadj_max_low", high_name="_imgadj_max_high",
                     mode="slider", format_str="%.4g")),
            Item("_imgadj_max_auto", label="Auto", editor=BooleanEditor()),
        ),
        HGroup(
            Item("_imgadj_brightness", label="Brightness", editor=RangeEditor(
                     low_name="_imgadj_brightness_low",
                     high_name="_imgadj_brightness_high", mode="slider",
                     format_str="%.4g"))
        ),
        HGroup(
            Item("_imgadj_contrast", label="Contrast", editor=RangeEditor(
                     low=0.0, high=2.0, mode="slider", format_str="%.3g")),
        ),
        HGroup(
            Item("_imgadj_alpha", label="Opacity", editor=RangeEditor(
                 low=0.0, high=1.0, mode="slider", format_str="%.3g")),
        ),
        
        HGroup(
            # alpha blending controls
            Item("_imgadj_alpha_blend_check", label="Blend",
                 enabled_when="not _merge_chls"),
            Item("_imgadj_alpha_blend", show_label=False, editor=RangeEditor(
                 low=0.0, high=1.0, mode="slider", format_str="%.3g"),
                 enabled_when="_imgadj_alpha_blend_check"),
        ),
        label="Adjust Image",
    )
    
    # import panel
    panel_import = VGroup(
        VGroup(
            Item("_import_file_btn", label="Multiplane file(s) to import"),
            Item("_import_dir_btn", label="Directory of single-plane images"),
            label="Import File Selection"
        ),
        VGroup(
            Item("_import_paths", editor=_import_table, show_label=False),
        ),
        VGroup(
            Item("_import_res", label="Resolutions (x,y,z)", format_str="%.4g"),
            HGroup(
                Item("_import_mag", label="Objective magnification"),
                Item("_import_zoom", label="Zoom"),
            ),
            label="Microscope Metadata",
        ),
        VGroup(
            Item("_import_shape", label="Shape (chl, x, y, z, time)"),
            HGroup(
                Item("_import_prefix", style="simple", label="Path"),
            ),
            HGroup(
                Item("_import_bit", style="simple", label="Data Bit",
                     editor=CheckListEditor(
                         values=tuple(_IMPORT_BITS.keys()), cols=1)),
                Item("_import_data_type", style="simple", show_label=False,
                     editor=CheckListEditor(
                         values=tuple(_IMPORT_DATA_TYPES.keys()), cols=1)),
                Item("_import_byte_order", style="simple", show_label=False,
                     editor=CheckListEditor(
                         values=tuple(_IMPORT_BYTE_ORDERS.keys()), cols=1)),
            ),
            label="Output image file"
        ),
        HGroup(
            Item("_import_btn", show_label=False,
                 enabled_when="_import_btn_enabled"),
            Item("_import_clear_btn", show_label=False),
        ),
        Item("_import_feedback", style="custom", show_label=False),
        label="Import",
    )

    # BrainGlobe panel
    panel_brain_globe = VGroup(
        Item("_region_name", label="Region",
             editor=EnumEditor(
                 name="object._region_names.selections",
                 completion_mode="popup", evaluate=True)),
        HGroup(
            Item("_region_id", label="IDs",
                 editor=TextEditor(
                     auto_set=False, enter_set=True, evaluate=str)),
            Item("_region_options", style="custom", show_label=False,
                 editor=CheckListEditor(
                     values=[e.value for e in RegionOptions],
                     cols=len(RegionOptions), format_func=lambda x: x)),
        ),
        HGroup(
            Item("_structure_scale", label="Atlas level",
                 editor=RangeEditor(
                     low_name="_structure_scale_low",
                     high_name="_structure_scale_high",
                     mode="slider")),
            Item("_structure_remap_btn", show_label=False),
        ),
        VGroup(
            Item("_bg_atlases", editor=_bg_atlases_table, show_label=False),
            label="BrainGlobe Atlases"
        ),
        HGroup(
            Item("_bg_access_btn", show_label=False),
            Item("_bg_remove_btn", show_label=False),
        ),
        Item("_bg_feedback", style="custom", show_label=False),
        label="Atlases",
    )

    # tabbed panel of options
    panel_options = Tabbed(
        panel_roi_selector,
        panel_detect,
        panel_profiles,
        panel_imgadj,
        panel_import,
        panel_brain_globe,
    )

    # tabbed panel with ROI Editor, Atlas Editor, and Mayavi scene
    # (if installed)
    mayavi_ed = NullEditor if SceneEditor is None else SceneEditor(
        scene_class=MayaviScene)
    panel_figs = Tabbed(
        # set a small width to allow window to be resized down to this size
        Item("_roi_ed_fig", label="ROI Editor", show_label=False,
             editor=MPLFigureEditor(), width=100),
        Item("_atlas_ed_fig", label="Atlas Editor", show_label=False,
             editor=MPLFigureEditor()),
        Item("scene", label="3D Viewer", show_label=False,
             editor=mayavi_ed),
    )

    # icon as a Pyface resource if image file exists
    icon_img = (ImageResource(str(config.ICON_PATH))
                if config.ICON_PATH.exists() else None)
    
    # set up the GUI layout
    view = View(
        # control the HSplit width ratio by setting min widths for an item in
        # each Tabbed view and initial window total width; panel_figs expands
        # to fill the remaining space
        HSplit(
            panel_options,
            panel_figs,
            id=f"{__name__}.panel_split",
        ),
        # initial window width, which can be resized down to the minimum
        # widths of each panel in the HSplit
        width=1200,
        # keeps window top bar within screen in small displays on Windows
        height=600,
        handler=vis_handler.VisHandler(),
        title="MagellanMapper",
        statusbar=[
            "_status_bar_msg",
            # reduce width of progress message so it's next to the bar
            StatusItem("_prog_msg", width=20)],
        resizable=True,
        icon=icon_img,
        # set ID to trigger saving TraitsUI preferences for window size/position
        id=f"{__name__}.{__qualname__}",
    )
    
    def __init__(self):
        """Initialize GUI."""
        HasTraits.__init__(self)
        
        # get saved preferences
        prefs = config.prefs
        
        # set up callback flags
        self._ignore_roi_offset_change = False
        
        # set up ROI Editor option
        self._set_border(True)
        self._circles_2d = [self.validate_pref(
            prefs.roi_circles, roi_editor.ROIEditor.CircleStyles.CIRCLES.value,
            roi_editor.ROIEditor.CircleStyles)]
        self._planes_2d = [self.validate_pref(
            prefs.roi_plane, self._DEFAULTS_PLANES_2D[0],
            self._DEFAULTS_PLANES_2D)]
        self._styles_2d = [self.validate_pref(
            prefs.roi_styles, Styles2D.SQUARE.value, Styles2D)]
        # self._check_list_2d = [self._DEFAULTS_2D[1]]
        
        self._check_list_3d = [
            Vis3dOptions.RAW.value, Vis3dOptions.SURFACE.value]
        if (config.roi_profile["vis_3d"].lower()
                == Vis3dOptions.SURFACE.value.lower()):
            # check "surface" if set in profile
            self._check_list_3d.append(Vis3dOptions.SURFACE.value)
        # self._structure_scale = self._structure_scale_high
        
        # set up atlas region names
        self._region_names = TraitsList()
        self._region_id_map = {}
        self._region_options_prev: Dict[Enum, bool] = {}
        self._region_options = [RegionOptions.INCL_CHILDREN.value]
        
        # set up blobs
        self.blobs = detector.Blobs()
        self._blob_color_style = [BlobColorStyles.ATLAS_LABELS.value]
        model = config.classifier.model
        self._segs_model_path = model if model else ""

        # set up profiles selectors
        self._profiles_cats = [ProfileCats.ROI.value]
        self._update_profiles_names()
        self._ignore_profiles_update = True
        self._init_profiles()
        
        # set up settings controls
        self._profiles_ver = libmag.get_version(True)
        self._profiles_reset_prefs = False

        # set up image import
        self._import_browser_paths: Optional[Sequence[str]] = None
        self._clear_import_files(False)
        
        # set up BrainGlobe atlases
        self._brain_globe_panel: bg_controller.BrainGlobeCtrl \
            = self._setup_brain_globe()

        # ROI margin for extracting previously detected blobs
        self._margin = config.plot_labels[config.PlotLabels.MARGIN]
        if self._margin is None:
            self._margin = (5, 5, 3)  # x,y,z
        
        #: ROI offset at which viewers were drawn. May differ from the
        #: current state of the offset sliders.
        self._drawn_offset: Sequence[int] = self._curr_offset()

        # setup interface for image
        self._ignore_filename = False  # ignore file update trigger
        self._reset_filename = True  # reset when updating loaded image
        if config.filename:
            # show image filename in file selector without triggering update
            self.update_filename(config.filename, True, False)

        # create figs after applying Matplotlib style and theme
        rc_params = None
        if Visualization.is_dark_mode():
            if len(config.rc_params) < 2:
                # change figs theme if no themes have been added by the user
                print("Dark mode detected; applying dark theme to "
                      "Matplotlib figures")
                rc_params = [config.Themes.DARK]
        
        # ROI and Atlas Editors are currently designed for Seaborn style
        plot_2d.setup_style("seaborn", rc_params)
        
        # set up ROI and Atlas Editor figures without constrained layout
        # because of performance impact at least as of Matplotlib 3.2
        self.roi_ed: Optional["roi_editor.ROIEditor"] = None
        self._roi_ed_fig = figure.Figure()
        self._atlas_ed_fig = figure.Figure()
        
        # set up Atlas Editor controls
        self._atlas_ed_options = [
            AtlasEditorOptions.SHOW_LABELS.value,
            AtlasEditorOptions.SYNC_ROI.value,
            AtlasEditorOptions.CROSSHAIRS.value]
        self._atlas_ed_options_prev: Dict[Enum, bool] = {}
        self._atlas_ed_plot_left = [self._PLOT_ED_PLANES[1]]
        self._atlas_ed_plot_right_up = [self._PLOT_ED_PLANES[2]]
        self._atlas_ed_plot_right_down = [self._PLOT_ED_PLANES[3]]
        
        # set up detector controls
        self._segs_visible = [BlobsVisibilityOptions.VISIBLE.value]
        
        # 3D visualization object
        self._vis3d = None
        
        #: Verifier Editor to verify blob classifications.
        self.verifier_ed: Optional["verifier_editor.VerifierEditor"] = None
        
        # set up rest of image adjustment during image setup
        self.stale_viewers = self.reset_stale_viewers()
        self._img3ds = None
        self._imgadj_min_ignore_update: bool = False
        self._imgadj_max_ignore_update: bool = False
        self._imgadj_ignore_update: bool = False
        
        # set up rest of registered images during image setup
        
        # dictionary of registered image suffix without ext to suffix with
        # ext of a matching existing file
        self._reg_img_names: OrderedDict[str, str] = OrderedDict()
        self._main_img_names_avail = TraitsList()
        self._main_img_names = TraitsList()
        self._labels_img_names = TraitsList()
        self._ignore_main_img_name_changes = False

        # prevent prematurely destroying threads
        self._import_thread = None
        self._remap_level_thread = None
        self._annotate_labels_thread = None
        
        # set up image
        self._setup_for_image()

    @staticmethod
    def validate_pref(val, default, choices):
        """Validate a preference setting.
        
        Args:
            val: Preference value to assign.
            default: Default value if ``val`` is invalid.
            choices: Valid preference values.

        Returns:
            ``val`` if a valid value, otherwise ``default``.

        """
        try:
            if issubclass(choices, Enum):
                choices = [e.value for e in choices]
        except TypeError:
            pass
        if val not in choices:
            val = default
        return val

    def _init_channels(self):
        """Initialize channel check boxes for the currently loaded main image.

        """
        # reset channel check boxes, storing selected channels beforehand
        chls_pre = list(self._channel)
        self._channel_names = TraitsList()
        self._segs_chls_names = TraitsList()
        # 1 channel if no separate channel dimension
        num_chls = (1 if config.image5d is None or config.image5d.ndim < 5
                    else config.image5d.shape[4])
        self._channel_names.selections = [str(i) for i in range(num_chls)]

        if config.blobs is not None and config.blobs.blobs is not None:
            # set detection channels to those in loaded blobs
            self._segs_chls_names.selections = [
                str(n) for n in np.unique(detector.Blobs.get_blobs_channel(
                    config.blobs.blobs).astype(int))]
        else:
            # add detection channel selectors for all image channels
            self._segs_chls_names.selections = list(
                self._channel_names.selections)
        
        # pre-select channels for both main and profiles selector
        if not chls_pre and config.channel:
            # use config if all chls were unchecked, eg if setting for 1st time
            self._channel = sorted([
                str(c) for i, c in enumerate(config.channel) if i < num_chls])
        else:
            # select all channels
            self._channel = self._channel_names.selections
        
        # select detection and profile channels to match main channels
        self._segs_chls = list(self._channel)
        self._profiles_chls = self._channel

    def _init_imgadj(self):
        """Initialize image adjustment controls for the currently loaded images.
        
        """
        # create entries for each possible image but only add existing images
        self._img3ds = {
            "Main": config.image5d,
            "Labels": config.labels_img,
            "Borders": config.borders_img}
        self._imgadj_names = TraitsList()
        self._imgadj_names.selections = [
            k for k in self._img3ds.keys() if self._img3ds[k] is not None]
        if self._imgadj_names.selections:
            self._imgadj_name = self._imgadj_names.selections[0]
        self._setup_imgadj_channels()

    @property
    def imgadj_chls(self) -> str:
        """Selected image adjustment channel.
        
        Returns:
            "0" as a string if in RGB mode, or the selected value.

        """
        if self._rgb:
            # to get the first (and only) image
            return "0"
        else:
            return self._imgadj_chls

    def _setup_imgadj_channels(self):
        """Set up channels in the image adjustment panel for the given image."""
        img3d = self._img3ds.get(self._imgadj_name) if self._img3ds else None
        if self._imgadj_name == "Main":
            if self._rgb:
                # treat all channels as one
                chls = ["RGB"]
            else:
                # limit image adjustment channel options to currently selected
                # channels in ROI panel channel selector for main image
                chls = self._channel
        elif img3d is not None:
            # use all channels, or 1 if no channel dimension
            chls = [str(n) for n in (
                range(0, 1 if len(img3d.shape) < 4 else img3d.shape[3]))]
        else:
            chls = []
        
        # populate channels dropdown and select first channel
        self._imgadj_chls_names = TraitsList()
        self._imgadj_chls_names.selections = chls
        if self._imgadj_chls_names.selections:
            self._imgadj_chls = self._imgadj_chls_names.selections[0]

    @on_trait_change("_imgadj_name")
    def _update_imgadj_limits(self):
        img3d = self._img3ds.get(self._imgadj_name)
        if img3d is None: return
        info = libmag.get_dtype_info(img3d)
        self._setup_imgadj_channels()

        # min/max based on near min/max pre-calculated from whole image
        # including all channels, falling back to data type range; cannot
        # used percentile or else need to load whole image from disk
        min_inten = 0
        if config.near_min is not None:
            min_near_min = min(config.near_min)
            if min_near_min < 0:
                # set min to 0 unless near min is < 0
                min_inten = 2 * min_near_min
        # default near max is an array of -1; assume that measured near
        # max values are positive
        max_near_max = -1 if config.near_max is None else max(config.near_max)
        max_inten = info.max if max_near_max < 0 else max_near_max * 2
        self._imgadj_min_low = min_inten
        self._imgadj_min_high = max_inten
        self._imgadj_max_low = min_inten
        self._imgadj_max_high = max_inten

        # range brightness symmetrically to max limit
        self._imgadj_brightness_low = -max_inten
        self._imgadj_brightness_high = max_inten
        
        # update control values
        self.update_imgadj_for_img()

    def _get_mpl_viewers(self) -> TypeList["plot_support.ImageSyncMixin"]:
        """Get Matplotlib-based viewers."""
        viewers = []
        if self.selected_viewer_tab is vis_handler.ViewerTabs.ROI_ED:
            if self.roi_ed:
                viewers.append(self.roi_ed)
        elif self.selected_viewer_tab is vis_handler.ViewerTabs.ATLAS_ED:
            if self.atlas_eds:
                # support multiple Atlas Editors (currently only have one)
                viewers.extend(self.atlas_eds)
        return viewers
    
    def _get_curr_plot_ax_img(self) -> Optional["plot_editor.PlotAxImg"]:
        """Get the first displayed image in the current viewer."""
        viewers = self._get_mpl_viewers()
        plot_ax_img = None
        if viewers:
            # get the first displayed image from the viewer
            plot_ax_img = viewers[0].get_img_display_settings(
                self._imgadj_names.selections.index(self._imgadj_name),
                chl=int(self.imgadj_chls))
        return plot_ax_img
    
    @on_trait_change("_imgadj_chls")
    def update_imgadj_for_img(self):
        """Update image adjustment controls based on the currently selected
        viewer and channel.

        """
        if not self.imgadj_chls:
            # resetting image adjustment channel names triggers update as
            # empty array
            return

        # get currently displayed image
        plot_ax_img = self._get_curr_plot_ax_img()
        if plot_ax_img is None: return
        
        # ignore triggers
        self._imgadj_ignore_update = True
        
        # populate controls with intensity settings
        self._imgadj_brightness = plot_ax_img.brightness
        self._imgadj_contrast = plot_ax_img.contrast
        self._imgadj_alpha = plot_ax_img.alpha

        # populate intensity limits, auto-scaling, and current val (if not auto)
        self._adapt_imgadj_limits(plot_ax_img)
        
        if plot_ax_img.vmin is None:
            self._imgadj_min_auto = True
        else:
            self._imgadj_min = plot_ax_img.ax_img.norm.vmin
            self._imgadj_min_auto = False
        
        if plot_ax_img.vmax is None:
            self._imgadj_max_auto = True
        else:
            self._imgadj_max = plot_ax_img.ax_img.norm.vmax
            self._imgadj_max_auto = False

        # respond to triggers
        self._imgadj_ignore_update = False
    
    def _adapt_imgadj_limits(self, plot_ax_img: "plot_editor.PlotAxImg"):
        """Adapt image adjustment slider limits based on values in the
        given plotted image.
        
        Args:
            plot_ax_img: Plotted image.

        """
        if plot_ax_img is None: return
        norm = plot_ax_img.ax_img.norm
        input_img = plot_ax_img.input_img
        inten_lim = (np.amin(input_img), np.amax(input_img))

        if inten_lim[0] < self._imgadj_max_low:
            # ensure that lower limit is beyond current plane's lower limit;
            # bottom out at 0 unless current low is < 0
            low_thresh = inten_lim[0] if norm.vmin is None else min(
                inten_lim[0], norm.vmin)
            low = 2 * low_thresh if low_thresh < 0 else 0
            self._imgadj_min_low = low
            self._imgadj_max_low = low
        
        high = None
        if inten_lim[1] > self._imgadj_max_high:
            # ensure that upper limit is beyond current plane's limits;
            # cap at 0 if current high is < 0 in case image is fully neg
            high_thresh = inten_lim[1] if norm.vmax is None else max(
                inten_lim[1], norm.vmax)
            high = 2 * high_thresh if high_thresh > 0 else 0
        elif (0 < inten_lim[1] < 0.1 * self._imgadj_max_high
              and self._imgadj_max_high >= 0):
            # reduce upper limit if current max is comparatively very small
            high = 10 * inten_lim[1]
        if high is not None:
            # make brightness symmetric around upper limit
            self._imgadj_min_high = high
            self._imgadj_max_high = high
            self._imgadj_brightness_low = -high
            self._imgadj_brightness_high = high
    
    def _set_inten_min_to_curr(self, plot_ax_img):
        """Set min intensity to current image value."""
        if plot_ax_img is not None:
            vmin = plot_ax_img.ax_img.norm.vmin
            self._adapt_imgadj_limits(plot_ax_img)
            if self._imgadj_min != vmin:
                self._imgadj_min_ignore_update = True
                self._imgadj_min = vmin

    def _set_inten_max_to_curr(self, plot_ax_img):
        """Set max intensity to current image value."""
        if plot_ax_img is not None:
            vmax = plot_ax_img.ax_img.norm.vmax
            self._adapt_imgadj_limits(plot_ax_img)
            if self._imgadj_max != vmax:
                self._imgadj_max_ignore_update = True
                self._imgadj_max = vmax

    @on_trait_change("_imgadj_min")
    def _adjust_img_min(self):
        if self._imgadj_min_ignore_update or self._imgadj_ignore_update:
            self._imgadj_min_ignore_update = False
            return
        self._imgadj_ignore_update = True
        self._imgadj_min_auto = False
        self._imgadj_ignore_update = False
        plot_ax_img = self._adjust_displayed_imgs(minimum=self._imgadj_min)
        
        # # intensity max may have been adjusted to remain >= min
        # self._set_inten_max_to_curr(plot_ax_img)
    
    def _adjust_img_auto(self, mode: str = "min"):
        """Handle changes to the image range auto control.
        
        Args:
            mode: "min" for minimum and "max" for maximum auto modes.

        """
        if self._imgadj_ignore_update:
            # get a currently displayed image
            plot_ax_img = self._get_curr_plot_ax_img()
        else:
            # adjust the image's min or max auto setting
            if mode == "min":
                min_inten = None if self._imgadj_min_auto else self._imgadj_min
                args = dict(minimum=min_inten)
            else:
                max_inten = None if self._imgadj_max_auto else self._imgadj_max
                args = dict(maximum=max_inten)
            plot_ax_img = self._adjust_displayed_imgs(**args)
        
        if plot_ax_img:
            # synchronize intensity controls with the displayed image
            self._set_inten_min_to_curr(plot_ax_img)
            self._set_inten_max_to_curr(plot_ax_img)

    @on_trait_change("_imgadj_min_auto")
    def _adjust_img_min_auto(self):
        self._adjust_img_auto("min")

    @on_trait_change("_imgadj_max_auto")
    def _adjust_img_max_auto(self):
        self._adjust_img_auto("max")

    @on_trait_change("_imgadj_max")
    def _adjust_img_max(self):
        if self._imgadj_max_ignore_update or self._imgadj_ignore_update:
            self._imgadj_max_ignore_update = False
            return
        self._imgadj_ignore_update = True
        self._imgadj_max_auto = False
        self._imgadj_ignore_update = False
        plot_ax_img = self._adjust_displayed_imgs(maximum=self._imgadj_max)
        
        # # intensity min may have been adjusted to remain <= max
        # self._set_inten_min_to_curr(plot_ax_img)

    @on_trait_change("_imgadj_brightness")
    def _adjust_img_brightness(self):
        if self._imgadj_ignore_update: return
        # include contrast to restore its value while adjusting the original img
        self._adjust_displayed_imgs(
            brightness=self._imgadj_brightness, contrast=self._imgadj_contrast)

    @on_trait_change("_imgadj_contrast")
    def _adjust_img_contrast(self):
        if self._imgadj_ignore_update: return
        # include brightness to restore its value while adjusting the orig img
        self._adjust_displayed_imgs(
            brightness=self._imgadj_brightness, contrast=self._imgadj_contrast)

    @on_trait_change("_imgadj_alpha")
    def _adjust_img_alpha(self):
        if self._imgadj_ignore_update: return
        self._adjust_displayed_imgs(alpha=self._imgadj_alpha)

    @on_trait_change("_imgadj_alpha_blend")
    def _adjust_img_alpha_blend(self):
        """Adjust the alpha blending."""
        self._adjust_displayed_imgs(alpha_blend=self._imgadj_alpha_blend)

    @on_trait_change("_imgadj_alpha_blend_check")
    def _adjust_img_alpha_blend_check(self):
        """Toggle alpha bledning."""
        if self._imgadj_alpha_blend_check:
            # turn on alpha blending
            self._adjust_img_alpha_blend()
        else:
            # turn off alpha blending and reset alpha values
            self._adjust_displayed_imgs(alpha_blend=False)

    def _adjust_displayed_imgs(
            self, refresh: bool = False, **kwargs) -> "plot_editor.PlotAxImg":
        """Adjust image display settings for the currently selected viewer.

        Args:
            refresh: True to refresh all Plot Editors; defaults to False.
                Overridden to True if merge channels is on.
            **kwargs: Arguments to update the currently selected viewer.
        
        Returns:
            The last updated axes image plot, assumed to have the same values
            as all the other updated plots, or None if the selected tab does
            not have an axes image, such as an unloaded tab or 3D visualization
            tab.

        """
        plot_ax_img = None
        if self.selected_viewer_tab is vis_handler.ViewerTabs.MAYAVI:
            # update 3D visualization Mayavi/VTK settings
            self._vis3d.update_img_display(**kwargs)
        else:
            # update all Matplotlib-based viewers
            viewers = self._get_mpl_viewers()
            plot_ax_img = None
            
            # always fully refresh when channels are merged
            refresh = refresh or self._merge_chls
            
            for viewer in viewers:
                if not viewer: continue
                
                # update settings for the viewer
                plot_ax_img = viewer.update_imgs_display(
                    self._imgadj_names.selections.index(self._imgadj_name),
                    chl=int(self.imgadj_chls), refresh=refresh, **kwargs)
        
        return plot_ax_img

    @staticmethod
    def is_dark_mode(max_rgb=100):
        """Check whether dark mode is turned on.

        Args:
            max_rgb (int): Max allowed RGB value for the window palette
                in dark mode; defaults to 100.

        Returns:
            bool: True if dark mode is on as inferred by the max RGB
            value of the Qt application window.

        """
        palette = QtWidgets.QApplication.instance().palette()
        return max(palette.color(palette.Window).getRgb()[:3]) < max_rgb

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
        seg_str.append(str(int(detector.Blobs.get_blob_confirmed(seg))))
        seg_str.append(str(int(detector.Blobs.get_blobs_channel(seg))))
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
        segs_transposed = []
        segs_to_delete = []
        offset = self._drawn_offset
        curr_roi_size = self.roi_array[0].astype(int)
        print("Preparing to insert segments to database with border widths {}"
              .format(self.border))
        feedback = ["Preparing segments:"]
        if self.segments is not None:
            for i in range(len(self.segments)):
                seg = self.segments[i]
                # uses absolute coordinates from end of seg
                seg_db = detector.Blobs.blob_for_db(seg)
                if seg[4] == -1 and seg[3] < config.POS_THRESH:
                    # attempts to delete user added segments, where radius
                    # assumed to be < 0, that are no longer selected
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
                detector.Blobs.get_blob_confirmed(segs_transposed_np) == -1, 
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
        exp_name = sqlite.get_exp_name(
            config.img5d.path_img if config.img5d else None)
        exp_id = config.db.select_or_insert_experiment(exp_name)
        roi_id, out = sqlite.select_or_insert_roi(
            config.db.conn, config.db.cur, exp_id, config.series, 
            np.add(offset, self.border).tolist(),
            np.subtract(curr_roi_size, np.multiply(self.border, 2)).tolist())
        sqlite.delete_blobs(
            config.db.conn, config.db.cur, roi_id, segs_to_delete)
        
        # delete the original entry of blobs that moved since replacement
        # is based on coordinates, so moved blobs wouldn't be replaced
        for i in range(len(self._segs_moved)):
            self._segs_moved[i] = detector.Blobs.blob_for_db(
                self._segs_moved[i])
        sqlite.delete_blobs(
            config.db.conn, config.db.cur, roi_id, self._segs_moved)
        self._segs_moved = []
        
        # insert blobs into DB and save ROI in GUI
        sqlite.insert_blobs(
            config.db.conn, config.db.cur, roi_id, segs_transposed)
        
        # insert blob matches
        if self.blobs.blob_matches is not None:
            self.blobs.blob_matches.update_blobs(
                detector.Blobs.shift_blob_rel_coords, offset[::-1])
            config.db.insert_blob_matches(roi_id, self.blobs.blob_matches)
        
        # add ROI to selection dropdown
        roi = sqlite.select_roi(config.db.cur, roi_id)
        self._append_roi(roi, self._rois_dict)
        self._rois_selections.selections = list(self._rois_dict.keys())

        # calculate basic accuracy stats
        print(segs_transposed_np)
        blob_stats = [verifier.meas_detection_accuracy(
            segs_transposed_np, treat_maybes=i)[2] for i in range(3)]
        for i, blob_stat in enumerate(blob_stats):
            if blob_stat is not None:
                feedback.insert(i, blob_stat)
        feedback.extend(("\n", out))

        # provide feedback on the blob insertion and stats
        feedback_str = "\n".join(feedback)
        print(feedback_str)
        self.segs_feedback = feedback_str
        
        if self.roi_ed:
            # reset the ROI Editor's edit flag
            self.roi_ed.edited = False

    def _btn_save_segments_fired(self):
        """Handler to save blobs to database when triggered by Trait."""
        self.save_segs()

    def _reset_segments(self):
        """Resets the saved segments.
        """
        self.segments = None
        self.blobs = detector.Blobs()
        self.segs_in_mask = None
        self.labels = None
        # window with circles may still be open but would lose segments 
        # table and be unsavable anyway; TODO: warn before resetting segments
        self._circles_opened_type = None
        self.segs_feedback = ""
    
    def _update_structure_level(
            self, curr_offset: Sequence[int], curr_roi_size: Sequence[int]):
        """Handle structure level changes.
        
        Args:
            curr_offset: Current ROI offset in ``x, y, z``.
            curr_roi_size: Current ROI size in ``x, y, z``.

        """
        self._atlas_label = None
        if self._mlab_title is not None:
            self._mlab_title.remove()
            self._mlab_title = None
        
        # set level in ROI and Atlas Editors
        level = self.structure_scale
        if self.roi_ed:
            self.roi_ed.set_labels_level(level)
        if self.atlas_eds:
            for ed in self.atlas_eds:
                ed.set_labels_level(level)
        
        if (config.labels_img is not None and config.labels_ref is not None and
                config.labels_ref.ref_lookup is not None and
                curr_offset is not None and curr_roi_size is not None):
            # get atlas label at ROI center
            center = np.add(
                curr_offset, 
                np.around(np.divide(curr_roi_size, 2)).astype(int))
            self._atlas_label = ontology.get_label(
                center[::-1], config.labels_img, config.labels_ref.ref_lookup, 
                config.labels_scaling, level, rounding=True)
            
            if self._atlas_label is not None and self.scene_3d_shown:
                title = ontology.get_label_name(self._atlas_label)
                if title is not None and self.scene is not None:
                    # update title in 3D viewer
                    self._mlab_title = self.scene.mlab.title(title)
    
    def _post_3d_display(
            self, title: str = "magmap3d", show_orientation: bool = True):
        """Show axes and saved ROI parameters after 3D display.
        
        Args:
            title: Path without extension to save file if
                :attr:``config.savefig`` is set to an extension. Defaults to
                "magmap3d".
            show_orientation: True to show orientation axes; defaults to True.
        """
        if self.scene is None: return
        
        if self.scene_3d_shown:
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
                #   and have not found a way to turn them off easily, so
                #   consider turning them off by default and deferring to the
                #   GUI to turn them back on
                self.show_orientation_axes(self.flipz)
        # updates the GUI here even though it doesn't elsewhere for some reason
        if self._rois_selections.selections:
            self.rois_check_list = self._rois_selections.selections[0]
        self._img_region = None
        #print("reset selected ROI to {}".format(self.rois_check_list))
    
    def _get_max_offset(self):
        """Get the maximum image offset based on the ROI controls
        
        Tyypically the same as the shape of the main image.
        
        Returns:
            tuple[int]: Tuple of ``x,y,z``.

        """
        return self.x_high, self.y_high, self.z_high
    
    def _check_roi_position(self):
        # ensure that ROI does not exceed image boundaries
        curr_roi_size = self.roi_array[0].astype(int)
        roi_size_orig = np.copy(curr_roi_size)
        curr_offset = list(self._curr_offset())
        offset_orig = np.copy(curr_offset)
        max_offset = self._get_max_offset()

        # keep offset within bounds
        for i, offset in enumerate(curr_offset):
            if offset >= max_offset[i]:
                curr_offset[i] = max_offset[i] - 1
            elif offset < 0:
                curr_offset[i] = 0
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
        return curr_offset, curr_roi_size, feedback

    def show_3d(self):
        """Show the 3D plot and prepare for detections.
        
        Type of 3D display depends on configuration settings. A lightly 
        preprocessed image will be displayed in 3D, and afterward the 
        ROI will undergo full preprocessing in preparation for detection 
        and 2D filtered displays steps.
        """
        if config.image5d is None:
            print("Main image has not been loaded, cannot show 3D Viewer")
            return
        
        if self.scene is None:
            self.update_status_bar_msg(
                config.format_import_err("mayavi", task="3D viewing"))
            return
        
        # show raw 3D image unless selected not to
        curr_offset, curr_roi_size, feedback = self._check_roi_position()
        if Vis3dOptions.CLEAR.value in self._check_list_3d:
            self._vis3d.clear_scene()
        if Vis3dOptions.RAW.value in self._check_list_3d:
            # show region of interest based on raw image
            self.roi = plot_3d.prepare_roi(
                config.image5d, curr_offset, curr_roi_size)
            
            if Vis3dOptions.SURFACE.value in self._check_list_3d:
                # surface rendering, segmenting to clean up image 
                # if 2D segmentation option checked
                segment = self._DEFAULTS_2D[2] in self._check_list_2d
                self._vis3d.plot_3d_surface(
                    self.roi, config.channel, segment, 
                    self.flipz, curr_offset[::-1])
                self.scene_3d_shown = True
            else:
                # 3D point rendering
                self.scene_3d_shown = self._vis3d.plot_3d_points(
                    self.roi, config.channel, self.flipz, curr_offset[::-1])
        
        # show shadow images around the points if selected
        if Vis3dOptions.PANES.value in self._check_list_3d:
            self._vis3d.plot_2d_shadows(self.roi, self.flipz)
        
        # show title from labels reference if available
        self._update_structure_level(curr_offset, curr_roi_size)

        if feedback:
            self._update_roi_feedback(" ".join(feedback), print_out=True)
        self.stale_viewers[vis_handler.ViewerTabs.MAYAVI] = None
        
        if Vis3dOptions.CLEAR.value in self._check_list_3d:
            # after clearing the scene, re-orient camera to the new surface
            self.orient_camera()
    
    def show_label_3d(self, label_id):
        """Show 3D region of main image corresponding to label ID.
        
        Args:
            label_id: ID of label to display.
        """
        if self.scene is None: return
        
        # get bounding box for label region
        bbox = cv_nd.get_label_bbox(config.labels_img, label_id)
        if bbox is None: return
        shape, slices = cv_nd.get_bbox_region(
            bbox, 10, config.labels_img.shape)
        
        # update GUI dimensions
        self.roi_array = [shape[::-1]]
        self.z_offset, self.y_offset, self.x_offset = [
            slices[i].start for i in range(len(slices))]
        self.scene_3d_shown = True
        
        # show main image corresponding to label region
        # TODO: provide option to show label without main image?
        if isinstance(label_id, (tuple, list)):
            label_mask = np.isin(config.labels_img[tuple(slices)], label_id)
        else:
            label_mask = config.labels_img[tuple(slices)] == label_id
        self.roi = np.copy(config.image5d[0][tuple(slices)])
        self.roi[~label_mask] = 0
        if Vis3dOptions.CLEAR.value in self._check_list_3d:
            self._vis3d.clear_scene()
        offset = self._curr_offset()
        if Vis3dOptions.SURFACE.value in self._check_list_3d:
            # show as surface
            self._vis3d.plot_3d_surface(
                self.roi, config.channel, flipz=self.flipz,
                offset=offset[::-1])
        else:
            # show as points
            self._vis3d.plot_3d_points(
                self.roi, self.scene.mlab, self.flipz, offset[::-1])
        
        # reposition camera to show all objects in the scene
        self.scene.mlab.view(*self.scene.mlab.view()[:3], "auto")
        name = os.path.splitext(os.path.basename(config.filename))[0]
        self._post_3d_display(
            title="label3d_{}".format(name), show_orientation=False)
        
        # turn off stale flag from ROI changes
        self.stale_viewers[vis_handler.ViewerTabs.MAYAVI] = None
    
    @on_trait_change("_main_img_name_avail")
    def _on_main_img_name_avail_changed(self):
        """Respond to main registered image availale suffix change."""
        self._swap_main_img_names(
            self._main_img_name_avail, self._main_img_names_avail,
            self._main_img_names)

    @on_trait_change("_main_img_name")
    def _on_main_img_name_changed(self):
        """Respond to main registered image selected suffix change."""
        self._swap_main_img_names(
            self._main_img_name, self._main_img_names,
            self._main_img_names_avail)
    
    def _swap_main_img_names(self, name_from, names_from, names_to):
        """Swamp main image suffix from one dropdown to another.
        
        Args:
            name_from (str): Name to move.
            names_from (List[str]): List to move name from.
            names_to (List[str]): List to move name to.

        """
        # assume that first item is label with counter and name_from is
        # currently selected item in names_from
        if (not self._ignore_main_img_name_changes
                and name_from != names_from.selections[0]):
            self._ignore_main_img_name_changes = True
            # add item to opposite dropdown but do not select it since
            # re-selecting the selected item does not trigger the callback
            names_to.selections.append(name_from)
            if name_from in names_from.selections:
                # remove item from current dropdown, reverting selection to
                # first item in names_from
                names_from.selections.remove(name_from)
            self._update_main_img_counter_disp()
            self._ignore_main_img_name_changes = False
    
    def _update_main_img_counter_disp(self):
        """Update counter in each main image dropdown for number of images."""
        # display counters of images available to provide feedback when the
        # user shifts an image and to clarify that multiple images can be shown
        self._main_img_names_avail.selections[0] = (
            self._MAIN_IMG_NAME_AVAIL_DEFAULT.format(len(
                self._main_img_names_avail.selections) - 1))
        self._main_img_names.selections[0] = (
            self._MAIN_IMG_NAME_DEFAULT.format(len(
                self._main_img_names.selections) - 1))
    
    def _setup_for_image(self):
        """Setup GUI parameters for the loaded image5d.
        """
        self.reset_stale_viewers()
        self._init_channels()
        self._vis3d = vis_3d.Vis3D(self.scene)
        self._vis3d.fn_update_coords = self.set_offset
        if config.img5d and config.img5d.img is not None:
            image5d = config.img5d.img
            config.img5d.rgb = self._rgb
            
            # TODO: consider subtracting 1 to avoid max offset being 1 above
            # true max, but currently convenient to display size and checked
            # elsewhere; "high_label" RangeEditor setting also does not
            # appear to be working
            self.z_high, self.y_high, self.x_high = image5d.shape[1:4]
            if config.roi_offset is not None:
                # apply user-defined offsets
                self.x_offset, self.y_offset, self.z_offset = config.roi_offset
                self._drawn_offset = self._curr_offset()
            self.roi_array = ([[100, 100, 12]] if config.roi_size is None
                              else [config.roi_size])
            
            # find matching registered images to populate dropdowns
            reg_img_names = OrderedDict()
            for reg_name in config.RegNames:
                # potential registered image path including any prefix
                reg_path = sitk_io.reg_out_path(
                    config.prefix if config.prefix else config.filename,
                    reg_name.value)
                base_path = reg_path[:-len(reg_name.value)]
                
                # match files with modifiers just before ext
                reg_name_globs = glob.glob(f"{libmag.splitext(reg_path)[0]}*")
                for reg_name_glob in reg_name_globs:
                    if libmag.splitext(reg_name_glob)[1] not in sitk_io.EXTS_3D:
                        # skip exts not in 3D list
                        continue
                    # add to list of available suffixes, storing the found
                    # extension to load directly and save to same ext
                    name = reg_name_glob[len(base_path):]
                    name = libmag.get_filename_without_ext(name)
                    reg_img_names[name] = (
                        f"{name}{libmag.splitext(reg_name_glob)[1]}")
            
            # shorten displayed names to avoid expanding panel
            self._reg_img_names = OrderedDict()
            for key_orig, key_short in zip(
                    reg_img_names.keys(),
                    libmag.crop_mid_str(tuple(reg_img_names.keys()), 12)):
                # map shortened name to full path for dropdowns
                self._reg_img_names[key_short] = reg_img_names[key_orig]
                
                # map original to shortened name to find shorted name
                reg_img_names[key_orig] = key_short
            
            # copy names to labels image dropdown
            self._labels_img_names.selections = list(self._reg_img_names.keys())
            self._labels_img_names.selections.insert(0, "")
            
            # set any registered names based on loaded images, defaulting to
            # image5d and no labels
            main_suffixes = [self._MAIN_IMG_NAME_DEFAULT]
            labels_suffix = self._labels_img_names.selections[0]
            main_img_names_avail = list(self._reg_img_names.keys())
            if config.reg_suffixes:
                # select atlas images in dropdowns corresponding to reg suffixes
                suffixes = config.reg_suffixes[config.RegSuffixes.ATLAS]
                if suffixes:
                    if not libmag.is_seq(suffixes):
                        suffixes = [suffixes]
                    for suffix in suffixes:
                        # get shortened from full name
                        suffix_stem = reg_img_names.get(
                            libmag.get_filename_without_ext(suffix))
                        if suffix_stem in main_img_names_avail:
                            # move from available to selected suffixes lists
                            main_suffixes.append(suffix_stem)
                            main_img_names_avail.remove(suffix_stem)
                
                # select first annotation image, ignoring rest
                suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
                if suffix:
                    suffix_stem = reg_img_names.get(
                        libmag.get_filename_without_ext(
                            libmag.get_if_within(suffix, 0, "")))
                    if suffix_stem in self._labels_img_names.selections:
                        labels_suffix = suffix_stem
            
            # show main image lists in two dropdowns, where selecting a suffix
            # from one list immediately moves it to the other 
            main_img_names_avail.insert(
                0, self._MAIN_IMG_NAME_AVAIL_DEFAULT)
            self._main_img_names_avail.selections = main_img_names_avail
            self._main_img_names.selections = main_suffixes
            self._update_main_img_counter_disp()
            
            # select first elements in each dropdown, which display counters
            # of the number of images in each list
            self._main_img_name_avail = self._main_img_names_avail.selections[0]
            self._main_img_name = self._main_img_names.selections[0]
            self._labels_img_name = labels_suffix
            
            # populate labels reference path field
            lbls_ref = config.labels_ref.path_ref if config.labels_ref else ""
            self._labels_ref_path = lbls_ref

            # populate region names combo box
            regions = []
            regions_map = {}
            if config.labels_ref:
                df = config.labels_ref.get_ref_lookup_as_df()
                if df is not None:
                    # truncate names to preserve panel min width; make unique
                    # by adding abbreviations or during cropping
                    abbr = df[config.ABAKeys.ACRONYM.value]
                    abbrs = abbr.unique()
                    is_abbr = len(abbrs) > 0 and abbrs[0]
                    regions = libmag.crop_mid_str(
                        df[config.ABAKeys.NAME.value], 40, not is_abbr)
                    if is_abbr:
                        regions = [f"{r} ({a})" for a, r in zip(abbr, regions)]
                    
                    # map region names to IDs since navigating to the region
                    # will require the ID
                    ids = df[config.ABAKeys.ABA_ID.value]
                    regions_map = {r: i for r, i in zip(regions, ids)}
            self._region_id_map = regions_map
            self._region_names.selections = regions
            
            # enable RGB button only if 3-4 channel
            self._rgb_enabled = (
                    len(image5d.shape) >= 5 and 3 <= image5d.shape[4] <= 4)

        # set up image adjustment controls
        self._init_imgadj()
        
        # set up selector for loading past saved ROIs
        self._rois_dict = {self._ROI_DEFAULT: None}
        img5d = config.img5d
        if config.db is not None and img5d and img5d.path_img is not None:
            self._rois = config.db.get_rois(sqlite.get_exp_name(
                img5d.path_img))
        self._rois_selections = TraitsList()
        if self._rois is not None and len(self._rois) > 0:
            for roi in self._rois:
                self._append_roi(roi, self._rois_dict)
        self._rois_selections.selections = list(self._rois_dict.keys())
        if self._rois_selections.selections:
            self.rois_check_list = self._rois_selections.selections[0]

    def update_filename(
            self, filename: str, ignore: bool = False, reset: bool = True):
        """Update main image filename, triggering image load.
        
        Args:
            filename: Path to image file to load.
            ignore: Set to :attr:`_ignore_filename` flag; deafults to False.
                True to av
            reset: Set the :attr:`_reset_filename` flag; defaults to False.
                True to reset additional image parameters such as registered
                image suffixes.

        """
        # if old and new filename are the same, no update will be triggered,
        # so need to set to non-image first
        self._filename = ""
        
        # update filename, triggering Trait change
        self._reset_filename = reset
        self._ignore_filename = ignore
        self._filename = filename
    
    def open_image(self, filename, new_window=True):
        """Open an image with handling for a new window.
        
        Args:
            filename (str): Path to image file to load.
            new_window (bool): True to load the image in a new app instance
                unless no image is loaded in the current instance.

        """
        if new_window and self._filename:
            # load the image in a completely separate app instance if an
            # image is currently loaded
            popen_args = [sys.executable, filename]
            if not getattr(sys, "frozen", False):
                # run the launch script unless in a frozen environment
                popen_args.insert(1, load_env.__file__)
            subprocess.Popen(popen_args)
        else:
            # load the image in the empty current app window
            self.update_filename(filename)
    
    @observe("_filename_btn")
    def _filename_updated(self, evt):
        """Open a Pyface file dialog to set the main image path."""
        open_dialog = FileDialog(
            action="open", default_path=os.path.dirname(self._filename))
        if open_dialog.open() == OK:
            # get user selected path
            self._filename = open_dialog.path

    @on_trait_change("_filename")
    def _image_path_updated(self):
        """Update the selected filename and load the corresponding image.
        
        Since an original (eg .czi) image can be processed in so many 
        different ways, assume that the user will select the Numpy image 
        file instead of the raw image. Image settings will be constructed 
        from the Numpy image filename. Processed files (eg ROIs, blobs) 
        will not be loaded for now.
        """
        if not self._filename:
            # skip if image path is empty
            return
        
        if self._reset_filename:
            # reset registered atlas settings
            config.reg_suffixes = dict.fromkeys(config.RegSuffixes, None)
            self._labels_ref_path = ""
        self._reset_filename = True
        
        if self._ignore_filename:
            # skip triggered file load, eg if only updating widget
            self._ignore_filename = False
            return

        # load image if possible without allowing import, deconstructing
        # filename from the selected imported image
        filename, offset, size, reg_suffixes = importer.deconstruct_img_name(
            self._filename)
        if filename is not None:
            importer.parse_deconstructed_name(
                filename, offset, size, reg_suffixes)
            np_io.setup_images(
                config.filename, offset=offset, size=size, allow_import=False,
                labels_ref_path=self._labels_ref_path)
            self._setup_for_image()
            self.redraw_selected_viewer()
            self.update_imgadj_for_img()
        else:
            print("Could not parse filename", self._filename)
        
        if config.image5d is None:
            # initiate import setup and direct user to import panel
            print("Could not open {}, directing to import panel"
                  .format(self._filename))
            # must assign before changing tab or else _filename is empty
            self._import_browser = self._filename
            self.select_controls_tab = ControlsTabs.IMPORT.value
    
    @on_trait_change("_reload_btn")
    def _reload_images(self):
        """Reload images to include registered images."""
        # update registered suffixes dict with selections
        reg_suffixes = {
            config.RegSuffixes.ATLAS: None,
            config.RegSuffixes.ANNOTATION: None,
        }
        # skip first element, which serves as a dropdown box label
        atlas_suffixes = self._main_img_names.selections[1:]
        if atlas_suffixes:
            atlas_paths = [self._reg_img_names.get(s) for s in atlas_suffixes]
            if len(atlas_paths) == 1:
                # reduce to str if only one element
                atlas_paths = atlas_paths[0]
            reg_suffixes[config.RegSuffixes.ATLAS] = atlas_paths
        
        if self._labels_img_names.selections.index(self._labels_img_name) != 0:
            # add if not the empty first selection
            reg_suffixes[
                config.RegSuffixes.ANNOTATION] = self._reg_img_names.get(
                    self._labels_img_name)
        config.reg_suffixes.update(reg_suffixes)
        
        # re-setup image
        self.update_filename(self._filename, reset=False)
    
    @on_trait_change("_labels_ref_btn")
    def _labels_ref_path_updated(self):
        """Open a Pyface file dialog with path set to current image directory.
        """
        open_dialog = FileDialog(
            action="open", default_path=os.path.dirname(self._filename))
        if open_dialog.open() == OK:
            # get user selected path
            self._labels_ref_path = open_dialog.path
    
    @on_trait_change("_channel")
    def update_channel(self):
        """Update the selected channel and image adjustment controls.
        """
        if not self._channel:
            # resetting channel names triggers channel update as empty array
            return
        config.channel = sorted([int(n) for n in self._channel])
        self._setup_imgadj_channels()
        print("Changed channel to {}".format(config.channel))
    
    @observe("_rgb", post_init=True)
    def _update_rgb(self, event):
        """Handle changes to the RGB button."""
        if config.img5d:
            # change image RGB setting, which editors reference
            config.img5d.rgb = self._rgb
            _logger.debug("Changed RGB to %s", config.img5d.rgb)
            
            # update image adjustment channel options
            self._update_imgadj_limits()
            
            # redraw viewer in selected RGB mode
            self.redraw_selected_viewer()

    def reset_stale_viewers(self, val=vis_handler.StaleFlags.IMAGE):
        """Reset the stale viewer flags for all viewers.
        
        Args:
            val (:class:`vis_handler.StaleFlags`): Enumeration to set for all
                viewers.

        """
        self.stale_viewers = dict.fromkeys(vis_handler.ViewerTabs, val)
        return self.stale_viewers
    
    @on_trait_change("x_offset,y_offset,z_offset")
    def _update_roi_offset(self):
        """Respond to ROI offset slider changes.
        
        Sets all stale viewer flags to the ROI flag.
        
        """
        print("x: {}, y: {}, z: {}"
              .format(self.x_offset, self.y_offset, self.z_offset))
        self.reset_stale_viewers(vis_handler.StaleFlags.ROI)
        
        # additional callbacks
        if self._ignore_roi_offset_change: return
        if self.selected_viewer_tab is vis_handler.ViewerTabs.ATLAS_ED:
            # immediately move to new offset if sync selected
            self.sync_atlas_eds_coords(check_option=True)

    @on_trait_change("roi_array")
    def _update_roi_array(self):
        """Respond to ROI size array changes."""
        # flag ROI changed for viewers
        self.reset_stale_viewers(vis_handler.StaleFlags.ROI)
        if self._DEFAULTS_2D[4] in self._check_list_2d:
            # update max intensity projection settings
            self._update_mip()
    
    def _update_mip(self, shape=None):
        """Update maximum intensity projection settings for Atlas Editors.
        
        Args:
            shape (Sequence[int]): Number of planes to include in the
                projection in ``z,y,x``; default to None to use the current
                ROI size settings.

        """
        if shape is None:
            shape = self.get_roi_size()
        for ed in self.atlas_eds:
            ed.update_max_intens_proj(shape, True)
        if self.roi_ed is not None:
            self.roi_ed.update_max_intens_proj(shape, True)

    @observe("_merge_chls")
    def _update_merge_channels(self, evt):
        """Handle changes to merge channels control.
        
        Args:
            evt: Event, ignored.

        """
        if self._merge_chls and self._imgadj_alpha_blend_check:
            # turn off alpha blending when merging channels
            self._imgadj_alpha_blend_check = False
        
        viewers = self._get_mpl_viewers()
        for viewer in viewers:
            # update additive blending
            viewer.additive_blend = self._merge_chls
        
        # refresh display images
        self._adjust_displayed_imgs(refresh=True)

    def update_status_bar_msg(self, msg):
        """Update the message displayed in the status bar.

        Args:
            msg (str): Text to display. None will be ignored.

        """
        if msg:
            self._status_bar_msg = msg
    
    @on_trait_change("_structure_scale")
    def _update_structure_scale(self):
        curr_offset = self._curr_offset()
        curr_roi_size = self.roi_array[0].astype(int)
        self._update_structure_level(curr_offset, curr_roi_size)
    
    def _update_prog(self, pct: int, msg: Optional[str]):
        """Update progress bar and message.
        
        Args:
            pct: Percentage completed. Can be -1 to indicate that progress is
                unknown, which will change the bar to a busy indicator.
            msg: Message. If None, the message will not be updated.

        """
        if pct == -1:
            # progress is unknown; change to busy indicator
            self._prog_pct_max = 0
            pct = 0
        elif not self._prog_pct_max:
            # reset progress bar
            self._prog_pct_max = 100
        
        # set progress percentage
        self._prog_pct = pct
        
        if msg:
            # set message
            self._prog_msg = msg
    
    @on_trait_change("_structure_remap_btn")
    def _remap_structure(self):
        """Remap atlas labels to the selected structure level."""
        def remap_success(labels_np):
            # update labels image in editors
            if self.roi_ed:
                self.roi_ed.labels_img = labels_np
            if self.atlas_eds:
                for ed in self.atlas_eds:
                    ed.labels_img = labels_np
            
            # redraw editor
            self.redraw_selected_viewer()
        
        # remap labels image in separate thread
        self._remap_level_thread = atlas_threads.RemapLevelThread(
            self.structure_scale, remap_success, self._update_prog)
        self._remap_level_thread.start()
    
    @property
    def structure_scale(self):
        """Atlas level, where the highest value is None."""
        level = self._structure_scale
        if level == self._structure_scale_high:
            # use drawn label rather than a specific level
            level = None
        return level
        
    @on_trait_change("btn_redraw")
    def _redraw_fired(self):
        """Respond to redraw button presses."""
        self.redraw_selected_viewer()
    
    @staticmethod
    def _open_help_docs(suffix: str = ""):
        """Open help documentation in the default web browser.
        
        Args:
            suffix: Name of page within the main documentation; defaults to
                an empty string to open the docs homepage.

        """
        Qt.QDesktopServices.openUrl(QtCore.QUrl(
            f"{config.DocsURLs.DOCS_URL.value}/{suffix}"))
    
    @on_trait_change("_roi_btn_help")
    def _help_roi(self):
        """Respond to ROI panel help button presses."""
        self._open_help_docs(config.DocsURLs.DOCS_URL_VIEWER.value)

    @on_trait_change("_profiles_btn_help")
    def _help_profiles(self):
        """Respond to profiles panel help button presses."""
        self._open_help_docs(config.DocsURLs.DOCS_URL_SETTINGS.value)
    
    def redraw_selected_viewer(self, clear=True):
        """Redraw the selected viewer.
        
        Args:
            clear (bool): True to clear the ROI and blobs; defaults to True.
        
        """
        def confirm(ed, fn):
            # prompt to save before redrawing if edited
            if ed.edited:
                response = confirmation_dialog.confirm(
                    None, "Edits have not been saved. Save before redrawing?",
                    "Save before redraw?", True, YES)
                if response == YES:
                    fn()
                elif response == CANCEL:
                    return False
            return True
        
        # check for need to save
        cont = True
        if self.selected_viewer_tab is vis_handler.ViewerTabs.ROI_ED:
            if self.roi_ed:
                cont = confirm(self.roi_ed, self.save_segs)
        elif self.selected_viewer_tab is vis_handler.ViewerTabs.ATLAS_ED:
            if self.atlas_eds and self.atlas_eds[0].edited:
                cont = confirm(self.atlas_eds[0], self.atlas_eds[0].save_atlas)
        if not cont:
            # user canceled
            return

        # reload profiles if any profile files have changed and reset ROI
        cli.update_profiles()
        self._drawn_offset = self._curr_offset()
        if clear:
            self.roi = None
            self._reset_segments()
            if self._rois_selections.selections:
                self.rois_check_list = self._rois_selections.selections[0]

        # redraw the currently selected viewer tab
        if self.selected_viewer_tab is vis_handler.ViewerTabs.ROI_ED:
            # disconnect existing ROI and Verifier Editors, which share the fig
            if self.roi_ed:
                self.roi_ed.on_close()
            if self.verifier_ed:
                self.verifier_ed.on_close()
            
            # launch ROI Editor
            self._launch_roi_editor()
        
        elif self.selected_viewer_tab is vis_handler.ViewerTabs.ATLAS_ED:
            if self.atlas_eds:
                # disconnect and reset existing Atlas Editors
                # TODO: re-support multiple Atlas Editor windows
                for ed in self.atlas_eds:
                    ed.on_close()
                self.atlas_eds = []
            
            # launch Atlas Editor
            self.launch_atlas_editor()
        
        elif self.selected_viewer_tab is vis_handler.ViewerTabs.MAYAVI:
            # launch 3D Viewer
            self.show_3d()
            self._post_3d_display()
    
    @on_trait_change("scene.activated")
    def orient_camera(self):
        """Provide a default camera orientation with orientation axes.

        """
        if self.scene is None: return
        
        view = self.scene.mlab.view(*self.scene.mlab.view()[:3], "auto")
        roll = self.scene.mlab.roll(-175)
        if self.scene_3d_shown:
            self.show_orientation_axes(self.flipz)
        #self.scene.mlab.outline() # affects zoom after segmenting
        #self.scene.mlab.axes() # need to adjust units to microns
        print("Scene activated with view:", view, "roll:", roll)
    
    def show_orientation_axes(self, flipud: bool = False):
        """Show orientation axes with option to flip z-axis.
        
        Allows adjusting z-axis to match handedness in Matplotlib images with
        z increasing upward.
        
        Args:
            flipud: True to invert z-axis, which also turns off arrowheads;
                defaults to True.
        
        """
        if self.scene is None: return
        
        orient = self.scene.mlab.orientation_axes()
        if flipud:
            # flip z-axis and turn off now upside-down arrowheads
            orient.axes.total_length = [1, 1, -1]
            orient.axes.cone_radius = 0
    
    @on_trait_change("scene.busy")
    def _scene_changed(self):
        if self.scene is None: return
        
        # show camera position after roll changes; only use roll for
        # simplification since almost any movement involves a roll change
        roll = self.scene.mlab.roll()
        if self._camera_pos is None or self._camera_pos["roll"] != roll:
            self._camera_pos = {"view": self.scene.mlab.view(), "roll": roll}
            print("camera:", self._camera_pos)

    @on_trait_change("btn_detect")
    def _blob_detection_fired(self):
        """Detect blobs when triggered by a button."""
        self.detect_blobs()
    
    def detect_blobs(self, segs=None, blob_matches=None):
        """Detect blobs within the current ROI.
        
        Args:
            segs (:obj:`np.ndarray`): Blobs to display, typically loaded
                from a database. Defaults to None, in which case blobs will
                be taken from a :attr:`config.blobs` if available or detected
                directly from the image.
            blob_matches (List[:obj:`sqlite.BlobMatch`): Sequence of blob
                matches; defaults to None.

        """
        if config.image5d is None:
            print("Main image has not been loaded, cannot show detect blobs")
            return
        self._reset_segments()
        self.blobs.blob_matches = blob_matches
        cli.update_profiles()
        
        # get selected channels for blob detection
        chls = sorted([int(n) for n in self._segs_chls])
        if len(chls) == 0:
            # provide feedback if no channels are selected and return
            self.segs_feedback = "Please select channels for blob detection"
            msg = "Please select channels in the Detect tab to show blobs"
            self.update_status_bar_msg(msg)
            self._update_roi_feedback(msg)
            return
        
        # extract ROI
        self._segs_visible = [BlobsVisibilityOptions.VISIBLE.value]
        offset = self._curr_offset()
        roi_size = self.roi_array[0].astype(int)
        self.roi = plot_3d.prepare_roi(config.image5d, offset, roi_size)
        
        if not libmag.is_binary(self.roi):
            # preprocess ROI in prep for showing filtered 2D view and blob
            # detectionsn; include all channels in case of spectral unmixing
            self.roi = plot_3d.saturate_roi(self.roi)
            self.roi = plot_3d.denoise_roi(self.roi)
        else:
            libmag.printv(
                "binary image detected, will not preprocess")

        # collect segments in ROI and padding region, ensuring coordinates
        # are relative to offset
        colocs = None
        if config.blobs is None or config.blobs.blobs is None:
            # on-the-fly blob detection, which includes border but not
            # padding region; already in relative coordinates
            roi = self.roi
            if config.roi_profile["thresholding"]:
                # thresholds prior to blob detection
                roi = plot_3d.threshold(roi)
            segs_all = detector.detect_blobs(roi, chls)
            self.blobs.blobs = segs_all
            
            if ColocalizeOptions.MATCHES.value in self._colocalize:
                # match blobs between two channels
                verify_tol = np.multiply(
                    detector.calc_overlap(),
                    config.roi_profile["verify_tol_factor"])
                matches = colocalizer.colocalize_blobs_match(
                    self.blobs, np.zeros(3, dtype=int), roi_size,
                    verify_tol, np.zeros(3, dtype=int))
                if matches and len(matches) > 0:
                    # TODO: include all channel combos
                    self.blobs.blob_matches = matches[tuple(matches.keys())[0]]
        else:
            # get all previously processed blobs in ROI plus additional
            # padding region to show surrounding blobs
            # TODO: set segs_all to None rather than empty list if no blobs?
            print("Selecting blobs in ROI from loaded blobs")
            segs_all, mask = detector.get_blobs_in_roi(
                config.blobs.blobs, offset, roi_size, self._margin)
            self.blobs.blobs = segs_all
            
            # shift coordinates to be relative to offset
            segs_all[:, :3] = np.subtract(segs_all[:, :3], offset[::-1])
            segs_all = self.blobs.format_blobs()
            segs_all, mask_chl = self.blobs.blobs_in_channel(
                segs_all, chls, return_mask=True)
            self.blobs.blobs = segs_all
            
            if ColocalizeOptions.MATCHES.value in self._colocalize:
                # get blob matches from whole-image match colocalization,
                # scaling the ROI to original blob space since blob matches
                # are not scaled during loading
                roi = [np.divide(a, config.blobs.scaling[:3])
                       for a in (offset[::-1], roi_size[::-1])]
                matches = colocalizer.select_matches(config.db, chls, *roi)
                
                # TODO: include all channel combos
                if matches is not None and len(matches) > 0:
                    # shift blobs to relative coordinates
                    matches = matches[tuple(matches.keys())[0]]
                    shift = [n * -1 for n in roi[0]]
                    matches.update_blobs(
                        self.blobs.shift_blob_rel_coords, shift)
                    matches.update_blobs(
                        self.blobs.multiply_blob_rel_coords,
                        config.blobs.scaling[:3])
                    matches.get_mean_coords()
                    self.blobs.blob_matches = matches
                _logger.debug(
                    f"Loaded blob matches:\n{self.blobs.blob_matches}")
            
            elif (ColocalizeOptions.INTENSITY.value in self._colocalize
                  and config.blobs.colocalizations is not None
                  and segs is None):
                # get corresponding blob co-localizations unless showing
                # blobs from database, which do not have colocs
                colocs = config.blobs.colocalizations[mask][mask_chl]
        _logger.debug(
            f"All blobs:\n{segs_all}\nTotal blobs: "
            f"{0 if segs_all is None else len(segs_all)}")
        
        if segs is not None:
            # segs are typically loaded from DB for a sub-ROI within the
            # current ROI, so fill in the padding area from segs_all
            # TODO: remove since blobs from different sources may be confusing?
            _, segs_in_mask = detector.get_blobs_in_roi(
                segs_all, np.zeros(3),
                roi_size, np.multiply(self.border, -1))
            segs_outside = segs_all[np.logical_not(segs_in_mask)]
            print("segs_outside:\n{}".format(segs_outside))
            segs[:, :3] = np.subtract(segs[:, :3], offset[::-1])
            blobs_segs = detector.Blobs(segs)
            blobs_segs.format_blobs()
            segs = blobs_segs.blobs_in_channel(blobs_segs.blobs, chls)
            segs_all = np.concatenate((segs, segs_outside), axis=0)
            self.blobs.blobs = segs_all
            
        if segs_all is not None and len(segs_all) > 0:
            # set confirmation flag to user-selected label for any
            # un-annotated blob
            confirmed = self.blobs.get_blob_confirmed(segs_all)
            confirmed[confirmed == -1] = self._segs_labels[0]
            self.blobs.set_blob_confirmed(segs_all, confirmed)
            
            # convert segments to visualizer table format and plot
            self.segments = self.blobs.shift_blob_abs_coords(
                segs_all, offset[::-1])
            self.blobs.blobs = self.segments
            if colocs is not None:
                self.blobs.colocalizations = colocs
            
            # make blobs mask and colormap for ROI Editor and 3D viewers
            self.segs_in_mask = detector.get_blobs_in_roi(
                self.segments, np.zeros(3),
                roi_size, np.multiply(self.border, -1))[1]
            alpha = 170
            if self._blob_color_style[0] is BlobColorStyles.UNIQUE.value:
                # unique color for each blob
                num_colors = len(self.segs_in_mask)
                self.segs_cmap = colormaps.discrete_colormap(
                    num_colors if num_colors >= 2 else 2, alpha, True,
                    config.seed)
            elif (self._blob_color_style[0]
                    is BlobColorStyles.ATLAS_LABELS.value
                    and config.labels_img is not None):
                
                def get_atlas_cmap(coords):
                    # get colors of corresponding atlas labels
                    if coords is None: return None
                    blob_ids = ontology.get_label_ids_from_position(
                        coords.astype(int), config.labels_img)
                    atlas_cmap = config.cmap_labels(
                        config.cmap_labels.convert_img_labels(blob_ids))
                    atlas_cmap[:, :3] *= 255
                    atlas_cmap[:, 3] *= alpha
                    return atlas_cmap
                
                # set up colormaps for blobs and blob matches
                self.segs_cmap = get_atlas_cmap(
                    self.blobs.get_blob_abs_coords(
                        segs_all[self.segs_in_mask]))
                if (self.blobs.blob_matches is not None and
                        self.blobs.blob_matches.coords is not None):
                    self.blobs.blob_matches.cmap = get_atlas_cmap(
                        np.add(self.blobs.blob_matches.coords, offset[::-1]))
            
            else:
                # default to color by channel
                blob_chls = self.blobs.get_blobs_channel(
                    segs_all[self.segs_in_mask]).astype(int)
                cmap = colormaps.discrete_colormap(
                    max(blob_chls) + 1, alpha, True, config.seed)
                self.segs_cmap = cmap[blob_chls]
        
        if self._DEFAULTS_2D[2] in self._check_list_2d:
            blobs = self.segments[self.segs_in_mask]
            # 3D-seeded watershed segmentation using detection blobs
            '''
            # could get initial segmentation from r-w
            walker = segmenter.segment_rw(
                self.roi, chls, erosion=1)
            '''
            self.labels = segmenter.segment_ws(
                self.roi, chls, None, blobs)
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
                self.roi, chls, beta=100000,
                blobs=blobs, remove_small=min_size)
            '''
        #detector.show_blob_surroundings(self.segments, self.roi)
        
        if self._segs_model_path and self.blobs.blobs is not None:
            try:
                # classify blobs if model is set
                classifier.classify_blobs(
                    self._segs_model_path, config.image5d, offset[::-1],
                    roi_size[::-1], chls, self.blobs, blobs_relative=True)
            except ModuleNotFoundError as e:
                _logger.exception(e)
                self.segs_feedback += f"\n{e.msg}"
        
        if (self.selected_viewer_tab is vis_handler.ViewerTabs.ROI_ED or
                self.selected_viewer_tab is vis_handler.ViewerTabs.MAYAVI and
                self.stale_viewers[vis_handler.ViewerTabs.MAYAVI]):
            # currently must redraw ROI Editor to include blobs; redraw
            # 3D viewer if stale and only if shown to limit performance impact
            # of 3D display; Atlas Editor does not show blobs
            self.redraw_selected_viewer(clear=False)
        
        if (self.selected_viewer_tab is vis_handler.ViewerTabs.MAYAVI or
                not self.stale_viewers[vis_handler.ViewerTabs.MAYAVI]):
            # show 3D blobs if 3D viewer is showing; if not, add to 3D display
            # if it is not stale so blobs do not need to be redetected to show
            self.show_3d_blobs()
        
        if ColocalizeOptions.INTENSITY.value in self._colocalize:
            # perform intensity-based colocalization
            self._colocalize_blobs()
    
    def show_3d_blobs(self):
        """Show blobs as spheres in 3D viewer."""
        if self.scene is None or self.segments is None or len(
                self.segments) < 1:
            return
        
        # get blobs in ROI and display as spheres in Mayavi viewer
        roi_size = self.roi_array[0].astype(int)
        show_shadows = Vis3dOptions.SHADOWS.value in self._check_list_3d
        scale, mask_size = self._vis3d.show_blobs(
            self.blobs, self.segs_in_mask, self.segs_cmap,
            self._curr_offset()[::-1], roi_size[::-1], show_shadows, self.flipz)
        
        # set the max scaling based on the starting value
        self._scale_detections_high = scale * 5
        self.scale_detections = scale
        
        # set the blob mask size
        self._segs_mask_high = mask_size * 5
        print("mask size", mask_size)
        self._segs_mask = mask_size

    @on_trait_change("_colocalize")
    def _colocalize_blobs(self):
        """Toggle blob co-localization label visibility.
        
        Return immediately if blobs have not been detected. Find and display
        co-localizations if they have not been found yet. Turn off the
        labels if the visibility flag is set to False.

        """
        if self.blobs.blobs is None: return
        if self.blobs.colocalizations is None:
            self.blobs.colocalizations = colocalizer.colocalize_blobs(
                self.roi, self.blobs.blobs)
        if self.roi_ed:
            self.roi_ed.show_colocalized_blobs(
                ColocalizeOptions.INTENSITY.value in self._colocalize)
    
    @on_trait_change("_segs_model_btn")
    def _segs_model_path_updated(self):
        """Open a Pyface file dialog with path set to the classifer model path.
        """
        open_dialog = FileDialog(
            action="open", default_path=os.path.dirname(self._filename))
        if open_dialog.open() == OK:
            # get user selected path
            self._segs_model_path = open_dialog.path
    
    @on_trait_change("_segs_visible")
    def _update_blob_visibility(self):
        """Change blob visibilty based on toggle check box."""
        if self.roi_ed:
            self.roi_ed.set_circle_visibility(
                BlobsVisibilityOptions.VISIBLE.value in self._segs_visible)
    
    @on_trait_change('scale_detections')
    def update_scale_detections(self):
        """Updates the glyph scale factor.
        """
        for glyph in (self._vis3d.blobs3d_in, self._vis3d.matches3d):
            if glyph is not None:
                glyph.glyph.glyph.scale_factor = self.scale_detections

    @on_trait_change("_segs_mask")
    def _segs_mask_changed(self):
        """Update the glyph mask for fraction of blobs to show."""
        for glyph in (self._vis3d.blobs3d_in, self._vis3d.matches3d):
            if glyph is not None:
                glyph.glyph.mask_points.on_ratio = self._segs_mask

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
            self._roi_feedback = "Reset circles after saving full annotations"

    def _atlas_ed_close_listener(self, evt, atlas_ed):
        """Handle Atlas Editor close events.

        Args:
            evt (:obj:`matplotlib.backend_bases.Event`): Event.
            atlas_ed (:obj:`gui.atlas_editor.AtlasEdtor`): Atlas editor
                that was closed.

        """
        self.atlas_eds[self.atlas_eds.index(atlas_ed)] = None

    @on_trait_change("controls_created")
    def _post_controls_created(self):
        # populate Matplotlib figure once controls have been created,
        # at which point the figure will allow connections
        self._launch_roi_editor()
        self.update_imgadj_for_img()

    def _add_mpl_fig_handlers(self, fig):
        # add additional event handlers for Matplotlib figures
        fig.canvas.mpl_connect("figure_enter_event", self._on_mpl_fig_enter)
        fig.canvas.mpl_connect("figure_leave_event", self._on_mpl_fig_leave)

    def _on_mpl_fig_enter(self, event):
        # event handler for entering a figure, storing the figure as shown
        # print("entered fig", event.canvas.figure)
        self.mpl_fig_active = event.canvas.figure

    def _on_mpl_fig_leave(self, event):
        # event handler for leaving a figure, resetting the shown figure
        # print("left fig", event.canvas.figure)
        self.mpl_fig_active = None

    def _launch_roi_editor(self):
        """Handle ROI Editor button events."""
        if config.image5d is None:
            print("Main image has not been loaded, cannot show ROI Editor")
            return
        
        if (not self._circles_opened_type 
                or self._circles_opened_type ==
                roi_editor.ROIEditor.CircleStyles.NO_CIRCLES):
            # set opened window type if not already set or non-editable window
            self._circles_opened_type = roi_editor.ROIEditor.CircleStyles(
                self._circles_2d[0])
        self._opened_window_style = self._styles_2d[0]
        
        # shows 2D plots
        curr_offset, curr_roi_size, feedback = self._check_roi_position()
        self._update_roi_feedback(" ".join(feedback))

        # update verify flag
        roi_editor.verify = self._DEFAULTS_2D[1] in self._check_list_2d
        roi = None
        if self._DEFAULTS_2D[0] in self._check_list_2d:
            print("showing processed 2D images")
            # denoised ROI processed during 3D display
            roi = self.roi
            if config.roi_profile["thresholding"]:
                # thresholds prior to blob detection
                roi = plot_3d.threshold(roi)
        
        blobs_truth_roi = None
        if config.truth_db is not None:
            # collect truth blobs from the truth DB if available
            blobs_truth_roi, _ = detector.get_blobs_in_roi(
                config.truth_db.blobs_truth, curr_offset, curr_roi_size, 
                self._margin)
            transpose = np.zeros(blobs_truth_roi.shape[1])
            transpose[0:3] = curr_offset[::-1]
            blobs_truth_roi = np.subtract(blobs_truth_roi, transpose)
            blobs_truth_roi[:, 5] = blobs_truth_roi[:, 4]
            #print("blobs_truth_roi:\n{}".format(blobs_truth_roi))
        
        # create ROI Editor
        filename_base = importer.filename_to_base(
            config.filename, config.series)
        grid = self._DEFAULTS_2D[3] in self._check_list_2d
        stack_args = (
            self.update_segment, filename_base, config.channel,
            curr_roi_size, curr_offset, self.segs_in_mask,
            self.segs_cmap, self._roi_ed_close_listener,
            # additional args with defaults
            self._full_border(self.border))
        roi_ed = roi_editor.ROIEditor(
            config.img5d, config.labels_img, self._img_region,
            self.show_label_3d, self.update_status_bar_msg)
        self.roi_ed = roi_ed
        
        roi_ed.plane = self._planes_2d[0].lower()
        if self._DEFAULTS_2D[4] in self._check_list_2d:
            # set MIP for the current plane
            roi_ed.update_max_intens_proj(curr_roi_size)
        roi_ed.zoom_shift = config.plot_labels[config.PlotLabels.ZOOM_SHIFT]
        roi_ed.fn_update_coords = self.set_offset
        roi_ed.fn_redraw = self.redraw_selected_viewer
        roi_ed.blobs = self.blobs
        roi_ed.additive_blend = self._merge_chls
        roi_cols = libmag.get_if_within(
            config.plot_labels[config.PlotLabels.LAYOUT], 0)
        stack_args_named = {
            "roi": roi, "labels": self.labels, "blobs_truth": blobs_truth_roi, 
            "circles": roi_editor.ROIEditor.CircleStyles(self._circles_2d[0]),
            "grid": grid,
            "roi_cols": roi_cols,
            "fig": self._roi_ed_fig,
            "region_name": ontology.get_label_name(self._atlas_label),
        }
        if self._styles_2d[0] == Styles2D.SQUARE_3D.value:
            # layout for square ROIs with 3D screenshot for square-ish fig
            screenshot = None
            if self.scene_3d_shown and self.screen:
                screenshot = self.scene.mlab.screenshot(
                    mode="rgba", antialiased=True)
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, mlab_screenshot=screenshot)
        elif self._styles_2d[0] == Styles2D.SINGLE_ROW.value:
            # single row
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=2, 
                single_roi_row=True, 
                z_level=roi_ed.ZLevels.MIDDLE)
        elif self._styles_2d[0] == Styles2D.WIDE.value:
            # layout for wide ROIs, which shows only one overview plot
            stack_args_named["roi_cols"] = 7
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named)
        elif self._styles_2d[0] == Styles2D.ZOOM3.value:
            # 3 level zoom overview plots with specific multipliers
            roi_ed.plot_2d_stack(
                *stack_args, **stack_args_named, zoom_levels=3)
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
        roi_ed.set_labels_level(self.structure_scale)
        roi_ed.set_show_labels(
            AtlasEditorOptions.SHOW_LABELS.value in self._atlas_ed_options)
        self._show_all_labels(
            [self.roi_ed], RegionOptions.SHOW_ALL.value in self._region_options)
        self._add_mpl_fig_handlers(roi_ed.fig)
        self.stale_viewers[vis_handler.ViewerTabs.ROI_ED] = None
        
        # store selected 2D options
        config.prefs.roi_circles = self._circles_2d[0]
        config.prefs.roi_plane = self._planes_2d[0]
        config.prefs.roi_styles = self._styles_2d[0]

    def launch_atlas_editor(self):
        if config.image5d is None:
            print("Main image has not been loaded, cannot show Atlas Editor")
            return
        # atlas editor; need to retain ref or else instance callbacks
        # created within AtlasEditor will be garbage collected
        title = config.filename
        if self.atlas_eds:
            # distinguish multiple Atlas Editor windows with number since
            # using the same title causes the windows to overlap
            title += " ({})".format(len(self.atlas_eds) + 1)
        atlas_ed = atlas_editor.AtlasEditor(
            config.img5d, config.labels_img, config.channel,
            self._curr_offset(center=False), self._atlas_ed_close_listener,
            config.borders_img, self.show_label_3d, title,
            self._refresh_atlas_eds, self._atlas_ed_fig,
            self.update_status_bar_msg)
        atlas_ed.additive_blend = self._merge_chls
        self.atlas_eds.append(atlas_ed)
        
        if self._DEFAULTS_2D[4] in self._check_list_2d:
            # show max intensity projection planes based on ROI size
            atlas_ed.update_max_intens_proj(self.get_roi_size())
        atlas_ed.fn_update_coords = self.set_offset
        
        # set Plot Editor planes from dropdowns
        atlas_ed.planes = (
            self._atlas_ed_plot_left[0],
            self._atlas_ed_plot_right_up[0],
            self._atlas_ed_plot_right_down[0])
        
        # show the Atlas Editor
        atlas_ed.show_atlas()
        atlas_ed.set_show_labels(
            AtlasEditorOptions.SHOW_LABELS.value in self._atlas_ed_options)
        atlas_ed.set_labels_level(self.structure_scale)
        atlas_ed.show_labels_annots(
            RegionOptions.SHOW_ALL.value in self._region_options)
        self._show_all_labels(
            [atlas_ed], RegionOptions.SHOW_ALL.value in self._region_options)
        self._add_mpl_fig_handlers(atlas_ed.fig)
        self.stale_viewers[vis_handler.ViewerTabs.ATLAS_ED] = None
    
    def sync_atlas_eds_coords(self, coords=None, check_option=False):
        """Synchronize Atlas Editors to ROI offset.
        
        Args:
            coords (Sequence[int]): ROI offset in ``z,y,x``; defaults to None
                to find from the current ROI controls.
            check_option (bool): True to synchronize only if the corresponding
                Atlas Editor is selected; defaults to False.

        """
        if (check_option and AtlasEditorOptions.SYNC_ROI.value
                not in self._atlas_ed_options):
            return
        if coords is None:
            # Atlas Editor uses offset as-is, without centering
            coords = self._curr_offset(False)[::-1]
        for ed in self.atlas_eds:
            if ed is None: continue
            ed.update_coords(coords)
    
    def _refresh_atlas_eds(self, ed_ignore):
        """Callback handler to refresh all other Atlas Editors

        Args:
            ed_ignore (:obj:`gui.atlas_editor.AtlasEditor`): Atlas Editor
                to not refresh, typically the calling editor.

        """
        for ed in self.atlas_eds:
            if ed is None or ed is ed_ignore: continue
            ed.refresh_images()
    
    @staticmethod
    def get_changed_options(
            curr_options: Sequence[str], prev_options: Dict[Enum, bool],
            enum_options: Type[Enum]
    ) -> Tuple[Dict[Enum, bool], Dict[Enum, bool]]:
        """Get dicts of the options and whether they have changed.

        Args:
            curr_options: Current options as a list of option names.
            prev_options: Previous options as a dictionary of Enum options
                to boolean values.
            enum_options: Enumeration of options.

        Returns:
            Tuple of:
            - ``options``: ``prev_options`` but for the current state.
            - ``changed``: Dictionary with same keys as ``options`` but
              values of whether the state changed compared to previously.

        """
        # option is true if it's present in the current options
        options = {e: e.value in curr_options for e in enum_options}
    
        # option changed if it's value differs from previous options
        changed = {e: e in prev_options and options[e] != prev_options[e]
                   for e in options}
        return options, changed

    @on_trait_change("_atlas_ed_options")
    def _atlas_ed_options_changed(self):
        """Respond to atlas editor option changes."""
        # get boolean dicts of the options and whether they have changed
        options, changed = self.get_changed_options(
            self._atlas_ed_options, self._atlas_ed_options_prev,
            AtlasEditorOptions)
        
        if self.roi_ed:
            if changed[AtlasEditorOptions.SHOW_LABELS]:
                # toggle showing atlas labels, applied during Plot Editor
                # mouseovers/refreshes
                self.roi_ed.set_show_labels(
                    options[AtlasEditorOptions.SHOW_LABELS])
        
        if self.atlas_eds:
            for atlas_ed in self.atlas_eds:
                if changed[AtlasEditorOptions.SHOW_LABELS]:
                    # toggle showing atlas labels
                    atlas_ed.set_show_labels(
                        options[AtlasEditorOptions.SHOW_LABELS])
                
                if changed[AtlasEditorOptions.ZOOM_ROI]:
                    # toggle ROI zoom
                    self._zoom_atlas_ed(
                        atlas_ed, options[AtlasEditorOptions.ZOOM_ROI])

                if changed[AtlasEditorOptions.CROSSHAIRS]:
                    # toggle Atlas Editor crosslines
                    atlas_ed.set_show_crosslines(
                        options[AtlasEditorOptions.CROSSHAIRS])
                    for plot_ed in atlas_ed.plot_eds.values():
                        plot_ed.draw_crosslines()
        
        if (self.selected_viewer_tab is vis_handler.ViewerTabs.ATLAS_ED
                and changed[AtlasEditorOptions.SYNC_ROI]):
            # move visible Atlas Editor to ROI offset if sync selected;
            # otherwise, defer sync to tab selection handler
            self.sync_atlas_eds_coords(check_option=True)
        
        self._atlas_ed_options_prev = options
    
    def _zoom_atlas_ed(self, atlas_ed, zoom):
        """Zoom the Atlas Editor into the ROI.
        
        The ROI offset is normally at the crosshairs unless the ROI offset
        is treated as the ROI center, in which case the crosshairs remain
        in the same place but the retrieved offset is shifted to the ROI's
        upper left corner.
        
        """
        if zoom:
            # zoom into the ROI
            offset = self._curr_offset()
            shape = self.roi_array[0].astype(int)
        else:
            # zoom out to the full image
            offset = np.zeros(3, dtype=int)
            shape = self._get_max_offset()
        atlas_ed.view_subimg(offset[::-1], shape[::-1])
    
    @observe("_atlas_ed_plot_left")
    @observe("_atlas_ed_plot_right_up")
    @observe("_atlas_ed_plot_right_down")
    def _atlas_ed_plot_changed(self, evt):
        """Handler for Atlas Editor plot plane dropdowns."""
        if self._atlas_ed_plot_ignore: return
        # get plane selections
        viewers = dict(
            _atlas_ed_plot_left=self._atlas_ed_plot_left,
            _atlas_ed_plot_right_up=self._atlas_ed_plot_right_up,
            _atlas_ed_plot_right_down=self._atlas_ed_plot_right_down,
        )
        
        self._atlas_ed_plot_ignore = True
        for key, val in viewers.items():
            if evt.new and evt.new[0] and val == evt.new and key != evt.name:
                # for a non-empty new selection already in another dropdown,
                # swap it with the old selection
                if key == "_atlas_ed_plot_left":
                    self._atlas_ed_plot_left = evt.old
                elif key == "_atlas_ed_plot_right_up":
                    self._atlas_ed_plot_right_up = evt.old
                elif key == "_atlas_ed_plot_right_down":
                    self._atlas_ed_plot_right_down = evt.old
        self._atlas_ed_plot_ignore = False
    
    @observe("_btn_verifier")
    def _launch_verifier_editor(self, evt):
        if self.verifier_ed:
            # disconnect existing editor since launched outside of refresh btn
            self.verifier_ed.on_close()
        
        verifier_ed = verifier_editor.VerifierEditor(
            config.img5d, self.blobs, "Verifier", self._roi_ed_fig,
            # necessary for immediate table refresh rather than after scroll
            self.update_segment)
        verifier_ed.additive_blend = self._merge_chls
        verifier_ed.show_fig()
        self.verifier_ed = verifier_ed
    
    @staticmethod
    def _get_save_path(default_path: str) -> str:
        """Get a save path from the user through a file dialog.
        
        Args:
            default_path: Default path to display in the dialog.

        Returns:
            Chosen path.
        
        Raises:
            FileNotFoundError: User canceled file selection.

        """
        # open a PyFace file dialog in save mode
        save_dialog = FileDialog(action="save as", default_path=default_path)
        if save_dialog.open() == OK:
            # get user selected path and update preferences
            path = save_dialog.path
            config.prefs.fig_save_dir = os.path.dirname(path)
            return path
        else:
            # user canceled file selection
            raise FileNotFoundError("User canceled file selection")
    
    @on_trait_change("_btn_save_fig")
    def _save_fig(self):
        """Save the figure in the currently selected viewer."""
        path = None
        try:
            # get the figure save directory from preferences
            save_dir = config.prefs.fig_save_dir
            
            if self.selected_viewer_tab is vis_handler.ViewerTabs.ROI_ED:
                if self.roi_ed is not None:
                    # save screenshot of current ROI Editor
                    path = os.path.join(save_dir, self.roi_ed.get_save_path())
                    path = self._get_save_path(path)
                    self.roi_ed.save_fig(path)
            
            elif self.selected_viewer_tab is vis_handler.ViewerTabs.ATLAS_ED:
                if self.atlas_eds:
                    # save screenshot of first Atlas Editor
                    # TODO: find active editor
                    path = os.path.join(
                        save_dir, self.atlas_eds[0].get_save_path())
                    path = self._get_save_path(path)
                    self.atlas_eds[0].save_fig(path)
            
            elif self.selected_viewer_tab is vis_handler.ViewerTabs.MAYAVI:
                if self.scene is not None and config.filename:
                    # save 3D image with extension in config
                    screenshot = self.scene.mlab.screenshot(
                        mode="rgba", antialiased=True)
                    ext = (config.savefig if config.savefig else
                           config.DEFAULT_SAVEFIG)
                    path = "{}.{}".format(naming.get_roi_path(
                        config.filename, self._curr_offset(),
                        self.roi_array[0].astype(int)), ext)
                    path = self._get_save_path(os.path.join(save_dir, path))
                    plot_2d.plot_image(screenshot, path)
            
            if not path:
                # notify that no figure is active to save
                self._roi_feedback = "Please open a figure to save"
        
        except FileNotFoundError as e:
            # user canceled path selection
            print(e)
    
    @on_trait_change('rois_check_list')
    def load_roi(self):
        """Load an ROI from database, including all blobs."""
        print("got {}".format(self.rois_check_list))
        if self.rois_check_list not in ("", self._ROI_DEFAULT):
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
            roi_id = roi["id"]
            blobs = config.db.select_blobs_by_roi(roi_id)[0]
            if len(blobs) > 0:
                # change to single-channel if all blobs are from same channel
                chls = np.unique(detector.Blobs.get_blobs_channel(blobs))
                if len(chls) == 1:
                    self._channel = [str(int(chls[0]))]
            
            # get matches between blobs, such as verifications
            blob_matches = config.db.select_blob_matches(roi_id)
            blob_matches.update_blobs(
                detector.Blobs.shift_blob_rel_coords,
                [n * -1 for n in config.roi_offset[::-1]])
            
            # display blobs
            self.detect_blobs(segs=blobs, blob_matches=blob_matches)
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
            if self._rois_selections.selections:
                self.rois_check_list = self._rois_selections.selections[0]
            self._reset_segments()
        
        # immediately update in Atlas Editors, where None uses the default max
        # intensity projection settings and 0's turns it off
        mip_shape = (None if self._DEFAULTS_2D[4] in self._check_list_2d
                     else (0, 0, 0))
        self._update_mip(mip_shape)
    
    @on_trait_change("_region_name")
    def _region_name_changed(self):
        """Handle changes to the region name combo box."""
        ids = str(self._region_id_map[self._region_name])
        if RegionOptions.APPEND.value in self._region_options:
            # append the region ID
            ids = f"{self._region_id},{ids}"
        # remove preceding commas
        self._region_id = ids.lstrip(",")
    
    def _show_all_labels(
            self, eds: Sequence[Union[
                "plot_support.ImageSyncMixin",
                Sequence["plot_support.ImageSyncMixin"]]], show=True):
        """Show all atlas labels.
        
        Args:
            eds: Sequence of individual or sequences of ROI and Atlas Editors.
                None values will be ignored.
            show: True (default) to show the labels.

        """
        # skip Nones and flatten list
        eds_all = []
        for ed in eds:
            if ed:
                if libmag.is_seq(ed):
                    eds_all.extend(ed)
                else:
                    eds_all.append(ed)
        
        if eds_all:
            # show in thread
            self._annotate_labels_thread = atlas_threads.AnnotateLabels(
                eds_all, show, lambda: None, self._update_prog)
            self._annotate_labels_thread.start()
    
    @on_trait_change("_region_options")
    def _region_options_changed(self):
        """Handle changes to the region options check boxes."""
        # get dict of whether each option changed
        options, changed = self.get_changed_options(
            self._region_options, self._region_options_prev, RegionOptions)
        
        if changed[RegionOptions.APPEND] and not options[RegionOptions.APPEND]:
            # trigger region ID change with current ID when unchecking append
            region_id = self._region_id
            self._region_id = ""
            self._region_id = region_id
        
        if changed[RegionOptions.SHOW_ALL]:
            # toggle showing all label annotations
            option = options[RegionOptions.SHOW_ALL]
            self._show_all_labels((self.roi_ed, self.atlas_eds), option)
        
        # store current options
        self._region_options_prev = options
    
    @on_trait_change("_region_id")
    def _region_id_changed(self):
        """Center the viewer on the region specified in the corresponding 
        text box.
        
        :const:``_PREFIX_BOTH_SIDES`` can be given to specify IDs with 
        both positive and negative values. All other non-integers will 
        be ignored.
        """
        print("region ID: {}".format(self._region_id))
        if RegionOptions.APPEND.value in self._region_options:
            self._roi_feedback = (
                "Add more regions, then uncheck \"Append\" to view them")
            return
        
        if config.labels_img is None:
            self._roi_feedback = "No labels image loaded to find region"
            return

        if config.labels_ref is None or config.labels_ref.ref_lookup is None:
            self._roi_feedback = "No labels reference loaded to find region"
            return

        # user-given region can be a comma-delimited list of region IDs
        # in the labels reference dict
        region_id_split = self._region_id.split(",")
        region_ids = []
        both_sides = []
        for region_id in region_id_split:
            # get IDs from all sub-regions contained within the given region
            region_id = region_id.strip()
            # specify both sides to get corresponding pos and neg IDs
            both = RegionOptions.BOTH_SIDES.value in self._region_options
            if region_id.startswith(self._PREFIX_BOTH_SIDES):
                both = True
                region_id = region_id[len(self._PREFIX_BOTH_SIDES):]
            both_sides.append(both)
            try:
                region_id = int(region_id)
            except ValueError:
                # return if cannot convert to an integer
                self._roi_feedback = (
                    "Region ID must be an integer, or preceded by \"+/-n\" "
                    "to include labels from both sides"
                )
                return
            region_ids.append(region_id)
        incl_chil = RegionOptions.INCL_CHILDREN.value in self._region_options
        centroid, self._img_region, region_ids = ontology.get_region_middle(
            config.labels_ref.ref_lookup, region_ids, config.labels_img,
            config.labels_scaling, both_sides=both_sides,
            incl_children=incl_chil)
        if centroid is None:
            self._roi_feedback = (
                "Could not find the region corresponding to ID {}"
                .format(self._region_id))
            return
        meas, vol = cv_nd.meas_region(
            self._img_region, config.resolutions[0])[:2]

        if Vis3dOptions.RAW.value in self._check_list_3d:
            # in "raw" mode, simply center the current ROI on the label
            # centroid, which may lie within a sub-label
            curr_roi_size = self.roi_array[0].astype(int)
            corner = np.subtract(
                centroid, 
                np.around(np.divide(curr_roi_size[::-1], 2)).astype(int))
            self.z_offset, self.y_offset, self.x_offset = corner
            self._check_roi_position()
            self.show_3d()
        else:
            # in non-"raw" mode, show the full label including sub-labels 
            # without non-label areas; TODO: consider making default or 
            # only option
            self.show_label_3d(region_ids)
        # sync with atlas editor to point at center of region
        self.sync_atlas_eds_coords(centroid)
        self._roi_feedback = (
            "Found region ID {} of size x={}, y={}, z={} \u00b5m, "
            "volume {} \u00b5m^3".format(self._region_id, *meas[::-1], vol))
    
    def _curr_offset(self, center=None):
        """Get ROI offset based on the slider controls.
        
        Args:
            center (bool): True if the slider controls point to the center
                of the ROI; False if they refer to the upper left corner.
                Defaults to None, in which case the center value is
                determined base on the :attr:`_roi_center` control.

        Returns:
            list[int]: Offset in ``x,y,z``.

        """
        # TODO: migrate to z,y,x
        offset = self.x_offset, self.y_offset, self.z_offset
        if center or (center is None and self._roi_center):
            offset = plot_3d.roi_center_to_offset(
                offset, self.roi_array[0].astype(int))
        return offset
    
    def set_offset(self, offset, center=None):
        """Set the offset sliders.
        
        Args:
            offset (List[int]): Offset in ``z,y,x``.
            center (bool): True if ``offset`` points to the center
                of the ROI; False if it refers to the upper left corner.
                Defaults to None, in which case the center value is
                determined base on the :attr:`_roi_center` control.

        """
        if center or (center is None and self._roi_center):
            # convert offset to position it at the center of the ROI
            offset = plot_3d.roi_center_to_offset(
                offset, self.roi_array[0].astype(int)[::-1], reverse=True)
        
        # update offset sliders without triggering additional callbacks
        self._ignore_roi_offset_change = True
        self.z_offset, self.y_offset, self.x_offset = offset
        self._ignore_roi_offset_change = False
        
        # update intensity limits, etc for the current image
        self.update_imgadj_for_img()
    
    def get_roi_size(self):
        """Get the current ROI size.
        
        Returns:
            list(int): ROI size in ``z,y,x``.

        """
        return self.roi_array[0].astype(int)[::-1]
    
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
        return detector.calc_overlap()[::-1]
    
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
        """Get index of blob in blobs table."""
        # must take from vis rather than saved copy in case user
        # manually updates the table
        # TODO: consider using ID instead of checking whole blob
        segi = np.nonzero((self.segments == segment).all(axis=1))
        if len(segi) > 0 and len(segi[0]) > 0:
            return segi[0][0]
        return -1
    
    def _flag_seg_for_deletion(self, seg):
        seg[3] = -1 * abs(seg[3])
        detector.Blobs.set_blob_confirmed(seg, -1)
    
    def update_segment(
            self, segment_new: np.ndarray,
            segment_old: Optional[np.ndarray] = None, remove: bool = False
    ) -> np.ndarray:
        """Update segments/blobs list with a new or updated segment.
        
        The blobs table will immediately reflect changes to :attr:`segments`.
        This handler further updates blobs with any other required changes
        and scroll the table to the changed blob.
        
        Args:
            segment_new: New or updated segment. Assumed to have absolute
                coordinates. The segment will be added if ``segment_old``
                is not given, or updated if provided.
            segment_old: Previous version of the segment, which will be
                replaced by ``segment_new`` if found. The absolute coordinates
                of ``segment_new`` will also be updated based on the relative
                coordinates' difference between ``segment_new`` and
                ``segments_old`` as a convenience. Defaults to None. Can be
                the same as ``segement_new`` to simply trigger a table refresh
                and scrolling.
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
        if remove:
            # remove segments, changing radius and confirmation values to
            # flag for deletion from database while saving the ROI
            segi = self._get_vis_segments_index(seg)
            seg = self.segments[segi]
            self._flag_seg_for_deletion(seg)
            self.segs_selected.append(segi)
        elif segment_old is not None:
            # new blob's abs coords are not shifted, so shift new blob's abs
            # coordinates by relative coords' diff between old and new blobs
            # TODO: consider requiring new seg to already have abs coord updated
            self._segs_moved.append(segment_old)
            diff = np.subtract(seg[:3], segment_old[:3])
            detector.Blobs.shift_blob_abs_coords(seg, diff)
            segi = self._get_vis_segments_index(segment_old)
            if segi == -1:
                # try to find old blob from deleted blobs
                self._flag_seg_for_deletion(segment_old)
                segi = self._get_vis_segments_index(segment_old)
            if segi != -1:
                # replace corresponding blob entry in table
                self.segments[segi] = seg
                _logger.debug("Updated segment: %s", seg)
            self.segs_selected.append(segi)
        else:
            # add a new segment to the visualizer table
            segs = [seg]  # for concatenation
            if self.segments is None or len(self.segments) == 0:
                # copy since the object may be changed elsewhere; cast to
                # float64 since original type causes an incorrect database
                # insertion for some reason
                self.segments = np.copy(segs).astype(np.float64)
            else:
                self.segments = np.concatenate((self.segments, segs))
            self.segs_selected.append(len(self.segments) - 1)
            _logger.debug("Added segment to table: %s", seg)
        
        # scroll to first selected row
        self._segs_row_scroll = min(self.segs_selected)
        
        # trigger refresh since the table may otherwise not refresh until scroll
        self._segs_refresh_evt = True
        
        if self.roi_ed:
            # flag ROI Editor as edited
            self.roi_ed.edited = True
        
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
    
    def _get_profile_category(self) -> Optional[profiles.SettingsDict]:
        """Get the profile instance eassociated with the selected category.
        
        Returns:
            Profile instance, which can be None.

        """
        cat = self._profiles_cats[0]
        if cat == ProfileCats.ROI.value:
            prof = config.roi_profile
        elif cat == ProfileCats.ATLAS.value:
            prof = config.atlas_profile
        else:
            prof = config.grid_search_profile
        return prof
    
    @on_trait_change("_profiles_cats")
    def _update_profiles_names(self):
        """Update the profile names dropdown box based on the selected category.
        """
        # get base profile matching category
        prof = self._get_profile_category()

        if prof:
            # add profiles and matching configuration files
            prof_names = [*prof.profiles.keys(), *prof.get_files()]
        else:
            # clear profile names
            prof_names = []
        # update dropdown box selections and choose the first item, required
        # even though the item will appear to be selected by default
        self._profiles_names = TraitsList()
        self._profiles_names.selections = prof_names
        if prof_names:
            self._profiles_name = prof_names[0]
        self._show_combined_profile()
    
    @on_trait_change("_profiles_name")
    def _select_profile(self):
        """Handle profile selections."""
        prof_cat = self._get_profile_category()
        if prof_cat:
            # show preview of selected profile
            self._profiles_preview = pprint.pformat(
                prof_cat.get_profile(self._profiles_name))
        else:
            self._profiles_preview = ""

    def _show_combined_profile(self):
        """Show the combined profile dictionary."""
        prof_cat = self._get_profile_category()
        self._profiles_combined = pprint.pformat(prof_cat) if prof_cat else ""

    @on_trait_change("_profiles_add_btn")
    def _add_profile(self):
        """Add the chosen profile to the profiles table."""
        for chl in self._profiles_chls:
            # get profile name for selected channel and add to table,
            # deferring profile load until gathering all profiles
            prof = [self._profiles_cats[0], self._profiles_name, chl]
            _logger.debug("Profile to add: %s", prof)
            self._ignore_profiles_update = True
            self._profiles.append(prof)
        
        # load profiles based on final selections
        self._refresh_profiles()
    
    @on_trait_change("_profiles[]")
    def _refresh_profiles(self):
        """Refresh the loaded profiles based on the profiles table."""
        if self._ignore_profiles_update or not self._profiles:
            # no profiles in the table to load or set to ignore
            self._ignore_profiles_update = False
            return
        
        # convert to Numpy array for fancy indexing
        _logger.debug("Profiles from table:\n%s", self._profiles)
        profs = np.array(self._profiles)

        # load ROI profiles to the given channel
        roi_profs = []
        profs = profs[profs[:, 0] == ProfileCats.ROI.value]
        if len(profs) > 0:
            print(profs, max(profs[:, 2].astype(int)))
            for i in range(max(profs[:, 2].astype(int)) + 1):
                roi_profs.append(",".join(profs[profs[:, 2] == str(i), 1]))

        # load atlas and grid search profiles regardless of channel
        atlas_profs = ",".join(
            profs[profs[:, 0] == ProfileCats.ATLAS.value, 1])
        grid_profs = ",".join(
            profs[profs[:, 0] == ProfileCats.GRID.value, 1])

        # set up all profiles and showed the profile for the selected category
        cli.setup_roi_profiles(roi_profs)
        cli.setup_atlas_profiles(atlas_profs)
        cli.setup_grid_search_profiles(grid_profs)
        self._show_combined_profile()

    @on_trait_change("_profiles_load_btn")
    def _load_profiles(self):
        """Reload available profiles."""
        self._update_profiles_names()

    def _init_profiles(self):
        """Initialize the profiles table based on the currently loaded profiles.
        """
        def add_profs(cat, prof, chl=0):
            # add rows for the given profile category
            if not prof:
                return
            for namei, name in enumerate(
                    prof[prof.NAME_KEY].split(prof.delimiter)):
                if namei == 0:
                    # skip default profile, which is included by default
                    continue
                profs.append([cat, name, chl])

        profs = []
        for i, roi_prof in enumerate(config.roi_profiles):
            add_profs(ProfileCats.ROI.value, roi_prof, i)
        add_profs(ProfileCats.ATLAS.value, config.atlas_profile)
        add_profs(ProfileCats.GRID.value, config.grid_search_profile)
        self._profiles = profs
    
    @on_trait_change("_profiles_reset_prefs_btn")
    def _reset_prefs(self):
        """Handle button to reset preferences."""
        # trigger resetting TraitsUI prefs in handler
        self._profiles_reset_prefs = True
        
        # reset preferences profile
        config.prefs = prefs_prof.PrefsProfile()
    
    @observe("_import_file_btn")
    @observe("_import_dir_btn")
    def _import_file_dir_btn_updated(self, evt):
        """Open a Pyface file/dir dialog to import files."""
        # open a file browser, defaulting to last opened location
        is_dir = evt.name == "_import_dir_btn"
        if is_dir:
            # open a dialog for selecting a directory
            open_dialog = DirectoryDialog(default_path=config.prefs.import_dir)
        else:
            # open a dialog for selecting a file
            open_dialog = FileDialog(
                action="open files", default_path=config.prefs.import_dir)
        
        if open_dialog.open() == OK:
            # save selected path in prefs
            config.prefs.import_dir = open_dialog.path
            if is_dir:
                self._import_browser_paths = None
            else:
                # set _import_browser_paths before triggering Trait when
                # setting import_browser
                self._import_browser_paths = open_dialog.paths
                
                # store file's parent dir
                config.prefs.import_dir = os.path.dirname(
                    config.prefs.import_dir)
            
            # trigger update for user selected path
            self._import_browser = open_dialog.path

    @on_trait_change("_import_browser")
    def _add_import_file(self):
        """Add a file or directory to import and populate the import table
        with all related files.
        """

        def setup_import(md):
            # populate the import metadata and output fields based on
            # extracted values
            res = md[config.MetaKeys.RESOLUTIONS]
            if res is not None:
                self._import_res = [res[::-1]]
            mag = md[config.MetaKeys.MAGNIFICATION]
            if mag is not None:
                self._import_mag = mag
            zoom = md[config.MetaKeys.ZOOM]
            if zoom is not None:
                self._import_zoom = zoom
            shape = md[config.MetaKeys.SHAPE]
            if shape is not None:
                self._import_shape = [shape[::-1]]
            
            dtype_str = md[config.MetaKeys.DTYPE]
            if dtype_str:
                # set data type related dropdowns based on each character of
                # the type-string of the Numpy data type object
                try:
                    dtype_str = np.dtype(dtype_str).str
                    len_dtype_str = len(dtype_str)
                    if len_dtype_str > 0:
                        byte_orders = libmag.get_dict_keys_from_val(
                            self._IMPORT_BYTE_ORDERS, dtype_str[0])
                        if byte_orders:
                            self._import_byte_order = [byte_orders[0]]
                    if len_dtype_str > 1:
                        data_types = libmag.get_dict_keys_from_val(
                            self._IMPORT_DATA_TYPES, dtype_str[1])
                        if data_types:
                            self._import_data_type = [data_types[0]]
                    if len_dtype_str > 2:
                        bits = libmag.get_dict_keys_from_val(
                            self._IMPORT_BITS, dtype_str[2])
                        if bits:
                            self._import_bit = [bits[0]]
                except TypeError:
                    print("Could not find data type for {}".format(dtype_str))
            
            if shape and dtype_str:
                # signal ready import
                self._update_import_feedback(
                    "Ready to import. Please check microscope metadata "
                    "and edit if necessary.")
            else:
                self._update_import_feedback(
                    "No image files were auto-detected but may be in a RAW "
                    "format. Please enter at least image output and data type "
                    "before importing.")
        
        import_path = self._import_browser
        for suffix in (config.SUFFIX_IMAGE5D, config.SUFFIX_META,
                       config.SUFFIX_SUBIMG):
            if import_path.endswith(suffix):
                # file already imported; initiate load and change to ROI panel
                self._update_roi_feedback(
                    f"{import_path} is already imported, loading image")
                # must assign before tab change or else _import_browser is empty
                self._filename = import_path
                self.select_controls_tab = ControlsTabs.ROI.value
                return
        
        # reset import fields
        self._clear_import_files(False)
        chl_paths = None
        base_path = None

        try:
            if os.path.isdir(import_path):
                # gather files within the directory to import
                self._import_mode = ImportModes.DIR
                chl_paths, import_md = importer.setup_import_dir(import_path)
                setup_import(import_md)
                base_path = os.path.join(
                    os.path.dirname(import_path),
                    importer.DEFAULT_IMG_STACK_NAME)
            
            elif import_path:
                # gather files matching the pattern of the selected single
                # path, or give list when multiple paths are selected
                self._import_mode = ImportModes.MULTIPAGE
                import_paths = self._import_browser_paths
                chl_paths, base_path = importer.setup_import_multipage(
                    import_paths if import_paths and len(import_paths) >= 2
                    else import_path)
                
                # extract metadata in separate thread given delay from Java
                # initialization for Bioformats
                self._update_import_feedback(
                    f"Gathering metadata related to {import_path}, please "
                    f"wait...")
                self._import_thread = import_threads.SetupImportThread(
                    chl_paths, setup_import)
                self._import_thread.start()
    
            if chl_paths:
                # populate the import table
                data = []
                for chl, paths in chl_paths.items():
                    for path in paths:
                        data.append([path, chl])
                self._import_paths = data
                self._import_prefix = base_path
        except FileNotFoundError:
            self._update_import_feedback(
                f"File to import does not exist: {import_path}\n"
                f"Please try another file")
    
    @on_trait_change("_import_shape")
    def _validate_import_readiness(self):
        """Activate import button once shape and required data type fields
        have beeen entered.
        
        """
        self._import_btn_enabled = (
                not np.equal(self._import_shape, 0).any()
                and self._import_bit
                and self._import_data_type
                and "" not in (
                    self._IMPORT_BITS[self._import_bit[0]],
                    self._IMPORT_DATA_TYPES[self._import_data_type[0]]))
    
    @on_trait_change("_import_bit")
    def _import_bit_changed(self):
        """Validate import readiness when the data bit field changes."""
        self._validate_import_readiness()
    
    @on_trait_change("_import_data_type")
    def _import_data_type_changed(self):
        """Validate import readiness when the data type field changes."""
        self._validate_import_readiness()
    
    @on_trait_change("_import_btn")
    def _import_files(self):
        """Import files based on paths in the import table.
        """
        
        # repopulate channel paths dict, including any user edits
        chl_paths = OrderedDict()
        for row in self._import_paths:
            chl_paths.setdefault(row[1], []).append(row[0])
        
        if chl_paths:
            # set metadata
            md = {
                # assumes only resolutions for chosen series has been set
                # TODO: add control to select series
                config.MetaKeys.RESOLUTIONS: self._import_res[0].astype(
                    float)[::-1],
                config.MetaKeys.MAGNIFICATION: self._import_mag,
                config.MetaKeys.ZOOM: self._import_zoom,
                config.MetaKeys.DTYPE:  "".join(
                    (self._IMPORT_BYTE_ORDERS[self._import_byte_order[0]],
                     self._IMPORT_DATA_TYPES[self._import_data_type[0]],
                     self._IMPORT_BITS[self._import_bit[0]])),
                config.MetaKeys.SHAPE: self._import_shape[0].astype(int)[::-1],
            }
            
            # initialize import in separate thread with a signal for
            # loading thew newly imported image
            self._import_thread = import_threads.ImportThread(
                self._import_mode, self._import_prefix, chl_paths, md,
                self._update_import_feedback,
                # update image path and trigger loading the image
                lambda: self.update_filename(self._import_prefix))
            self._import_thread.start()
    
    @observe("_import_clear_btn")
    def _clear_import_files_fired(self, evt):
        """Handle clearing import fields."""
        self._clear_import_files()
    
    def _clear_import_files(self, clear_import_browser=True):
        """Reset import setup.
        
        Args:
            clear_import_browser (bool): True to clear the import browser
                path; defaults to True. Should be False when clearing settings
                triggered by setting a new import path.

        """
        if clear_import_browser:
            self._import_browser = ""  # will reset table
        self._import_paths = []
        self._import_mode = None
        self._import_res = np.ones((1, 3))
        self._import_shape = np.zeros((1, 5), dtype=int)
        self._import_bit = [tuple(self._IMPORT_BITS.keys())[0]]
        self._import_data_type = [tuple(self._IMPORT_DATA_TYPES.keys())[0]]
        self._import_byte_order = [tuple(self._IMPORT_BYTE_ORDERS.keys())[0]]
        self._import_btn_enabled = False

    def _update_import_feedback(self, val):
        """Update the import feedback text box.
        
        Args:
            val (str): String to append as a new line.

        """
        self._import_feedback += "{}\n".format(val)

    def _update_roi_feedback(self, val: str, print_out: bool = False):
        """Update the ROI panel feedback text box.

        Args:
            val: String to append as a new line.
            print_out: True to print to log as well; defaults to False.

        """
        if print_out:
            _logger.info(val)
        self._roi_feedback += "{}\n".format(val)
    
    # BRAINGLOBE CONTROLS
    
    def _set_bg_atlases(self, val: Sequence):
        """Set BrainGlobe atlases table data."""
        self._bg_atlases = val
    
    def _set_bg_feedback(self, val: str, append: bool = True):
        """Set BrainGlobe feedback string.
        
        Args:
            val: Feedback string.
            append: True to append the string; otherwise, ``val`` replaces
                all current text. Defaults to True.

        Returns:

        """
        if append:
            # append text
            if self._bg_feedback:
                # add a newline if text is already present
                val = f"\n{val}"
            self._bg_feedback += val
        else:
            # replace text
            self._bg_feedback = val
    
    def _bg_open_handler(self, bg_atlas: "BrainGlobeAtlas"):
        """Handler to open a BrainGlobe atlas.
        
        Args:
            bg_atlas: Atlas to open.

        """
        # update displayed filename
        config.filename = str(bg_atlas.root_dir)
        self.update_filename(config.filename, True)
        
        # display the atlas images
        np_io.setup_images(
            config.filename, allow_import=False, bg_atlas=bg_atlas)
        self._setup_for_image()
        self.redraw_selected_viewer()
        self.update_imgadj_for_img()
    
    def _setup_brain_globe(self):
        """Set up the BrainGlobe controller."""
        panel = bg_controller.BrainGlobeCtrl(
            self._set_bg_atlases, self._set_bg_feedback, self._update_prog,
            self._bg_open_handler)
        return panel
        
    @on_trait_change("_bg_access_btn")
    def _open_brain_globe_atlas(self):
        """Open a BrainGlobe atlas."""
        if self._bg_atlases_sel:
            self._brain_globe_panel.open_atlas(self._bg_atlases_sel[0])
    
    @on_trait_change("_bg_remove_btn")
    def _remove_brain_globe_atlas(self):
        """Remove a local copy of a BrainGlobe atlas."""
        if self._bg_atlases_sel:
            self._brain_globe_panel.remove_atlas(self._bg_atlases_sel[0])


if __name__ == "__main__":
    print("Starting the MagellanMapper graphical interface...")
    cli.main()
    main()
