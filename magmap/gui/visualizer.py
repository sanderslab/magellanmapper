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

from collections import OrderedDict
from enum import Enum, auto
import os

import matplotlib
matplotlib.use("Qt5Agg")  # explicitly use PyQt5 for custom GUI events
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import figure
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

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
from pyface.api import FileDialog, OK
from traits.api import (HasTraits, Instance, on_trait_change, Button, Float,
                        Int, List, Array, Str, Bool, Any,
                        push_exception_handler, Property, File)
from traitsui.api import (View, Item, HGroup, VGroup, Tabbed, Handler,
                          RangeEditor, HSplit, TabularEditor, CheckListEditor, 
                          FileEditor, TextEditor, ArrayEditor, BooleanEditor)
from traitsui.basic_editor_factory import BasicEditorFactory
from traitsui.qt4.editor import Editor
from traitsui.tabular_adapter import TabularAdapter
from tvtk.pyface.scene_editor import SceneEditor
from tvtk.pyface.scene_model import SceneModelError
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
import vtk

from magmap.atlas import ontology
from magmap.cv import chunking, colocalizer, cv_nd, detector, segmenter,\
    verifier
from magmap.gui import atlas_editor, import_threads, roi_editor, vis_3d
from magmap.io import cli, importer, libmag, naming, np_io, sitk_io, sqlite
from magmap.plot import colormaps, plot_2d, plot_3d
from magmap.settings import config


# default ROI name
_ROI_DEFAULT = "None selected"


def main():
    """Starts the visualization GUI.
    
    Also sets up exception handling.
    """
    # show complete stacktraces for debugging
    push_exception_handler(reraise_exceptions=True)

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


class VisHandler(Handler):
    """Custom handler for Visualization object events."""

    def init(self, info):
        """Handle events after controls have been generated but prior to
        their display.

        Args:
            info (UIInfo): TraitsUI UI info.

        Returns:
            bool: True.

        """
        def handle_tab_changed(i):
            # set the enum for the currently selected tab and initialize
            # viewers if necessary
            tab = ViewerTabs(i + 1)  # enums auto-index starting from 1
            info.object.selected_viewer_tab = tab
            print("Changed to tab", i, tab)
            if (info.object.stale_viewers[tab] is StaleFlags.IMAGE
                    or tab is ViewerTabs.ROI_ED and not info.object.roi_ed
                    or tab is ViewerTabs.ATLAS_ED and not info.object.atlas_eds
                    or tab is ViewerTabs.MAYAVI
                    and not info.object.scene_3d_shown):
                # redraw if new image has not been drawn for tab, or the
                # corresponding viewer has not been shown before
                info.object.redraw_selected_viewer(clear=False)
                if tab is ViewerTabs.MAYAVI:
                    # initialize the camera orientation
                    info.object.orient_camera()
            elif tab is ViewerTabs.ATLAS_ED:
                # synchronize Atlas Editors to ROI offset if option selected
                info.object.sync_atlas_eds_coords(check_option=True)
            info.object.update_imgadj_for_img()

        # change Trait to flag completion of controls creation
        info.object.controls_created = True

        # add a change listener for the viewer tab widget, which is the
        # first found widget
        tab_widgets = info.ui.control.findChildren(QtWidgets.QTabWidget)
        tab_widgets[0].currentChanged.connect(handle_tab_changed)
        return True

    def closed(self, info, is_ok):
        """Shuts down the application when the GUI is closed."""
        cli.shutdown()

    def object_mpl_fig_active_changed(self, info):
        """Change keyboard focus depending on the shown tab.

        TraitsUI does not hand Matplotlib figures keyboard focus except
        when the ``Item`` initially requests focus in ``has_focus``, and
        even then the figure cannot regain focus once lost. As a workaround,
        store the active figure in a Trait and request focus on the
        figure from the underlying Qt widget.

        Args:
            info (UIInfo): TraitsUI UI info.

        """
        if info.object.mpl_fig_active is None:
            # into.object is the Visualization object
            return

        # get all Matplotlib figure canvases displayed via TraitsUI as
        # Qt widgets; the control is a _StickyDialog that extends QDialog
        mpl_figs = info.ui.control.findChildren(
            matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg)
        for fig in mpl_figs:
            if fig.figure == info.object.mpl_fig_active:
                # shift keyboard focus to canvas matching the currently
                # shown Matplotlib figure
                fig.setFocus()
    
    @staticmethod
    def scroll_to_bottom(eds, ed_name):
        """Scroll to the bottom of the given Qt editor.

        Args:
            eds (List): Sequence of Qt editors.
            ed_name (str): Name of editor to scroll.

        """
        for ed in eds:
            if ed.name == ed_name:
                # scroll to end of text display
                ed.control.moveCursor(QtGui.QTextCursor.End)
    
    def object__roi_feedback_changed(self, info):
        """Scroll to the bottom of the ROI feedback text display
        when the value is changed.

        Args:
            info (UIInfo): TraitsUI UI info.

        """
        self.scroll_to_bottom(info.ui._editors, "_roi_feedback")

    def object__import_feedback_changed(self, info):
        """Scroll to the bottom of the import feedback text display
        when the value is changed.
        
        Args:
            info (UIInfo): TraitsUI UI info.

        """
        self.scroll_to_bottom(info.ui._editors, "_import_feedback")
    
    def object_select_controls_tab_changed(self, info):
        """Select the given tab specified by
        :attr:`Visualization.select_controls_tab`.
        
        Args:
            info (UIInfo): TraitsUI UI info.

        """
        # the tab widget is the second found QTabWidget; subtract one since
        # Enums auto-increment from 1
        tab_widgets = info.ui.control.findChildren(QtWidgets.QTabWidget)
        tab_widgets[1].setCurrentIndex(info.object.select_controls_tab - 1)


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


class ProfileCats(Enum):
    """Profile categories enumeration."""
    ROI = "ROI"
    ATLAS = "Atlas"
    GRID = "Grid Search"


class TraitsList(HasTraits):
    """Generic Traits-enabled list."""
    selections = List([""])


class ProfilesArrayAdapter(TabularAdapter):
    """Profiles TraitsUI table adapter."""
    columns = [("Type", 0), ("Name", 1), ("Channel", 2)]
    widths = {0: 0.2, 1: 0.7, 2: 0.1}

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
    INCL_CHILDREN = "Include children"


class AtlasEditorOptions(Enum):
    """Enumerations for Atlas Editor options."""
    SHOW_LABELS = "Labels"
    SYNC_ROI = "Sync to ROI"


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


class ViewerTabs(Enum):
    """Enumerations for viewer tabs."""
    ROI_ED = auto()
    ATLAS_ED = auto()
    MAYAVI = auto()


class StaleFlags(Enum):
    """Enumerations for stale viewer states."""
    IMAGE = auto()  # loaded new image
    ROI = auto()  # changed ROI offset or size


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
        stale_viewers (dict[:class:`ViewerTabs`, :class:`StaleFlags`]):
            Dictionary of viewer tab Enums to stale flag Enums, where the
            flag indicates the update that the viewer would require.
            Defaults to all viewers set to :class:`StaleFlags.IMAGE`.
    
    """
    # File selection

    _filename = File  # file browser
    _ignore_filename = False  # ignore file update trigger
    _channel_names = Instance(TraitsList)
    _channel = List  # selected channels, 0-based
    
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
    
    _labels_ref_path = File  # labels ontology reference path
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
    roi = None  # combine with roi_array?
    _rois_selections = Instance(ListSelections)
    rois_check_list = Str
    _rois_dict = None
    _rois = None
    _roi_feedback = Str()
    
    # Detect panel
    
    btn_detect = Button("Detect")
    btn_save_segments = Button("Save Blobs")
    _segs_visible = List  # blob visibility options
    _colocalize = List  # blob co-localization options
    _blob_color_style = List  # blob coloring
    _segments = Array  # table of blobs
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

    # Profiles panel

    _profiles_cats = List
    _profiles_names = Instance(TraitsList)
    _profiles_name = Str
    _profiles_chls = List
    _profiles_table = TabularEditor(
        adapter=ProfilesArrayAdapter(), editable=True, auto_resize_rows=True,
        stretch_last_section=False)
    _profiles = List  # profiles table list
    _profiles_add_btn = Button("Add profile")
    _profiles_load_btn = Button("Load profiles")

    # Image adjustment panel

    _imgadj_names = Instance(TraitsList)
    _imgadj_name = Str
    _imgadj_chls_names = Instance(TraitsList)
    _imgadj_chls = Str
    _imgadj_min = Float
    _imgadj_min_low = Float
    _imgadj_min_high = Float
    _imgadj_min_auto = Bool
    _imgadj_max = Float
    _imgadj_max_low = Float
    _imgadj_max_high = Float
    _imgadj_max_auto = Bool
    _imgadj_brightness = Float
    _imgadj_brightness_low = Float
    _imgadj_brightness_high = Float
    _imgadj_contrast = Float
    _imgadj_alpha = Float

    # Image import panel

    _import_browser = File
    _import_table = TabularEditor(
        adapter=ImportFilesArrayAdapter(), editable=True, auto_resize_rows=True,
        stretch_last_section=False)
    _import_paths = List  # import paths table list
    _import_btn = Button("Import files")
    _import_btn_enabled = Bool
    _import_clear_btn = Button("Clear import")
    _import_res = Array(np.float, shape=(1, 3))
    _import_mag = Float(1.0)
    _import_zoom = Float(1.0)
    _import_shape = Array(np.int, shape=(1, 5), editor=ArrayEditor(
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

    # Image viewers

    atlas_eds = []  # open atlas editors
    flipz = True  # True to invert 3D vis along z-axis
    controls_created = Bool(False)
    mpl_fig_active = Any
    scene = Instance(MlabSceneModel, ())
    scene_3d_shown = False  # 3D Mayavi display shown
    selected_viewer_tab = ViewerTabs.ROI_ED
    select_controls_tab = Int(-1)
    
    # Viewer options
    
    _check_list_3d = List(
        tooltip="Raw: show raw intensity image; limit region to ROI\n"
                "Surface: render as surface; uncheck to render as points\n"
                "Clear: clear scene before drawing new objects\n"
                "Panes: show back and top panes\n"
                "Shadows: show blob shadows as circles")
    _check_list_2d = List(
        tooltip="Filtered: show filtered image after detection\n"
                "Border: margin around ROIs\n"
                "Seg: segment blobs\n"
                "Grid: overlay a grid\n"
                "MIP: maximum intensity projection")
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
                "Sync to ROI: move to the ROI offset whenever it changes")
    # select to zoom Atlas Ed into the ROI, with crosshairs at center of ROI
    # if ROI Center box also selected; unselect to zoom out to whole image
    _atlas_ed_zoom = Bool(
        tooltip="Select: zoom into ROI; select Center to center ROI on "
                "crosshairs\n"
                "Unselect: zoom out to full image")
    
    # atlas labels
    _atlas_label = None
    _structure_scale = Int  # ontology structure levels
    _structure_scale_low = -1
    _structure_scale_high = 20
    _region_id = Str
    _region_options = List
    
    _mlab_title = None
    _circles_opened_type = None  # enum of circle style for opened 2D plots
    _opened_window_style = None  # 2D plots window style curr open
    _img_region = None
    _PREFIX_BOTH_SIDES = "+/-"
    _camera_pos = None
    _roi_ed_fig = Instance(figure.Figure, ())
    _atlas_ed_fig = Instance(figure.Figure, ())
    _status_bar_msg = Str()  # text for status bar

    # ROI selector panel
    panel_roi_selector = VGroup(
        VGroup(
            HGroup(
                Item("_filename", label="File", style="simple",
                     editor=FileEditor(entries=10, allow_dir=False)),
            ),
            Item("_channel", label="Channels", style="custom",
                 editor=CheckListEditor(
                     name="object._channel_names.selections", cols=8)),
            label="Image path",
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
                Item("_labels_ref_path", label="Reference", style="simple",
                     editor=FileEditor(entries=10, allow_dir=False)),
            ),
            label="Registered Images",
        ),
        VGroup(
            Item("rois_check_list", label="ROIs",
                 editor=CheckListEditor(
                     name="object._rois_selections.selections")),
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
            label="Region of Interest",
        ),
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
            HGroup(
                Item("_atlas_ed_options", style="custom", label="Atlas Editor",
                     editor=CheckListEditor(
                         values=[e.value for e in AtlasEditorOptions],
                         cols=len(AtlasEditorOptions),
                         format_func=lambda x: x)),
                Item("_atlas_ed_zoom", label="Zoom to ROI",
                     editor=BooleanEditor()),
            ),
            Item("_check_list_3d", style="custom", label="3D Viewer",
                 editor=CheckListEditor(
                     values=[e.value for e in Vis3dOptions],
                     cols=len(Vis3dOptions), format_func=lambda x: x)),
            label="Viewer Options",
        ),
        HGroup(
            Item("_structure_scale", label="Atlas ontology level",
                 editor=RangeEditor(
                     low_name="_structure_scale_low",
                     high_name="_structure_scale_high",
                     mode="slider")),
        ),
        HGroup(
            Item("_region_id", label="Region",
                 editor=TextEditor(
                     auto_set=False, enter_set=True, evaluate=str)),
            Item("_region_options", style="custom", show_label=False,
                 editor=CheckListEditor(
                     values=[e.value for e in RegionOptions], cols=2,
                     format_func=lambda x: x)),
        ),
        Item("_roi_feedback", style="custom", show_label=False),
        HGroup(
            Item("btn_redraw", show_label=False),
            Item("_btn_save_fig", show_label=False),
        ),
        label="ROI",
    )
    
    # blob detections panel
    panel_detect = VGroup(
        HGroup(
            Item("btn_detect", show_label=False),
            Item("btn_save_segments", show_label=False),
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
        Item("scale_detections",
             editor=RangeEditor(
                 low_name="_scale_detections_low",
                 high_name="_scale_detections_high",
                 mode="slider")),
        VGroup(
            Item("_segments", editor=segs_table, show_label=False),
            Item("segs_feedback", style="custom", show_label=False),
        ),
        label="Detect",
    )

    # profiles panel
    panel_profiles = VGroup(
        HGroup(
            Item("_profiles_cats", style="simple", label="Profile",
                 editor=CheckListEditor(
                     values=[e.value for e in ProfileCats], cols=1,
                     format_func=lambda x: x)),
            Item("_profiles_name", label="Name",
                 editor=CheckListEditor(
                     name="object._profiles_names.selections",
                     format_func=lambda x: x)),
        ),
        Item("_profiles_chls", label="Channels", style="custom",
             editor=CheckListEditor(
                 name="object._channel_names.selections", cols=8)),
        HGroup(
            Item("_profiles_add_btn", show_label=False),
            Item("_profiles_load_btn", show_label=False),
        ),
        Item("_profiles", editor=_profiles_table, show_label=False),
        label="Profiles",
    )

    # image adjustment panel
    panel_imgadj = VGroup(
        HGroup(
            Item("_imgadj_name", label="Image",
                 editor=CheckListEditor(
                     name="object._imgadj_names.selections")),
            Item("_imgadj_chls", label="Channel",
                 editor=CheckListEditor(
                     name="object._imgadj_chls_names.selections")),
        ),
        HGroup(
            Item("_imgadj_min", label="Minimum", editor=RangeEditor(
                     low_name="_imgadj_min_low", high_name="_imgadj_min_high",
                     mode="slider", format="%.4g")),
            Item("_imgadj_min_auto", label="Auto", editor=BooleanEditor()),
        ),
        HGroup(
            Item("_imgadj_max", label="Maximum", editor=RangeEditor(
                     low_name="_imgadj_max_low", high_name="_imgadj_max_high",
                     mode="slider", format="%.4g")),
            Item("_imgadj_max_auto", label="Auto", editor=BooleanEditor()),
        ),
        Item("_imgadj_brightness", label="Brightness", editor=RangeEditor(
                 low_name="_imgadj_brightness_low",
                 high_name="_imgadj_brightness_high", mode="slider",
                 format="%.4g")),
        Item("_imgadj_contrast", label="Contrast", editor=RangeEditor(
                 low=0.0, high=2.0, mode="slider", format="%.3g")),
        Item("_imgadj_alpha", label="Opacity", editor=RangeEditor(
                 low=0.0, high=1.0, mode="slider", format="%.3g")),
        label="Adjust Image",
    )
    
    # import panel
    panel_import = VGroup(
        VGroup(
            HGroup(
                # prevent label from squeezing the width of rest of controls
                Item("_import_browser", label="Select first file to import",
                     style="simple",
                     editor=FileEditor(entries=10, allow_dir=True)),
            ),
            Item("_import_paths", editor=_import_table, show_label=False),
            label="Import File Selection"
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

    # tabbed panel of options
    panel_options = Tabbed(
        panel_roi_selector,
        panel_detect,
        panel_profiles,
        panel_imgadj,
        panel_import,
    )

    # tabbed panel with ROI Editor, Atlas Editor, and Mayavi scene
    panel_figs = Tabbed(
        Item("_roi_ed_fig", label="ROI Editor", show_label=False,
             editor=MPLFigureEditor(), width=1000, height=600),
        Item("_atlas_ed_fig", label="Atlas Editor", show_label=False,
             editor=MPLFigureEditor()),
        Item("scene", label="3D Viewer", show_label=False,
             editor=SceneEditor(scene_class=MayaviScene)),
    )

    # set up the GUI layout; control the HSplit width ratio using a width
    # for this whole view and a width for an item in panel_figs
    view = View(
        HSplit(
            panel_options,
            panel_figs,
        ),
        width=1500,
        handler=VisHandler(),
        title="MagellanMapper",
        statusbar="_status_bar_msg",
        resizable=True,
    )

    def __init__(self):
        """Initialize GUI."""
        HasTraits.__init__(self)

        # default options setup
        self._set_border(True)
        self._circles_2d = [
            roi_editor.ROIEditor.CircleStyles.CIRCLES.value]
        self._planes_2d = [self._DEFAULTS_PLANES_2D[0]]
        self._styles_2d = [Styles2D.SQUARE.value]
        # self._check_list_2d = [self._DEFAULTS_2D[1]]
        self._check_list_3d = [
            Vis3dOptions.RAW.value, Vis3dOptions.SURFACE.value]
        if (config.roi_profile["vis_3d"].lower()
                == Vis3dOptions.SURFACE.value.lower()):
            # check "surface" if set in profile
            self._check_list_3d.append(Vis3dOptions.SURFACE.value)
        # self._structure_scale = self._structure_scale_high
        self._region_options = [RegionOptions.INCL_CHILDREN.value]
        self.blobs = detector.Blobs()
        self._blob_color_style = [BlobColorStyles.ATLAS_LABELS.value]

        # set up profiles selectors
        self._profiles_cats = [ProfileCats.ROI.value]
        self._update_profiles_names()
        self._init_profiles()

        # set up image import
        self._clear_import_files(False)
        self._import_thread = None  # prevent prematurely destroying threads

        # ROI margin for extracting previously detected blobs
        self._margin = config.plot_labels[config.PlotLabels.MARGIN]
        if self._margin is None:
            self._margin = (5, 5, 3)  # x,y,z
        
        # store ROI offset for currently drawn plot in case user previews a
        # new ROI offset, which shifts the current offset sliders
        self._drawn_offset = self._curr_offset()

        # setup interface for image
        if config.filename:
            # show image filename in file selector without triggering update
            self._ignore_filename = True
            self._filename = config.filename

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
        self.roi_ed = None
        # no constrained layout because of performance impact at least as of
        # Matplotlib 3.2
        self._roi_ed_fig = figure.Figure()
        self._atlas_ed_fig = figure.Figure()
        self._atlas_ed_options = [
            AtlasEditorOptions.SHOW_LABELS.value,
            AtlasEditorOptions.SYNC_ROI.value]
        self._segs_visible = [BlobsVisibilityOptions.VISIBLE.value]
        
        # 3D visualization object
        self._vis3d = None
        
        # set up rest of image adjustment during image setup
        self.stale_viewers = self.reset_stale_viewers()
        self._img3ds = None
        self._imgadj_min_ignore_update = False
        self._imgadj_max_ignore_update = False
        
        # set up rest of registered images during image setup
        self._main_img_names_avail = TraitsList()
        self._main_img_names = TraitsList()
        self._labels_img_names = TraitsList()
        self._ignore_main_img_name_changes = False
        
        # set up image
        self._setup_for_image()

    def _init_channels(self):
        """Initialize channel check boxes for the currently loaded main image.

        """
        # reset channel check boxes, storing selected channels beforehand
        chls_pre = list(self._channel)
        self._channel_names = TraitsList()
        # 1 channel if no separate channel dimension
        num_chls = (1 if config.image5d is None or config.image5d.ndim < 5
                    else config.image5d.shape[4])
        self._channel_names.selections = [str(i) for i in range(num_chls)]
        
        # pre-select channels for both main and profiles selector
        if not chls_pre and config.channel:
            # use config if all chls were unchecked, eg if setting for 1st time
            self._channel = sorted([
                str(c) for i, c in enumerate(config.channel) if i < num_chls])
        else:
            # select all channels
            self._channel = self._channel_names.selections
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

    def _setup_imgadj_channels(self):
        """Set up channels in the image adjustment panel for the given image."""
        img3d = self._img3ds.get(self._imgadj_name) if self._img3ds else None
        if self._imgadj_name == "Main":
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
    
    @on_trait_change("_imgadj_chls")
    def update_imgadj_for_img(self):
        """Update image adjustment controls based on the currently selected
        viewer and channel.

        """
        if not self._imgadj_chls:
            # resetting image adjustment channel names triggers update as
            # empty array
            return
        
        # get the currently selected viewer
        ed = None
        if self.selected_viewer_tab is ViewerTabs.ROI_ED:
            ed = self.roi_ed
        elif self.selected_viewer_tab is ViewerTabs.ATLAS_ED:
            if self.atlas_eds:
                ed = self.atlas_eds[0]
        if ed is None: return

        # get the display settings from the viewer
        imgi = self._imgadj_names.selections.index(self._imgadj_name)
        plot_ax_img = ed.get_img_display_settings(
                imgi, chl=int(self._imgadj_chls))
        if plot_ax_img is None: return
        norm = plot_ax_img.ax_img.norm
        self._imgadj_brightness = plot_ax_img.brightness
        self._imgadj_contrast = plot_ax_img.contrast
        self._imgadj_alpha = plot_ax_img.ax_img.get_alpha()
        self._adapt_imgadj_limits(plot_ax_img)
        
        # populate controls with display settings
        if norm.vmin is None:
            self._imgadj_min_auto = True
        else:
            self._imgadj_min = norm.vmin
            self._imgadj_min_auto = False
        if norm.vmax is None:
            self._imgadj_max_auto = True
        else:
            self._imgadj_max = norm.vmax
            self._imgadj_max_auto = False
    
    def _adapt_imgadj_limits(self, plot_ax_img):
        """Adapt image adjustment slider limits based on values in the
        given plotted image.
        
        Args:
            plot_ax_img (:obj:`magmap.gui.plot_editor.PlotAxImg`): Plotted
                image.

        """
        if plot_ax_img is None: return
        norm = plot_ax_img.ax_img.norm
        inten_lim = (np.amin(plot_ax_img.img), np.amax(plot_ax_img.img))

        if inten_lim[0] < self._imgadj_max_low:
            # ensure that lower limit is beyond current plane's limits;
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
        # set min intensity to current image value
        if plot_ax_img is not None:
            self._imgadj_min_ignore_update = True
            vmin = plot_ax_img.ax_img.norm.vmin
            self._adapt_imgadj_limits(plot_ax_img)
            self._imgadj_min = vmin

    def _set_inten_max_to_curr(self, plot_ax_img):
        # set max intensity to current image value
        if plot_ax_img is not None:
            self._imgadj_max_ignore_update = True
            vmax = plot_ax_img.ax_img.norm.vmax
            self._adapt_imgadj_limits(plot_ax_img)
            self._imgadj_max = vmax

    @on_trait_change("_imgadj_min")
    def _adjust_img_min(self):
        if self._imgadj_min_ignore_update:
            self._imgadj_min_ignore_update = False
            return
        self._imgadj_min_auto = False
        plot_ax_img = self._adjust_displayed_imgs(minimum=self._imgadj_min)
        # intensity max may have been adjusted to remain >= min
        self._set_inten_max_to_curr(plot_ax_img)
    
    @on_trait_change("_imgadj_min_auto")
    def _adjust_img_min_auto(self):
        min_inten = None if self._imgadj_min_auto else self._imgadj_min
        plot_ax_img = self._adjust_displayed_imgs(minimum=min_inten)
        self._set_inten_min_to_curr(plot_ax_img)
        self._set_inten_max_to_curr(plot_ax_img)

    @on_trait_change("_imgadj_max")
    def _adjust_img_max(self):
        if self._imgadj_max_ignore_update:
            self._imgadj_max_ignore_update = False
            return
        self._imgadj_max_auto = False
        plot_ax_img = self._adjust_displayed_imgs(maximum=self._imgadj_max)
        # intensity min may have been adjusted to remain <= max
        self._set_inten_min_to_curr(plot_ax_img)

    @on_trait_change("_imgadj_max_auto")
    def _adjust_img_max_auto(self):
        max_inten = None if self._imgadj_max_auto else self._imgadj_max
        plot_ax_img = self._adjust_displayed_imgs(maximum=max_inten)
        self._set_inten_min_to_curr(plot_ax_img)
        self._set_inten_max_to_curr(plot_ax_img)

    @on_trait_change("_imgadj_brightness")
    def _adjust_img_brightness(self):
        self._adjust_displayed_imgs(brightness=self._imgadj_brightness)

    @on_trait_change("_imgadj_contrast")
    def _adjust_img_contrast(self):
        self._adjust_displayed_imgs(contrast=self._imgadj_contrast)

    @on_trait_change("_imgadj_alpha")
    def _adjust_img_alpha(self):
        self._adjust_displayed_imgs(alpha=self._imgadj_alpha)

    def _adjust_displayed_imgs(self, **kwargs):
        """Adjust image display settings for the currently selected viewer.

        Args:
            **kwargs: Arguments to update the currently selected viewer.
        
        Returns:
            :obj:`magmap.plot_editor.PlotAxImg`: The last updated axes image
            plot, assumed to have the same values as all the other
            updated plots, or None if the selected tab does not have an
            axes image, such as an unloaded tab or 3D visualization tab.

        """
        plot_ax_img = None
        if self.selected_viewer_tab is ViewerTabs.MAYAVI:
            # update 3D visualization Mayavi/VTK settings
            self._vis3d.update_img_display(**kwargs)
        else:
            # update any selected Matplotlib-based viewer settings
            eds = []
            if self.selected_viewer_tab is ViewerTabs.ROI_ED:
                eds.append(self.roi_ed)
            elif self.selected_viewer_tab is ViewerTabs.ATLAS_ED:
                # support multiple Atlas Editors (currently only have one)
                eds.extend(self.atlas_eds)
            plot_ax_img = None
            for ed in eds:
                if not ed: continue
                # update settings for the viewer
                plot_ax_img = ed.update_imgs_display(
                    self._imgadj_names.selections.index(self._imgadj_name),
                    chl=int(self._imgadj_chls), **kwargs)
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
        segs_transposed = []
        segs_to_delete = []
        offset = self._curr_offset()
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
        exp_name = sqlite.get_exp_name(
            config.img5d.path_img if config.img5d else None)
        exp_id = sqlite.select_or_insert_experiment(
            config.db.conn, config.db.cur, exp_name, None)
        roi_id, out = sqlite.select_or_insert_roi(
            config.db.conn, config.db.cur, exp_id, config.series, 
            np.add(self._drawn_offset, self.border).tolist(),
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
        
        # insert blob matches
        if self.blobs.blob_matches is not None:
            self.blobs.blob_matches.update_blobs(
                detector.shift_blob_rel_coords, offset[::-1])
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

    def _btn_save_segments_fired(self):
        """Handler to save blobs to database when triggered by Trait."""
        self.save_segs()

    def _reset_segments(self):
        """Resets the saved segments.
        """
        self.segments = None
        self.blobs = detector.Blobs()
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
                # and have not found a way to turn them off easily, so 
                # consider turning them off by default and deferring to the 
                # GUI to turn them back on
                self.show_orientation_axes(self.flipz)
        # updates the GUI here even though it doesn't elsewhere for some reason
        self.rois_check_list = _ROI_DEFAULT
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
        self.stale_viewers[ViewerTabs.MAYAVI] = None
    
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
        self.stale_viewers[ViewerTabs.MAYAVI] = None
    
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
        if config.image5d is not None:
            # TODO: consider subtracting 1 to avoid max offset being 1 above
            # true max, but currently convenient to display size and checked
            # elsewhere; "high_label" RangeEditor setting also does not
            # appear to be working
            self.z_high, self.y_high, self.x_high = config.image5d.shape[1:4]
            if config.roi_offset is not None:
                # apply user-defined offsets
                self.x_offset = config.roi_offset[0]
                self.y_offset = config.roi_offset[1]
                self.z_offset = config.roi_offset[2]
            self.roi_array = ([[100, 100, 12]] if config.roi_size is None
                              else [config.roi_size])
            
            # find matching registered images to populate dropdowns
            main_img_names_avail = []
            for reg_name in config.RegNames:
                # check for potential registered image files
                reg_path = sitk_io.read_sitk(sitk_io.reg_out_path(
                    config.filename, reg_name.value), dryrun=True)[1]
                if reg_path:
                    # add to list of available suffixes
                    main_img_names_avail.append(reg_name.value)
            # show without extension since exts may differ
            main_img_names_avail = [
                os.path.splitext(s)[0] for s in main_img_names_avail]
            self._labels_img_names.selections = list(main_img_names_avail)
            self._labels_img_names.selections.insert(0, "")
            
            # set any registered names based on loaded images, defaulting to
            # image5d and no labels
            main_suffixes = [self._MAIN_IMG_NAME_DEFAULT]
            labels_suffix = self._labels_img_names.selections[0]
            if config.reg_suffixes:
                # use registered suffixes without ext, using first suffix
                # of each type
                suffixes = config.reg_suffixes[config.RegSuffixes.ATLAS]
                if suffixes:
                    if not libmag.is_seq(suffixes):
                        suffixes = [suffixes]
                    for suffix in suffixes:
                        suffix = os.path.splitext(suffix)[0]
                        if suffix in main_img_names_avail:
                            # move from available to selected suffixes lists
                            main_suffixes.append(suffix)
                            main_img_names_avail.remove(suffix)
                suffix = config.reg_suffixes[config.RegSuffixes.ANNOTATION]
                if suffix:
                    suffix = os.path.splitext(
                        libmag.get_if_within(suffix, 0, ""))[0]
                    if suffix in self._labels_img_names.selections:
                        labels_suffix = suffix
            
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
            
            if config.load_labels:
                # populate labels reference path field
                self._labels_ref_path = config.load_labels

        # set up image adjustment controls
        self._init_imgadj()
        
        # set up selector for loading past saved ROIs
        self._rois_dict = {_ROI_DEFAULT: None}
        img5d = config.img5d
        if config.db is not None and img5d and img5d.path_img is not None:
            self._rois = config.db.get_rois(sqlite.get_exp_name(
                img5d.path_img))
        self._rois_selections = ListSelections()
        if self._rois is not None and len(self._rois) > 0:
            for roi in self._rois:
                self._append_roi(roi, self._rois_dict)
        self._rois_selections.selections = list(self._rois_dict.keys())
        self.rois_check_list = _ROI_DEFAULT
    
    @on_trait_change("_filename")
    def _image_path_updated(self):
        """Update the selected filename and load the corresponding image.
        
        Since an original (eg .czi) image can be processed in so many 
        different ways, assume that the user will select the Numpy image 
        file instead of the raw image. Image settings will be constructed 
        from the Numpy image filename. Processed files (eg ROIs, blobs) 
        will not be loaded for now.
        """
        if self._ignore_filename or not self._filename:
            # ignore if only updating widget value, without triggering load
            self._ignore_filename = False
            return
        
        # load image if possible without allowing import
        filename, offset, size, reg_suffixes = importer.deconstruct_np_filename(
            self._filename)
        if filename is not None:
            config.filename = filename
            print("Changed filename to", config.filename)
            if offset is not None and size is not None:
                config.subimg_offsets = [offset]
                config.subimg_sizes = [size]
                print("Change sub-image offset to {}, size to {}"
                      .format(config.subimg_offsets, config.subimg_sizes))
            # TODO: consider loading processed images, blobs, etc
            if reg_suffixes:
                config.reg_suffixes.update(reg_suffixes)
            np_io.setup_images(
                config.filename, offset=offset, size=size, allow_import=False)
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
        atlas_suffixes = []
        for suffix in self._main_img_names.selections[1:]:
            # add empty extension for each selected atlas suffix
            atlas_suffixes.append("{}.".format(suffix))
        if atlas_suffixes:
            if len(atlas_suffixes) == 1:
                # reduce to str if only one element
                atlas_suffixes = atlas_suffixes[0]
            reg_suffixes[config.RegSuffixes.ATLAS] = atlas_suffixes
        
        if self._labels_img_names.selections.index(self._labels_img_name) != 0:
            # add if not the empty first selection
            reg_suffixes[config.RegSuffixes.ANNOTATION] = "{}.".format(
                self._labels_img_name)
        config.reg_suffixes.update(reg_suffixes)
        
        if self._labels_ref_path:
            # set up labels
            cli.setup_labels([self._labels_ref_path])
        
        # re-setup image
        filename = self._filename
        self._filename = ""
        self._filename = filename
    
    @on_trait_change("_channel")
    def update_channel(self):
        """Update the selected channel, resetting the current state to 
        prevent displaying the old channel.
        """
        if not self._channel:
            # resetting channel names triggers channel update as empty array
            return
        config.channel = sorted([int(n) for n in self._channel])
        self._setup_imgadj_channels()
        self.rois_check_list = _ROI_DEFAULT
        self._reset_segments()
        print("Changed channel to {}".format(config.channel))
    
    def reset_stale_viewers(self, val=StaleFlags.IMAGE):
        """Reset the stale viewer flags for all viewers.
        
        Args:
            val (:class:`StaleFlags`): Enumeration to set for all viewers.

        """
        self.stale_viewers = dict.fromkeys(ViewerTabs, val)
        return self.stale_viewers
    
    @on_trait_change("x_offset,y_offset,z_offset")
    def _update_roi_offset(self):
        """Respond to ROI offset slider changes.
        
        Sets all stale viewer flags to the ROI flag.
        
        """
        print("x: {}, y: {}, z: {}"
              .format(self.x_offset, self.y_offset, self.z_offset))
        self.reset_stale_viewers(StaleFlags.ROI)
        if self.selected_viewer_tab is ViewerTabs.ATLAS_ED:
            # immediately move to new offset if sync selected
            self.sync_atlas_eds_coords(check_option=True)

    @on_trait_change("roi_array")
    def _update_roi_array(self):
        """Respond to ROI size array changes."""
        # flag ROI changed for viewers
        self.reset_stale_viewers(StaleFlags.ROI)
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

    @on_trait_change("btn_redraw")
    def _redraw_fired(self):
        """Respond to redraw button presses."""
        self.redraw_selected_viewer()
    
    def redraw_selected_viewer(self, clear=True):
        """Redraw the selected viewer.
        
        Args:
            clear (bool): True to clear the ROI and blobs; defaults to True.
        
        """
        # reload profiles if any profile files have changed and reset ROI
        cli.update_profiles()
        self._drawn_offset = self._curr_offset()
        if clear:
            self.roi = None
            self._reset_segments()

        # redraw the currently selected viewer tab
        if self.selected_viewer_tab is ViewerTabs.ROI_ED:
            self._launch_roi_editor()
        elif self.selected_viewer_tab is ViewerTabs.ATLAS_ED:
            if self.atlas_eds:
                # TODO: re-support multiple Atlas Editor windows
                self.atlas_eds = []
            self.launch_atlas_editor()
        elif self.selected_viewer_tab is ViewerTabs.MAYAVI:
            self.show_3d()
            self._post_3d_display()
    
    @on_trait_change("scene.activated")
    def orient_camera(self):
        """Provide a default camera orientation with orientation axes.

        """
        view = self.scene.mlab.view(*self.scene.mlab.view()[:3], "auto")
        roll = self.scene.mlab.roll(-175)
        if self.scene_3d_shown:
            self.show_orientation_axes(self.flipz)
        #self.scene.mlab.outline() # affects zoom after segmenting
        #self.scene.mlab.axes() # need to adjust units to microns
        print("Scene activated with view:", view, "roll:", roll)
    
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
        
        # process ROI in prep for showing filtered 2D view and segmenting
        self._segs_visible = [BlobsVisibilityOptions.VISIBLE.value]
        offset = self._curr_offset()
        roi_size = self.roi_array[0].astype(int)
        self.roi = plot_3d.prepare_roi(config.image5d, offset, roi_size)
        if not libmag.is_binary(self.roi):
            self.roi = plot_3d.saturate_roi(
                self.roi, channel=config.channel)
            self.roi = plot_3d.denoise_roi(self.roi, config.channel)
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
            segs_all = detector.detect_blobs(roi, config.channel)
            
            if ColocalizeOptions.MATCHES.value in self._colocalize:
                # match blobs between two channels
                verify_tol = np.multiply(
                    chunking.calc_overlap(),
                    config.roi_profile["verify_tol_factor"])
                matches = colocalizer.colocalize_blobs_match(
                    segs_all, np.zeros(3, dtype=int), roi_size, verify_tol,
                    np.zeros(3, dtype=int))
                if matches:
                    # TODO: include all channel combos
                    self.blobs.blob_matches = matches[tuple(matches.keys())[0]]
        else:
            # get all previously processed blobs in ROI plus additional 
            # padding region to show surrounding blobs
            # TODO: set segs_all to None rather than empty list if no blobs?
            print("Selecting blobs in ROI from loaded blobs")
            segs_all, mask = detector.get_blobs_in_roi(
                config.blobs.blobs, offset, roi_size, self._margin)
            
            # shift coordinates to be relative to offset
            segs_all[:, :3] = np.subtract(segs_all[:, :3], offset[::-1])
            segs_all = detector.format_blobs(segs_all)
            segs_all, mask_chl = detector.blobs_in_channel(
                segs_all, config.channel, return_mask=True)
            if ColocalizeOptions.MATCHES.value in self._colocalize:
                # get blob matches from whole-image match colocalization,
                # shifting blobs to relative coordinates
                matches = colocalizer.select_matches(
                    config.db, config.channel, offset[::-1], roi_size[::-1])
                # TODO: include all channel combos
                if matches is not None:
                    matches = matches[tuple(matches.keys())[0]]
                    shift = [n * -1 for n in offset[::-1]]
                    matches.update_blobs(detector.shift_blob_rel_coords, shift)
                self.blobs.blob_matches = matches
                print("loaded blob matches:\n", self.blobs.blob_matches)
            elif (ColocalizeOptions.INTENSITY.value in self._colocalize
                  and config.blobs.colocalizations is not None
                  and segs is None):
                # get corresponding blob co-localizations unless showing
                # blobs from database, which do not have colocs
                colocs = config.blobs.colocalizations[mask][mask_chl]
        print("segs_all:\n{}".format(segs_all))
        
        if segs is not None:
            # segs are typically loaded from DB for a sub-ROI within the
            # current ROI, so fill in the padding area from segs_all
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
                # same colors as corresponding atlas labels
                blob_ids = ontology.get_label_ids_from_position(
                    segs_all[self.segs_in_mask, :3].astype(np.int),
                    config.labels_img)
                self.segs_cmap = config.cmap_labels(
                    config.cmap_labels.convert_img_labels(blob_ids))
                self.segs_cmap[:, :3] *= 255
                self.segs_cmap[:, 3] *= alpha
            else:
                # default to color by channel
                cmap = colormaps.discrete_colormap(
                    np_io.get_num_channels(config.image5d), alpha, True,
                    config.seed)
                self.segs_cmap = cmap[detector.get_blobs_channel(
                    segs_all[self.segs_in_mask]).astype(np.int)]
        
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
        
        if (self.selected_viewer_tab is ViewerTabs.ROI_ED or
                self.selected_viewer_tab is ViewerTabs.MAYAVI and
                self.stale_viewers[ViewerTabs.MAYAVI]):
            # currently must redraw ROI Editor to include blobs; redraw
            # 3D viewer if stale and only if shown to limit performance impact
            # of 3D display; Atlas Editor does not show blobs
            self.redraw_selected_viewer(clear=False)
        
        if (self.selected_viewer_tab is ViewerTabs.MAYAVI or
                not self.stale_viewers[ViewerTabs.MAYAVI]):
            # show 3D blobs if 3D viewer is showing; if not, add to 3D display
            # if it is not stale so blobs do not need to be redetected to show
            self.show_3d_blobs()
        
        if ColocalizeOptions.INTENSITY.value in self._colocalize:
            # perform intensity-based colocalization
            self._colocalize_blobs()
    
    def show_3d_blobs(self):
        """Show blobs as spheres in 3D viewer."""
        if self.segments is None or len(self.segments) < 1:
            return
        
        # get blobs in ROI and display as spheres in Mayavi viewer
        roi_size = self.roi_array[0].astype(int)
        show_shadows = Vis3dOptions.SHADOWS.value in self._check_list_3d
        self.segs_pts, scale = self._vis3d.show_blobs(
            self.segments, self.segs_in_mask, self.segs_cmap,
            self._curr_offset()[::-1], roi_size[::-1], show_shadows, self.flipz)
        
        # reduce number of digits to make the slider more compact
        scale = float(libmag.format_num(scale, 4))
        self._scale_detections_high = scale * 2
        self.scale_detections = scale

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
        filename_base = importer.filename_to_base(
            config.filename, config.series)
        grid = self._DEFAULTS_2D[3] in self._check_list_2d
        stack_args = (
            self.update_segment, filename_base, config.channel,
            curr_roi_size, curr_offset, self.segs_in_mask,
            self.segs_cmap, self._roi_ed_close_listener,
            # additional args with defaults
            self._full_border(self.border), self._planes_2d[0].lower())
        roi_ed = roi_editor.ROIEditor(
            config.image5d, config.labels_img, self._img_region,
            self.show_label_3d, self.update_status_bar_msg)
        roi_ed.max_intens_proj = self._DEFAULTS_2D[4] in self._check_list_2d
        roi_ed.zoom_shift = config.plot_labels[config.PlotLabels.ZOOM_SHIFT]
        roi_ed.fn_update_coords = self.set_offset
        roi_ed.fn_redraw = self.redraw_selected_viewer
        roi_ed.blobs = self.blobs
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
            screenshot = self.scene.mlab.screenshot(
                mode="rgba", antialiased=True) if self.scene_3d_shown else None
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
        roi_ed.set_show_labels(self._get_show_labels())
        self.roi_ed = roi_ed
        self._add_mpl_fig_handlers(roi_ed.fig)
        self.stale_viewers[ViewerTabs.ROI_ED] = None

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
            config.image5d, config.labels_img, config.channel, 
            self._curr_offset(center=False), self._atlas_ed_close_listener,
            config.borders_img, self.show_label_3d, title,
            self._refresh_atlas_eds, self._atlas_ed_fig,
            self.update_status_bar_msg)
        self.atlas_eds.append(atlas_ed)
        
        # show the Atlas Editor
        if self._DEFAULTS_2D[4] in self._check_list_2d:
            # show max intensity projection planes based on ROI size
            atlas_ed.update_max_intens_proj(self.get_roi_size())
        atlas_ed.fn_update_coords = self.set_offset
        atlas_ed.show_atlas()
        atlas_ed.set_show_labels(self._get_show_labels())
        self._add_mpl_fig_handlers(atlas_ed.fig)
        self.stale_viewers[ViewerTabs.ATLAS_ED] = None
    
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
            coords = self._curr_offset()[::-1]
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
    
    def _get_show_labels(self):
        """Get the current value of the show atlas labels option.
        
        Returns:
            bool: True if the box is selected, False otherwise.

        """
        return AtlasEditorOptions.SHOW_LABELS.value in self._atlas_ed_options
    
    @on_trait_change("_atlas_ed_options")
    def _atlas_ed_options_changed(self):
        """Respond to atlas editor option changes."""
        # toggle atlas show labels attributes in ROI and Atlas Editors
        show_labels = self._get_show_labels()
        if self.roi_ed:
            self.roi_ed.set_show_labels(show_labels)
        if self.atlas_eds:
            self.atlas_eds[0].set_show_labels(show_labels)
        
        if self.selected_viewer_tab is ViewerTabs.ATLAS_ED:
            # move visible Atlas Editor to ROI offset if sync selected;
            # otherwise, defer sync to tab selection handler
            self.sync_atlas_eds_coords(check_option=True)
    
    @on_trait_change("_atlas_ed_zoom")
    def _zoom_atlas_ed(self):
        """Zoom the Atlas Editor into the ROI.
        
        The ROI offset is normally at the crosshairs unless the ROI offset
        is treated as the ROI center, in which case the crosshairs remain
        in the same place but the retrieved offset is shifted to the ROI's
        upper left corner.
        
        """
        # assume that the first Atlas Editor is to be zoomed
        if not self.atlas_eds: return
        atlas_ed = self.atlas_eds[0]
        if self._atlas_ed_zoom:
            # zoom into the ROI
            offset = self._curr_offset()
            shape = self.roi_array[0].astype(int)
        else:
            # zoom out to the full image
            offset = np.zeros(3, dtype=int)
            shape = self._get_max_offset()
        atlas_ed.view_subimg(offset[::-1], shape[::-1])

    @staticmethod
    def _get_save_path(default_path):
        """Get a save path from the user through a file dialog.
        
        Args:
            default_path (str): Default path to display in the dialog.

        Returns:
            str: Chosen path.
        
        Raises:
            FileNotFoundError: User canceled file selection.

        """
        # open a PyFace file dialog in save mode
        save_dialog = FileDialog(action="save as", default_path=default_path)
        if save_dialog.open() == OK:
            # get user selected path
            return save_dialog.path
        else:
            # user canceled file selection
            raise FileNotFoundError("User canceled file selection")
    
    @on_trait_change("_btn_save_fig")
    def _save_fig(self):
        """Save the figure in the currently selected viewer."""
        path = None
        try:
            if self.selected_viewer_tab is ViewerTabs.ROI_ED:
                if self.roi_ed is not None:
                    # save screenshot of current ROI Editor
                    path = self._get_save_path(self.roi_ed.get_save_path())
                    self.roi_ed.save_fig(path)
            elif self.selected_viewer_tab is ViewerTabs.ATLAS_ED:
                if self.atlas_eds:
                    # save screenshot of first Atlas Editor
                    # TODO: find active editor
                    path = self._get_save_path(self.atlas_eds[0].get_save_path())
                    self.atlas_eds[0].save_fig(path)
            elif self.selected_viewer_tab is ViewerTabs.MAYAVI:
                if config.filename:
                    # save 3D image with extension in config
                    screenshot = self.scene.mlab.screenshot(
                        mode="rgba", antialiased=True)
                    ext = (config.savefig if config.savefig else
                           config.DEFAULT_SAVEFIG)
                    path = "{}.{}".format(naming.get_roi_path(
                        config.filename, self._curr_offset(),
                        self.roi_array[0].astype(int)), ext)
                    path = self._get_save_path(path)
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
            roi_id = roi["id"]
            blobs = config.db.select_blobs_by_roi(roi_id)[0]
            if len(blobs) > 0:
                # change to single-channel if all blobs are from same channel
                chls = np.unique(detector.get_blobs_channel(blobs))
                if len(chls) == 1:
                    self._channel = [str(int(chls[0]))]
            
            # get matches between blobs, such as verifications
            blob_matches = config.db.select_blob_matches(roi_id)
            blob_matches.update_blobs(
                detector.shift_blob_rel_coords,
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
            self.rois_check_list = _ROI_DEFAULT
            self._reset_segments()
        
        # immediately update in Atlas Editors, where None uses the default max
        # intensity projection settings and 0's turns it off
        mip_shape = (None if self._DEFAULTS_2D[4] in self._check_list_2d
                     else (0, 0, 0))
        self._update_mip(mip_shape)
    
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
            self._roi_feedback = "No labels image loaded to find region"
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
            config.labels_ref_lookup, region_ids, config.labels_img,
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
                np.around(np.divide(curr_roi_size[::-1], 2)).astype(np.int))
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
            offset = plot_3d.roi_center_to_offset(
                offset, self.roi_array[0].astype(int)[::-1], reverse=True)
        self.z_offset, self.y_offset, self.x_offset = offset
    
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
        detector.set_blob_confirmed(seg, -1)
    
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
            # new blob's abs coords are not shifted, so shift new blob's abs
            # coordinates by relative coords' diff between old and new blobs
            # TODO: consider requiring new seg to already have abs coord updated
            self._segs_moved.append(segment_old)
            diff = np.subtract(seg[:3], segment_old[:3])
            detector.shift_blob_abs_coords(seg, diff)
            segi = self._get_vis_segments_index(segment_old)
            if segi == -1:
                # try to find old blob from deleted blobs
                self._flag_seg_for_deletion(segment_old)
                segi = self._get_vis_segments_index(segment_old)
            if segi != -1:
                # replace corresponding blob entry in table
                self.segments[segi] = seg
                print("updated seg: {}".format(seg))
                self._force_seg_refresh(segi, show=True)
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

    @on_trait_change("_profiles_cats")
    def _update_profiles_names(self):
        """Update the profile names dropdown box based on the selected
        profiles category.
        """
        # get base profile matching category
        cat = self._profiles_cats[0]
        if cat == ProfileCats.ROI.value:
            prof = config.roi_profile
        elif cat == ProfileCats.ATLAS.value:
            prof = config.atlas_profile
        else:
            prof = config.grid_search_profile

        if prof:
            # add profiles and matching configuration files
            prof_names = list(prof.profiles.keys())
            prof_names.extend(prof.get_files())
        else:
            # clear profile names
            prof_names = []
        # update dropdown box selections and choose the first item, required
        # even though the item will appear to be selected by default
        self._profiles_names = TraitsList()
        self._profiles_names.selections = prof_names
        self._profiles_name = prof_names[0]

    @on_trait_change("_profiles_add_btn")
    def _add_profile(self):
        """Add the chosen profile to the profiles table."""
        # construct profile from selected options
        for chl in self._profiles_chls:
            prof = [self._profiles_cats[0], self._profiles_name, chl]
            print("profile to add", prof)
            self._profiles.append(prof)

    @on_trait_change("_profiles_load_btn")
    def _load_profiles(self):
        """Load profiles based on profiles added to the table."""
        # update profile names list
        self._update_profiles_names()
        
        print("profiles from table:\n", self._profiles)
        if not self._profiles:
            # no profiles in the table to load
            return
        
        # convert to Numpy array for fancy indexing
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

        # set up all profiles
        cli.setup_roi_profiles(roi_profs)
        cli.setup_atlas_profiles(grid_profs)
        cli.setup_grid_search_profiles(grid_profs)

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
                    "Please enter at least image output and data type "
                    "before importing.")
        
        for suffix in (config.SUFFIX_IMAGE5D, config.SUFFIX_META,
                       config.SUFFIX_SUBIMG):
            if self._import_browser.endswith(suffix):
                # file already imported; initiate load and change to ROI panel
                self._update_roi_feedback(
                    "{} is already imported, loading image"
                    .format(self._import_browser))
                # must assign before tab change or else _import_browser is empty
                self._filename = self._import_browser
                self.select_controls_tab = ControlsTabs.ROI.value
                return
        
        # reset import fields
        self._clear_import_files(False)
        chl_paths = None
        base_path = None

        try:
            if os.path.isdir(self._import_browser):
                # gather files within the directory to import
                self._import_mode = ImportModes.DIR
                chl_paths, import_md = importer.setup_import_dir(
                    self._import_browser)
                setup_import(import_md)
                base_path = os.path.join(
                    os.path.dirname(self._import_browser),
                    importer.DEFAULT_IMG_STACK_NAME)
                self._import_btn_enabled = True
            
            elif self._import_browser:
                # gather files matching the pattern of the selected file to import
                self._import_mode = ImportModes.MULTIPAGE
                chl_paths, base_path = importer.setup_import_multipage(
                    self._import_browser)
                
                # extract metadata in separate thread given delay from Java
                # initialization for Bioformats
                self._update_import_feedback(
                    "Gathering metadata related to {}, please wait"
                    "...".format(self._import_browser))
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
                "File to import does not exist: {}\nPlease try another file"
                .format(self._import_browser))
    
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
        def update_filename():
            # update image path and trigger loading the image
            self._filename = ""
            self._filename = self._import_prefix
        
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
                self._update_import_feedback, update_filename)
            self._import_thread.start()
    
    @on_trait_change("_import_clear_btn")
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
        self._import_shape = np.zeros((1, 5), dtype=np.int)
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

    def _update_roi_feedback(self, val, print_out=False):
        """Update the ROI panel feedback text box.

        Args:
            val (str): String to append as a new line.
            print_out (bool): True to print to console as well; defaults
                to False.

        """
        if print_out:
            print(val)
        self._roi_feedback += "{}\n".format(val)


if __name__ == "__main__":
    print("Starting the MagellanMapper graphical interface...")
    cli.main()
    main()
