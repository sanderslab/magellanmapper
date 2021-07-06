# Visualization handler
"""TraitsUI handler for Visualization class."""

from enum import Enum, auto
import pathlib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets, QtGui
from traits.trait_base import traits_home
from traitsui.api import Handler

from magmap.gui import event_handlers
from magmap.io import cli
from magmap.settings import config

_logger = config.logger.getChild(__name__)


class VisHandler(Handler):
    """Custom handler for Visualization object events."""
    
    #: :class:`magmap.gui.event_handlers.FileOpenHandler`: File open event
    # handler to retain the object reference.
    _file_open_handler = None

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
        
        # handle file open events such as Apple Events from PyInstaller
        app = QtWidgets.QApplication.instance()
        self._file_open_handler = event_handlers.FileOpenHandler(
            info.object.open_image)
        app.installEventFilter(self._file_open_handler)
        
        # create TraitsUI preferences database if it does not exist
        pathlib.Path(traits_home()).mkdir(parents=True, exist_ok=True)
        db = info.ui.get_ui_db("c")
        if db is not None:
            if config.verbose:
                # show MagellanMapper related db entries
                for k, v in db.items():
                    if k.startswith("magmap"):
                        _logger.debug("TraitsUI preferences for %s: %s", k, v)
            db.close()
        
        # WORKAROUND: TraitsUI icon does not work in Mac; use PyQt directly to
        # display application window icon using abs path; ignored in Windows
        app.setWindowIcon(QtGui.QIcon(str(config.ICON_PATH)))
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
        mpl_figs = info.ui.control.findChildren(FigureCanvasQTAgg)
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


class ViewerTabs(Enum):
    """Enumerations for viewer tabs."""
    ROI_ED = auto()
    ATLAS_ED = auto()
    MAYAVI = auto()


class StaleFlags(Enum):
    """Enumerations for stale viewer states."""
    IMAGE = auto()  # loaded new image
    ROI = auto()  # changed ROI offset or size
