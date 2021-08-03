"""Panel controller for BrainGlobe access"""

from typing import Callable, Optional, Sequence, TYPE_CHECKING

from PyQt5 import QtCore

from magmap.brain_globe import bg_model

if TYPE_CHECKING:
    from bg_atlasapi import BrainGlobeAtlas


class SetupAtlasesThread(QtCore.QThread):
    """Thread for setting atlases by fetching the BrainGlobe atlas listing.

    Attributes:
        brain_globe_mm: BrainGlobe-MagellanMapper model.
        fn_success: Signal function taking no arguments, to be emitted upon
            successfull import.
        fn_progress: Signal function taking a string argument to emit feedback.

    """
    
    signal = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str)
    
    def __init__(
            self, brain_globe_mm: bg_model.BrainGlobeMM,
            fn_success: Callable[[], None], fn_progress: Callable[[str], None]):
        """Initialize the setup thread."""
        super().__init__()
        self.bg_mm: bg_model.BrainGlobeMM = brain_globe_mm
        self.signal.connect(fn_success)
        self.progress.connect(fn_progress)
    
    def run(self):
        """Fetch the atlas listing."""
        atlases = self.bg_mm.get_avail_atlases()
        msg = ("Fetched atlases available from BrainGlobe" if atlases
               else "Unable to access atlas listing from BrainGlobe. "
                    "Showing atlases dowloaded from BrainGlobe.")
        self.progress.emit(msg)
        self.signal.emit()


class AccessAtlasThread(QtCore.QThread):
    """Thread for setting up a specific access.

    Attributes:
        fn_success: Signal function taking no arguments, to be emitted upon
            successfull import.
        fn_progress: Signal function taking a string argument to emit feedback.

    """
    
    signal = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(str)
    
    def __init__(
            self, brain_globe_mm: bg_model.BrainGlobeMM, name: str,
            fn_success: Callable[[], None], fn_progress: Callable[[str], None]):
        """Initialize the atlas access thread."""
        super().__init__()
        self.bg_mm: bg_model.BrainGlobeMM = brain_globe_mm
        self.name = name
        self.signal.connect(fn_success)
        self.progress.connect(fn_progress)
    
    def run(self):
        """Access the atlas, including download if necessary."""
        self.progress.emit(
            f"Accessing atlas '{self.name}', downloading if necessary...")
        atlas = self.bg_mm.get_atlas(self.name)
        self.progress.emit(f"Atlas '{self.name}' accessed:\n{atlas}")
        self.signal.emit(atlas)


class BrainGlobeCtrl:
    def __init__(
            self, fn_set_atlases_table: Callable[[Sequence], None],
            fn_set_feedback: Callable[[str], None],
            fn_opened_atlas: Optional[Callable[
                ["BrainGlobeAtlas"], None]] = None):
        # set up attributes
        self.fn_set_atlases_table = fn_set_atlases_table
        self.fn_set_feedback = fn_set_feedback
        self.fn_opened_atlas = fn_opened_atlas
        
        # set up BrainGlobe-MagellanMapper interface
        self.bg_mm = bg_model.BrainGlobeMM()
        
        # fetch listing of available atlases
        self._thread = SetupAtlasesThread(
            self.bg_mm, self.update_atlas_panel, self.fn_set_feedback)
        self._thread.start()
    
    def update_atlas_panel(self):
        atlases = self.bg_mm.atlases_avail
        atlases_local = self.bg_mm.get_local_atlases()
        data = []
        if atlases:
            for name, ver in atlases.items():
                if name in atlases_local:
                    installed = (
                        "Yes, latest version" if atlases_local[name] == ver
                        else "Update available")
                else:
                    installed = "No"
                data.append([name, ver, installed])
        for name, ver in atlases_local.items():
            if not atlases or name not in atlases:
                data.append([name, ver, "Yes"])
        self.fn_set_atlases_table(data)
    
    def _open_atlas_handler(self, atlas):
        self.update_atlas_panel()
        if self.fn_opened_atlas:
            self.fn_opened_atlas(atlas)
    
    def open_atlas(self, name):
        self._thread = AccessAtlasThread(
            self.bg_mm, name, self._open_atlas_handler, self.fn_set_feedback)
        self._thread.start()
    
    def remove_atlas(self, name):
        self.bg_mm.remove_local_atlas(name)
        self.fn_set_feedback(f"Removed atlas '{name}'")
        self.update_atlas_panel()
