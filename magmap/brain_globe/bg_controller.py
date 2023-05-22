"""Panel controller for BrainGlobe access"""

from typing import Callable, Optional, Sequence, TYPE_CHECKING

from PyQt5 import QtCore

from magmap.brain_globe import bg_model
from magmap.io import libmag

if TYPE_CHECKING:
    from bg_atlasapi import BrainGlobeAtlas


class SetupAtlasesThread(QtCore.QThread):
    """Thread for setting atlases by fetching the BrainGlobe atlas listing.

    """
    
    signal = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int, str)
    
    def __init__(
            self, brain_globe_mm,
            fn_success: Callable[[], None],
            fn_progress: Callable[[int, str], None]):
        """Initialize the setup thread.
        
        Params:
            fn_success: Signal function taking no arguments, to be emitted upon
                successfull import.
            fn_progress: Signal function taking a string argument to emit
                feedback.
        
        """
        super().__init__()
        #: BrainGlobe-MagellanMapper model.
        self.bg_mm: "bg_model.BrainGlobeMM" = brain_globe_mm
        
        self.signal.connect(fn_success)
        self.progress.connect(fn_progress)
    
    def update_prog(self, pct, msg):
        """Update progress bar."""
        self.progress.emit(pct, msg)

    def run(self):
        """Fetch the atlas listing."""
        self.progress.emit(1, "Getting available BrainGlobe atlases")
        atlases = self.bg_mm.get_avail_atlases()
        msg = ("Fetched atlases available from BrainGlobe" if atlases
               else "Unable to access atlas listing from BrainGlobe. "
                    "Showing atlases dowloaded from BrainGlobe.")
        self.progress.emit(100, msg)
        self.signal.emit()


class AccessAtlasThread(QtCore.QThread):
    """Thread for setting up a specific access.

    """
    
    signal = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int, str)
    
    def __init__(
            self, brain_globe_mm, name,
            fn_success: Callable[[], None],
            fn_progress: Callable[[int, str], None]):
        """Initialize the atlas access thread.
        
        Params:
            fn_success: Signal function taking no arguments, to be emitted upon
                successfull import.
            fn_progress: Signal function taking a string argument to emit
                feedback.
        
        """
        super().__init__()
        self.bg_mm: "bg_model.BrainGlobeMM" = brain_globe_mm
        self.name: str = name
        
        self.signal.connect(fn_success)
        self.progress.connect(fn_progress)
    
    def update_prog(self, done: float, tot: float):
        """Update progress bar for downloading an atlas.
        
        Args:
            done: Amount completed.
            tot: Total amount for completion. 0 indicates that the done
                and total amounts are unknown.

        """
        msg = f"Downloading '{self.name}' atlas"
        if tot == 0:
            # flag % as unknown, eg when content-header is 0
            pct = -1
        else:
            # show % downloaded and size in human-readable units
            pct = (done / tot) * 100
            msg += f": {libmag.format_bytes(done)} of " \
                   f"{libmag.format_bytes(tot)} ({pct:.1f}%)"
        self.progress.emit(pct, msg)

    def run(self):
        """Access the atlas, including download if necessary."""
        # reset progress bar
        self.progress.emit(0, None)
        
        # get atlas
        atlas = self.bg_mm.get_atlas(self.name, fn_update=self.update_prog)
        
        # show retrieved atlas
        self.progress.emit(100, f"Atlas '{self.name}' accessed")
        self.signal.emit(atlas)


class BrainGlobeCtrl:
    """BrainGlobe controller.
    
    """
    def __init__(
            self, fn_set_atlases_table, fn_feedback, fn_progress,
            fn_opened_atlas=None):
        """Initialize the controller."""
        #: Handler for setting the atlases table.
        self.fn_set_atlases_table: Callable[
            [Sequence], None] = fn_set_atlases_table
        #: Handler for outputting feedback messages.
        self.fn_feedback: Callable[[str], None] = fn_feedback
        #: Handler for atlas download progress updates; defaults to None.
        self.fn_progress: Callable[[int, str], None] = fn_progress
        #: Handler for opening an atlas; defaults to None.
        self.fn_opened_atlas: Optional[Callable[
            ["BrainGlobeAtlas"], None]] = fn_opened_atlas
        
        #: BrainGlobe-MagellanMapper interface.
        self.bg_mm = bg_model.BrainGlobeMM()
        
        # fetch listing of available atlases; save thread to avoid garbage
        # collection
        self._thread = SetupAtlasesThread(
            self.bg_mm, self.update_atlas_table, self.fn_progress)
        self._thread.start()
    
    def update_atlas_table(self):
        """Update the atlas table."""
        # use existing listing of available cloud atlases
        atlases = self.bg_mm.atlases_avail
        
        # get updated listing of local atlases
        atlases_local = self.bg_mm.get_local_atlases()
        
        data = []
        if atlases:
            for name, ver in atlases.items():
                # add available atlas to table, checking for local version
                if name in atlases_local:
                    installed = (
                        "Yes, latest version" if atlases_local[name] == ver
                        else "Update available")
                else:
                    installed = "No"
                data.append([name, ver, installed])
        
        for name, ver in atlases_local.items():
            if not atlases or name not in atlases:
                # add local atlas if not listed in cloud atlases
                data.append([name, ver, "Yes"])
        self.fn_set_atlases_table(data)
    
    def _open_atlas_handler(self, atlas: "BrainGlobeAtlas"):
        """Handler to open an atlas."""
        # update table for changes to installed status
        self.update_atlas_table()
        self.fn_feedback(str(atlas))
        if self.fn_opened_atlas:
            # call handler
            self.fn_opened_atlas(atlas)
    
    def open_atlas(self, name: str):
        """Open atlas in a separate thread
        
        Args:
            name: Atlas name

        """
        self._thread = AccessAtlasThread(
            self.bg_mm, name, self._open_atlas_handler, self.fn_progress)
        self._thread.start()
    
    def remove_atlas(self, name: str):
        """Remove local copy of atlas.
        
        Args:
            name: Atlas name.

        """
        self.bg_mm.remove_local_atlas(name)
        self.fn_progress(100, f"Removed atlas '{name}'")
        self.update_atlas_table()
