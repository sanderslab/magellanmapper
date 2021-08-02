"""Panel for BrainGlobe access"""

from PyQt5 import QtCore

from magmap.atlas import brain_globe


class SetupAtlasesThread(QtCore.QThread):
    """Thread for setting up file import by extracting image metadata.

    Attributes:
        fn_success (func): Signal taking
            no arguments, to be emitted upon successfull import; defaults
            to None.

    """
    
    signal = QtCore.pyqtSignal()
    
    def __init__(self, brain_globe_mm, fn_success):
        """Initialize the import thread."""
        super().__init__()
        self.bg_mm: brain_globe.BrainGlobeMM = brain_globe_mm
        self.signal.connect(fn_success)
    
    def run(self):
        """Set up image import metadata."""
        print("running")
        self.bg_mm.get_avail_atlases()
        self.signal.emit()


class AccessAtlasThread(QtCore.QThread):
    """Thread for setting up file import by extracting image metadata.

    Attributes:
        fn_success (func): Signal taking
            no arguments, to be emitted upon successfull import; defaults
            to None.

    """
    
    signal = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str)
    
    def __init__(self, brain_globe_mm, name, fn_success, fn_progress):
        """Initialize the import thread."""
        super().__init__()
        self.bg_mm: brain_globe.BrainGlobeMM = brain_globe_mm
        self.name = name
        print("gonna get atlases")
        self.signal.connect(fn_success)
        self.progress.connect(fn_progress)
        print("connected")
    
    def run(self):
        """Set up image import metadata."""
        print("running")
        self.progress.emit(
            f"Accessing atlas '{self.name}', downloading if necessary...")
        atlas = self.bg_mm.get_atlas(self.name)
        self.progress.emit(f"Atlas '{self.name}' accessed:\n{atlas}")
        self.signal.emit()


class BrainGlobePanel:
    def __init__(self, fn_set_atlases_table, fn_set_feedback):
        self.fn_set_atlases_table = fn_set_atlases_table
        self.fn_set_feedback = fn_set_feedback
        
        # set up BrainGlobe-MagellanMapper interface
        self.bg_mm = brain_globe.BrainGlobeMM()
        
        # fetch listing of available atlases
        self._thread = SetupAtlasesThread(self.bg_mm, self.update_atlas_panel)
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
            if atlases and name not in atlases:
                data.append([name, ver, "Yes"])
        self.fn_set_atlases_table(data)
    
    def open_atlas(self, name):
        self._thread = AccessAtlasThread(
            self.bg_mm, name, self.update_atlas_panel, self.fn_set_feedback)
        self._thread.start()
    
    def remove_atlas(self, name):
        self.bg_mm.remove_local_atlas(name)
        self.fn_set_feedback(f"Removed atlas '{name}'")
        self.update_atlas_panel()
