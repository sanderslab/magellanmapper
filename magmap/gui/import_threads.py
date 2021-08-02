# Image file import PyQt5 threads
from typing import Any, Callable, Dict, List, Optional

from PyQt5 import QtCore

from magmap.gui import visualizer
from magmap.io import importer
from magmap.settings import config


class SetupImportThread(QtCore.QThread):
    """Thread for setting up file import by extracting image metadata.

    Attributes:
        chl_paths: Dictionary of channel numbers
            to sequences of paths within the given channel.
        fn_success: Signal function taking a dictionary of metadata enum keys
            to metadata values, to be emitted upon successfull import.

    """
    
    signal = QtCore.pyqtSignal(object)
    
    def __init__(
            self, chl_paths: Dict[int, List[str]],
            fn_success: Callable[[Dict[config.MetaKeys, Any]], None]):
        """Initialize the import thread."""
        super().__init__()
        self.chl_paths = chl_paths
        self.signal.connect(fn_success)
    
    def run(self):
        """Set up image import metadata."""
        # extract metadata for the given image series (eg tile)
        md = importer.setup_import_metadata(
            self.chl_paths, series=config.series)
        self.signal.emit(md)


class ImportThread(QtCore.QThread):
    """Thread for importing files into a Numpy array.
    
    Attributes:
        mode: Import mode enum.
        prefix: Destination base path from which the output path
            will be constructed.
        chl_paths: Dictionary of channel numbers
            to sequences of paths within the given channel.
        import_md: Import metadata dictionary.
        fn_feedback: Function taking a string to display feedback;
            defaults to None.
        fn_success: Function taking no arguments to be called upon
            successfull import; defaults to None.
    
    """
    
    signal = QtCore.pyqtSignal()
    
    def __init__(
            self, mode: "visualizer.ImportModes", prefix: str,
            chl_paths: Dict[int, List[str]],
            import_md: Dict[config.MetaKeys, Any],
            fn_feedback: Optional[Callable[[str], None]] = None,
            fn_success: Optional[Callable[[], None]] = None):
        """Initialize the import thread."""
        super().__init__()
        self.mode = mode
        self.prefix = prefix
        self.chl_paths = chl_paths
        self.import_md = import_md
        self.fn_feedback = fn_feedback
        self.fn_success = fn_success
        if fn_success:
            self.signal.connect(fn_success)

    def run(self):
        """Import files based on the import mode and set up the image."""
        img5d = None
        try:
            if self.mode is visualizer.ImportModes.DIR:
                # import single plane files from a directory
                img5d = importer.import_planes_to_stack(
                    self.chl_paths, self.prefix, self.import_md,
                    fn_feedback=self.fn_feedback)
            elif self.mode is visualizer.ImportModes.MULTIPAGE:
                # import multi-plane files
                img5d = importer.import_multiplane_images(
                    self.chl_paths, self.prefix, self.import_md, config.series,
                    fn_feedback=self.fn_feedback)
        finally:
            if img5d is not None:
                # set up the image for immediate use within MagellanMapper
                self.fn_feedback("Import completed, loading image\n")
                if self.fn_success:
                    self.signal.emit()
            else:
                self.fn_feedback(
                    "Could not complete import, please try again\n")
