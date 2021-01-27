# Image file import PyQt5 threads

from PyQt5 import QtCore

from magmap.gui import visualizer
from magmap.io import importer
from magmap.settings import config


class SetupImportThread(QtCore.QThread):
    """Thread for setting up file import by extracting image metadata.

    Attributes:
        chl_paths (dict[int, List[str]]): Dictionary of channel numbers
            to sequences of paths within the given channel.
        fn_success (func): Signal taking
            no arguments, to be emitted upon successfull import; defaults
            to None.

    """
    
    signal = QtCore.pyqtSignal(object)
    
    def __init__(self, chl_paths, fn_success):
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
        mode (:obj:`ImportModes`): Import mode enum.
        prefix (str): Destination base path from which the output path
            will be constructed.
        chl_paths (dict[int, List[str]]): Dictionary of channel numbers
            to sequences of paths within the given channel.
        import_md (dict[:obj:`config.MetaKeys`]): Import metadata dictionary.
        fn_feedback (func): Function taking a string to display feedback;
            defaults to None.
        fn_success (func): Function taking no arguments to be called upon
            successfull import; defaults to None.
    
    """
    
    signal = QtCore.pyqtSignal()
    
    def __init__(self, mode, prefix, chl_paths, import_md, fn_feedback=None,
                 fn_success=None):
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
