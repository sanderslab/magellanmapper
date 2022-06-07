# Atlas-related PyQt5 threads
from typing import Any, Callable, Optional

from PyQt5 import QtCore

from magmap.atlas import ontology
from magmap.settings import config

_logger = config.logger.getChild(__name__)


class RemapLevelThread(QtCore.QThread):
    """Thread for remapping a labels image to a different atlas level."""
    
    signal = QtCore.pyqtSignal(object)
    signal_prog = QtCore.pyqtSignal(object, object)
    
    def __init__(
            self, level: Optional[int], fn_success: Callable[[Any], None],
            fn_prog: Callable[[int, str], None]):
        """Initialize the import thread."""
        super().__init__()
        self.level: Optional[int] = level
        self.signal.connect(fn_success)
        self.signal_prog.connect(fn_prog)
    
    def update_prog(self, pct, msg):
        """Update progress bar."""
        self.signal_prog.emit(pct, msg)
    
    def run(self):
        """Set up image import metadata."""
        self.update_prog(0, "Initializing atlas level remapping")
        if (config.labels_img is None or config.labels_ref is None or
                config.labels_ref.ref_lookup is None or self.level is None):
            # skip if labels, reference, or level are not available
            self.update_prog(0, "Labels image or reference not available")
            return
        
        # remap atlas labels image to the given level
        labels_np = ontology.make_labels_level(
            config.labels_img, config.labels_ref, self.level, self.update_prog)
        self.signal.emit(labels_np)
        self.update_prog(100, "Completed atlas level remapping")

