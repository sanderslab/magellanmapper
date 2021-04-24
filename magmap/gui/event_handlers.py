# PyQt event handlers
"""Handlers for PyQt events."""

from PyQt5.QtCore import QObject, QEvent

from magmap.settings import config

_logger = config.logger.getChild(__name__)


class FileOpenHandler(QObject):
    """Handle file opening events.
    
    These events are triggered by Apple Events through the PyInstaller
    bootloader.
    
    Attributes:
        fn_open_image (func): Function to open an image, taking the image path.
    
    """
    def __init__(self, fn_open_image, parent=None):
        """Create a new instance of the file open handler.
        
        Args:
            fn_open_image (func): Function to open an image.
            parent (:class:`PyQt5.QtCore.QObject`): Parent object.
        
        """
        super().__init__(parent=parent)
        self.fn_open_image = fn_open_image

    def eventFilter(self, watched, event):
        """Handle open file events.
        
        Args:
            watched (:class:`PyQt5.QtCore.QObject`): Watched object. 
            event (:class:`PyQt5.QtCore.QEvent`): Event oject.

        Returns:
            bool: True if the event was filtered out; otherwise the return
            output from the base class.

        """
        if event.type() == QEvent.FileOpen:
            _logger.debug("File open event:", event.url())
            url = event.url().toString()
            scheme = "file://"
            if url.startswith(scheme):
                # remove file scheme and trigger file opening
                url = url[len(scheme):]
                self.fn_open_image(url)
                return True
        return super().eventFilter(watched, event)
