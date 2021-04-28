# MagellanMapper logging
"""Logging utilities."""

import logging
from logging import handlers
import pathlib


class LogWriter:
    """File-like object to write standard output to logging functions.
    
    Attributes:
        fn_logging (func): Logging function
        buffer (list[str]): String buffer.
    
    """
    def __init__(self, fn_logging):
        """Create a writer for a logging function."""
        self.fn_logger = fn_logging
        self.buffer = []

    def write(self, msg):
        """Write to logging function with buffering.
        
        Args:
            msg (str): Line to write, from which trailing newlines will be
                removed.

        """
        if msg.endswith("\n"):
            # remove trailing newlines in buffer and pass to logging function
            self.buffer.append(msg.rstrip("\n"))
            self.fn_logger("".join(self.buffer))
            self.buffer = []
        else:
            self.buffer.append(msg)

    def flush(self):
        """Empty function, deferring to logging handler's flush."""
        pass


def setup_logger():
    """Set up a basic root logger with a stream handler.
    
    Returns:
        :class:`logging.Logger`: Root logger for the application.

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # set up handler for console
    handler_stream = logging.StreamHandler()
    handler_stream.setLevel(logging.INFO)
    handler_stream.setFormatter(logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler_stream)
    
    return logger


def update_log_level(logger, level):
    """Update the logging level.
    
    Args:
        logger (:class:`logging.Logger`): Logger to update.
        level (Union[str, int]): Level given either as a string corresponding
            to ``Logger`` levels, or their corresponding integers, ranging
            from 0 (``NOTSET``) to 50 (``CRITICAL``). For convenience,
            values can be given from 0-5, which will be multiplied by 10.

    Returns:
        :class:`logging.Logger`: The logger for chained calls.

    """
    if isinstance(level, str):
        # specify level by level name
        level = level.upper()
    elif isinstance(level, int):
        # specify by level integer (0-50)
        if level < 10:
            # for convenience, assume values under 10 are 10-fold
            level *= 10
    else:
        return
    
    try:
        # set level for the logger and all its handlers
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        if level < logging.INFO:
            # avoid lots of debugging messages from Matplotlib
            logging.getLogger("matplotlib").setLevel(logging.INFO)
    except (TypeError, ValueError) as e:
        logger.error(e, exc_info=True)
    return logger


def add_file_handler(logger, path, backups=5):
    """Add a rotating log file handler with a new log file.
    
    Rotates the file each time this function is called for the given number
    of backups rather than by size or time. Avoids file conflicts from
    multiple instances when permission errors occur (eg on Windows), instead
    creating a log filed named with an incremented number (eg ``out1.log``).
    
    Args:
        logger (:class:`logging.Logger`): Logger to update.
        path (str): Path to log. Increments to ``path<n>.<ext>`` if the
            file at ``path`` cannot be rotated.
        backups (int): Number of backups to maintain; defaults to 5.

    Returns:
        :class:`logging.Logger`: The logger for chained calls.

    """
    # check if log file already exists
    pathl = pathlib.Path(path)
    roll = pathl.is_file()
    
    # create a rotations file handler to manage number of backups while
    # manually managing rollover based on file presence rather than size
    pathl.parent.mkdir(parents=True, exist_ok=True)
    i = 0
    handler_file = None
    while handler_file is None:
        try:
            # if the existing file at path cannot be rotated, increment the
            # filename to create a new series of rotating log files
            path_log = (pathl if i == 0 else
                        f"{pathl.parent / pathl.stem}{i}{pathl.suffix}")
            logger.debug(f"Trying logger path: {path_log}")
            handler_file = handlers.RotatingFileHandler(
                path_log, backupCount=backups)
            if roll:
                # create a new log file if exists, backing up the old one
                handler_file.doRollover()
        except (PermissionError, FileNotFoundError) as e:
            # permission errors occur on Windows if the log is opened by
            # another application instance; file not found errors occur if a
            # backup file is skipped in backup series
            logger.debug(e)
            handler_file = None
            i += 1
    
    handler_file.setLevel(logger.level)
    handler_file.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler_file)
    
    return logger
