# MagellanMapper logging
"""Logging utilities."""

import logging


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
    except (TypeError, ValueError) as e:
        logger.error(e, exc_info=True)
    return logger


def add_file_handler(logger, path):
    """Add a log file handler.
    
    Args:
        logger (:class:`logging.Logger`): Logger to update.
        path (str): Path to log.

    Returns:
        :class:`logging.Logger`: The logger for chained calls.

    """
    handler_file = logging.FileHandler(path)
    handler_file.setLevel(logger.level)
    handler_file.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler_file)
    return logger


def has_file_handler(logger):
    """Checks if a file handler exists in the give logger.
    
    Args:
        logger (:class:`logging.Logger`): Logger.

    Returns:
        bool: True if any file handler exists, False otherwise.

    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return True
    return False
