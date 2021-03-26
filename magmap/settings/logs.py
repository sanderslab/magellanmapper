# MagellanMapper logging
"""Logging utilities."""

import logging


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
