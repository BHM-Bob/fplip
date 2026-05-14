import inspect
import logging


def get_logger():
    """
    Return a module‑specific logger for the caller.
    No handlers are added – the caller may configure them as they wish.
    """
    name = inspect.getmodule(inspect.stack()[1]).__name__
    logger = logging.getLogger(name if name != '__main__' else 'fplip')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

logger = get_logger()