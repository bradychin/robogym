# --------- Standard library imports --------- #
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# --------- Config imports--------- #
from utils.config_manager import ConfigManager
config_manager = ConfigManager()

utilities_config = config_manager.get('utilities_config')

# --------- Set up logging paths --------- #
_custom_log_path = None

def set_log_path(log_path):
    """Set a custom log path for the current run"""
    global _custom_log_path
    _custom_log_path = log_path

def get_log_path():
    """Get current log path"""
    global _custom_log_path
    if _custom_log_path:
        return _custom_log_path

    try:
        paths_config = config_manager.get('paths_config', validate=False)
        return paths_config['log_path']
    except Exception:
        return './log/robogym.log'

# --------- Logging function --------- #
def get_logger(name, log_file=None, console=True):
    """
    Get or create a logger instance

    :param name: Logger name
    :param log_file: Optional specific log file path
    :param console: Whether to ouput to console
    :return: Logger instance
    """
    if log_file is None:
        log_file = get_log_path()

    # Get date format
    try:
        date_format = utilities_config['date_time']
    except Exception as e:
        # Fallback date format
        date_format = '%Y%m%d_%H%M'

    # make sure folder exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(filename)s -> %(message)s',
                                  datefmt=date_format)

    # file handler (all logs)
    fh = TimedRotatingFileHandler(log_file, when="midnight", backupCount=7)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler (optional)
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False
    return logger