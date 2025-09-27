# --------- Standard library imports ---------#
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config')
utilities_config = config_manager.get('utilities.config')

# --------- Logging function ---------#
def get_logger(name, log_file=paths_config.log_path, console=True):
    # make sure folder exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(filename)s -> %(message)s',
                                  datefmt=utilities_config.date_time)

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