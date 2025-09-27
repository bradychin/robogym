# --------- Standard library imports ---------#
import os
import shutil
from datetime import datetime

# --------- Local imports ---------#
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config imports ---------#
from utils.config_manager import config_manager
utilities_config = config_manager.get('utilities_config')

# --------- Timestamp function ---------#
def add_timestamp(best_model_path, tensorboard_path):
    timestamp = datetime.now().strftime(utilities_config.date_time)
    model_file = os.path.join(best_model_path, 'best_model.zip')

    if os.path.exists(model_file):
        logger.info(f'Archiving previous model with timestamp: {timestamp}.')
        timestamped_model = os.path.join(best_model_path, f'{timestamp}_best_model.zip')
        shutil.move(model_file, timestamped_model)

    if os.path.exists(tensorboard_path):
        logger.info(f'Archiving previous tensorboard logs with timestamp: {timestamp}.')
        timestamped_tb_dir = f'{timestamp}_{tensorboard_path}'
        shutil.move(tensorboard_path, timestamped_tb_dir)

    return timestamp