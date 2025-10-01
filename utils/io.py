"""Input/output for file and path operations"""

# --------- Standard library imports ---------#
import os
from datetime import datetime

# --------- Local imports ---------#
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
utilities_config = config_manager.get('utilities_config')

# --------- Model filename function ---------#
def rename_path(base_path, env_name, agent_name, item_type, timestamp=None, extension=None):
    if timestamp is None:
        timestamp = datetime.now().strftime(utilities_config['date_time'])

    file_name = f'{env_name}_{agent_name}_{item_type}_{timestamp}'

    if extension:
        file_name += f'.{extension}'

    full_path = os.path.join(base_path, file_name)

    if not extension:
        os.makedirs(full_path, exist_ok=True)

    return full_path