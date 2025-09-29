# --------- Standard library imports ---------#
import os
import shutil
from datetime import datetime

# --------- Local imports ---------#
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
utilities_config = config_manager.get('utilities_config')

# --------- Get user input function ---------#
def get_user_choice(item_type: str, available_items: list):
    """Function to get user choice"""

    print(f'Available {item_type}:')
    for i, item in enumerate(available_items, 1):
        print(f'{i}, {item}')

    choice = input(f"\nSelect {item_type} (1-{len(available_items)}) or enter name: ").strip()

    # Handle numeric choice
    if choice.isdigit():
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(available_items):
            return available_items[choice_idx]
        else:
            print(f"Invalid choice: {choice}")
            return None

    # Handle name choice
    if choice.lower() in [item.lower() for item in available_items]:
        return choice.lower()

    print(f"Invalid choice: {choice}")
    return None

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

# --------- Timestamp function ---------#
def add_timestamp(best_model_path):
    timestamp = datetime.now().strftime(utilities_config['date_time'])
    model_file = os.path.join(best_model_path, 'best_model.zip')

    if os.path.exists(model_file):
        logger.info(f'Archiving previous model with timestamp: {timestamp}.')
        timestamped_model = os.path.join(best_model_path, f'{timestamp}_best_model.zip')
        shutil.move(model_file, timestamped_model)

    return timestamp