# --------- Standard library imports ---------#
import os
import shutil
from datetime import datetime
import glob

# --------- Local imports ---------#
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
utilities_config = config_manager.get('utilities_config')
paths_config = config_manager.get('paths_config')

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

# --------- Find latest model function ---------#
def find_latest_model(env_name, agent_name):
    """Find the most recent model for a given environment and agent"""
    pattern = os.path.join(paths_config['best_model_path'], f'{env_name}_{agent_name}_model_*.zip')
    models = glob.glob(pattern)

    if not models:
        return None

    latest_model = max(models, key=os.path.getmtime)
    return latest_model

# --------- Get action choice function ---------#
def get_action_choice(has_model):
    """Get user's choice for action to perform"""
    if has_model:
        print('\nModel found!')
        print('What would you like to do?')
        print('1. Train a new model')
        print('2. Evaluate the current model')
        print('3. Run a demo on the current model')

        choice = input('\nSelection action (1-3) or enter name: ').strip()
        actions = ['train', 'evaluate', 'demo']

        if choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(actions):
                return actions[choice_idx]

        if choice.lower() in actions:
            return choice.lower()

        print(f'Invalid choice: {choice}')
        return None

    else:
        print('\nNo existing model found.')
        choice = input('Would you like to train a new model? (y/n): ').strip().lower()
        if choice in ['y', 'yes']:
            return 'train'
        else:
            logger.info('Exiting without training.')
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