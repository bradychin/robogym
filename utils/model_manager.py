"""Model management for loading, saving, and finding models"""

# --------- Standard library imports ---------#
import os
import glob
import shutil

# --------- Third-party imports ---------#
from stable_baselines3 import PPO

# --------- Local imports ---------#
from utils.io import rename_path
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)

# --------- Find latest model function ---------#
def find_latest_model(env_name, agent_name):
    """Find the most recent model for a given environment and agent"""
    pattern = os.path.join(paths_config['best_model_path'], f'{env_name}_{agent_name}_model_*.zip')
    models = glob.glob(pattern)

    if not models:
        return None

    latest_model = max(models, key=os.path.getmtime)
    return latest_model

# --------- Load model function ---------#
def load_model(agent, model_path):
    try:
        logger.info(f'Loading model from: {model_path}')
        if hasattr(agent, 'get_algorithm_class'):
            model_class = agent.get_algorithm_class()
        else:
            logger.warning("Agent doesn't have _get_algorithm_class(). Falling back to PPO")
            model_class = PPO

        agent.model = model_class.load(model_path)
        logger.info('Model loaded successfully!')
        return True
    except FileNotFoundError:
        logger.error(f'Model file not found: {model_path}')
        return False
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        return False

# --------- Save trained model function ---------#
def save_model(env_name, agent_name, source_path=None):
    if source_path is None:
        source_path = os.path.join(paths_config['best_model_path'], 'best_model.zip')

    if not os.path.exists(source_path):
        logger.warning(f'Model file not found at: {source_path}')
        return None

    try:
        model_path = rename_path(paths_config['best_model_path'],
                                 env_name,
                                 agent_name,
                                 'model',
                                 extension='zip')
        shutil.move(source_path, model_path)
        logger.info(f'Model saved to: {model_path}')
        return model_path
    except Exception as e:
        logger.error(f'Failed to save model: {e}')
        return None