"""Model management for loading, saving, and finding models"""

# --------- Standard library imports ---------#
import os
import glob
from datetime import datetime

# --------- Third-party imports ---------#
from stable_baselines3 import PPO

# --------- Local imports ---------#
from utils.logger import logger, global_logger
logger = logger(__name__)
global_logger = global_logger()

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)

# --------- Find all models function ---------#
def find_all_models(env_name: str, agent_name: str, limit: int=10) -> list[str]:
    """
    Find all models for a given environment and agent

    :param env_name: Name of the environment
    :param agent_name: Name of the agent
    :param limit: Limit of models to search for
    :return: Array of models
    """

    run_pattern = os.path.join(paths_config['run_path'], f'*_{env_name}_{agent_name}', 'model.zip')
    run_models = glob.glob(run_pattern)

    if not run_models:
        logger.info(f'No models found with pattern: {run_pattern}')
        return []

    sorted_models = sorted(run_models, key=os.path.getmtime, reverse=True)[:limit]
    logger.info(f'Found {len(sorted_models)} model(s) for {env_name}_{agent_name}')
    return sorted_models

# --------- Find latest model function ---------#
def find_latest_model(env_name: str, agent_name: str) -> str | None:
    """
    Find the most recent model for a given environment and agent

    :param env_name: Name of the environment
    :param agent_name: Name of the agent
    :return: Most recent model
    """

    models = find_all_models(env_name, agent_name, limit=1)
    return models[0] if models else None

# --------- Load model function ---------#
def load_model(agent, model_path: str) -> bool:
    """
    Load a trained model into the agent

    :param agent: Agent instance to load the model into
    :param model_path: Path to the model file
    :return: True if successful, False if unsuccessful
    """

    try:
        logger.info(f'Loading model from: {model_path}')
        if not os.path.exists(model_path):
            logger.error(f'Model file not found: {model_path}')
            return False

        if hasattr(agent, 'get_algorithm_class'):
            model_class = agent.get_algorithm_class()
        else:
            logger.warning("Agent doesn't have get_algorithm_class(). Falling back to PPO")
            model_class = PPO

        loaded_model = model_class.load(model_path)
        if loaded_model is None:
            logger.error('Model loaded but returned None')
            return False
        agent.model = loaded_model
        logger.info('Model loaded successfully!')
        return True
    except FileNotFoundError:
        logger.error(f'Model file not found: {model_path}')
        return False
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        return False

# ---------Select model for action function ---------#
def select_model_for_action(env_name: str, agent_name: str, action='action'):
    """
    Select an existing model for a user requested action

    :param env_name: Environment name
    :param agent_name: Agent name
    :param action: User requested action
    :return: The existing selected model
    """

    all_models = find_all_models(env_name, agent_name, limit=10)
    if not all_models:
        logger.error(f'No models found for {action}.')
        global_logger.error(f'No models found for {action}.')
    else:
        print(f'\nFound {len(all_models)} model(s):')
        for i, model_path in enumerate(all_models, 1):
            mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            print(f'{i}. {os.path.basename(os.path.dirname(model_path))} - {mtime.strftime("%Y-%m-%d %H:%M")}')
        choice = input(f'\nSelect model (1-{len(all_models)}) or press Enter for most recent: ').strip()

        if choice == '':
            selected_model = all_models[0]
        elif choice.isdigit() and 1 <= int(choice) <= len(all_models):
            selected_model = all_models[int(choice) - 1]
        else:
            logger.error('Invalid selection.')
            global_logger.error('Invalid selection.')
            selected_model = None

        return selected_model

# ---------Select model for action function ---------#
def load_model_for_action(agent, env_name: str, agent_name: str, action_name: str) -> bool:
    """
    Select and load a model for a specific action

    :param agent: Agent instance to load model into
    :param env_name: Environment name
    :param agent_name: Agent name
    :param action_name: Action being performed (display only)
    :return: True if successful, False otherwise
    """

    selected_model = select_model_for_action(env_name, agent_name, action_name)

    if not selected_model:
        logger.error(f'No model selected for {action_name}')
        global_logger.error(f'No model selected for {action_name}')
        return False

    if load_model(agent, selected_model):
        logger.info(f'Model loaded successfully for {action_name}')
        return True
    else:
        logger.error(f'Failed to load model for {action_name}')
        global_logger.error(f'Failed to load model for {action_name}')
        return False