# --------- Standard library imports ---------#
import os
import shutil

from stable_baselines3 import PPO

# --------- Local imports ---------#
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory
from utils.functions import get_user_choice, find_latest_model, get_action_choice
from utils.functions import get_user_choice, find_latest_model
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config import ---------#
from utils.config_manager import ConfigManager
from utils.functions import rename_path
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)

# --------- Setup environment ---------#
def setup_environment(env_name):
    """Setup environment"""
    try:
        # Creating training environment
        training_env = EnvironmentFactory.create(env_name, 'training')
        vec_env = training_env.create_vec_env()

        # Creating evaluation environment
        eval_env_wrapper = EnvironmentFactory.create(env_name, 'evaluation')
        eval_env = eval_env_wrapper.create_env()

        logger.info(f'Environment "{env_name}" created.')
        return training_env, vec_env, eval_env_wrapper, eval_env

    except ValueError as e:
        logger.error(f'Environment created failed: {e}')
        return None, None, None, None

# --------- Setup agent ---------#
def setup_agent(agent_name, vec_env, eval_env, env_name):
    """Setup agent"""
    try:
        agent = AgentFactory.create(agent_name,
                                    vec_env,
                                    eval_env,
                                    tensorboard_log=paths_config['tensorboard_log_path'],
                                    env_name=env_name)
        logger.info(f'Agent "{agent_name}" created.')
        return agent

    except ValueError as e:
        logger.error(f'Agent creation failed: {e}')
        return

# --------- Train a new model ---------#
def train_model(agent, env_config, env_name, agent_name):
    """Train a new model"""
    agent.train(env_config)

    # Move model to a timestamped filename
    best_model_path = os.path.join(paths_config['best_model_path'], 'best_model.zip')
    if os.path.exists(best_model_path):
        model_path = rename_path(paths_config['best_model_path'],
                                 env_name,
                                 agent_name,
                                 'model',
                                 extension='zip')
        shutil.move(best_model_path, model_path)
        logger.info(f'Model saved to: {model_path}')
        return model_path
    return None

# --------- Load existing model function ---------#
def load_existing_model(agent, model_path):
    """Load existing model"""
    try:
        logger.info(f'Loading model from: {model_path}')
        agent.model = PPO.load(model_path)
        logger.info('Model loaded successfully!')
        return True
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        return False

# --------- Main function ---------#
def main():
    logger.info('Starting RL pipeline.')

    # Choose environment
    available_envs = EnvironmentFactory.get_available_environments()
    env_name = get_user_choice('environment', available_envs)
    if env_name is None:
        return

    # Choose agent
    available_agents = AgentFactory.get_available_agents()
    agent_name = get_user_choice('agent', available_agents)
    if agent_name is None:
        return

    # Check for existing model
    existing_model = find_latest_model(env_name, agent_name)
    has_model = existing_model is not None

    if has_model:
        logger.info(f'Found existing model: {existing_model}')

    # Get users action choice
    action = get_action_choice(has_model)
    if action is None:
        return

    # Load environment config
    try:
        config_name = f'{env_name}_config'
        env_config = config_manager.get(config_name)
        config_manager.validate_config(env_config)
        logger.info(f'Loaded configuration for {env_name}: {env_config}')
    except (FileNotFoundError, ImportError, ValueError) as e:
        logger.error(f'Configuration loading failed: {e}')
        return

    # Setup environment
    training_env, vec_env, eval_env_wrapper, eval_env = setup_environment(env_name)
    if vec_env is None:
        return

    # Setup agent
    agent = setup_agent(agent_name, vec_env, eval_env, env_name)
    if agent is None:
        vec_env.close()
        eval_env.close()
        return

    # Execute requested action
    try:
        if action == 'train':
            train_model(agent, env_config, env_name, agent_name)
            # Ask if user wants to evaluate or demo after training
            follow_up = input('\nWould you like to (e)valuate or (d)emo the model? (e/d/n): ').strip().lower()
            if follow_up == 'e':
                agent.evaluate()
            elif follow_up == 'd':
                training_env.demo(agent, max_steps=env_config.get('demo', {}).get('max_steps', 2000))

        elif action == 'evaluate':
            if load_existing_model(agent, existing_model):
                agent.evaluate()

        elif action == 'demo':
            if load_existing_model(agent, existing_model):
                max_steps = env_config.get('demo', {}).get('max_steps', 2000)
                training_env.demo(agent, max_steps=max_steps)

    except KeyboardInterrupt:
        logger.warning('Operation interrupted by user.')
    except Exception as e:
        logger.error(f'Operation failed: {e}')
    finally:
        vec_env.close()
        eval_env.close()
        training_env.close()
        eval_env_wrapper.close()

    logger.info('RL pipeline completed successfully!')

if __name__ == '__main__':
    main()