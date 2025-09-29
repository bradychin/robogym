# --------- Standard library imports ---------#
import os
import shutil

# --------- Local imports ---------#
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory
from utils.functions import get_user_choice
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config import ---------#
from utils.config_manager import ConfigManager
from utils.functions import rename_path
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)

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
    try:
        # Creating training environment
        training_env = EnvironmentFactory.create(env_name, 'training')
        vec_env = training_env.create_vec_env()

        # Creating evaluation environment
        eval_env_wrapper = EnvironmentFactory.create(env_name, 'evaluation')
        eval_env = eval_env_wrapper.create_env()

        logger.info(f'Environment "{env_name}" created.')
    except ValueError as e:
        logger.error(f'Environment created failed: {e}')
        return

    # Setup agent
    try:
        agent = AgentFactory.create(agent_name,
                                    vec_env,
                                    eval_env,
                                    tensorboard_log=paths_config['tensorboard_log_path'],
                                    env_name=env_name)
        logger.info(f'Agent "{agent_name}" created.')
    except ValueError as e:
        logger.error(f'Agent creation failed: {e}')
        return

    # Run
    try:
        # Train
        agent.train(env_config)
        best_model_path = os.path.join(paths_config['best_model_path'], 'best_model.zip')
        if os.path.exists(best_model_path):
            model_path = rename_path(paths_config['best_model_path'],
                                                env_name,
                                                agent_name,
                                                'model',
                                                extension='zip')
            shutil.move(best_model_path, model_path)

        # Evaluate
        agent.evaluate()
        # Demo
        training_env.demo(agent)
    finally:
        vec_env.close()
        eval_env.close()
        training_env.close()
        eval_env_wrapper.close()

    logger.info('RL pipeline completed successfully!')

if __name__ == '__main__':
    main()