# --------- Local imports ---------#
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config import ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)

# --------- Choose environment ---------#
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
    logger.info('Creating environment.')
    try:
        # Creating training environment
        training_env = EnvironmentFactory.create(env_name)
        vec_env = training_env.create_vec_env()

        # Creating evaluation environment
        eval_env_wrapper = EnvironmentFactory.create(env_name)
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
                                    tensorboard_log=paths_config['tensorboard_log_path'])
        logger.info(f'Agent "{agent_name}" created.')
    except ValueError as e:
        logger.error(f'Agent creation failed: {e}')
        return

    # Run training pipeline
    try:
        logger.info(f'Starting {agent_name.upper()} training on {env_name}...')

        agent.train(env_config)
        agent.evaluate()
        training_env.demo(agent)
    finally:
        vec_env.close()
        eval_env.close()
        training_env.close()
        eval_env_wrapper.close()

    logger.info('RL pipeline completed successfully!')

if __name__ == '__main__':
    main()