# --------- Local imports ---------#
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory
from utils.user_interface import get_user_choice, get_action_choice, get_follow_up_action
from utils.model_io import find_latest_model, load_model, save_model
from utils.run_manager import RunManager
from utils.logger import get_logger, set_log_path
logger = get_logger(__name__)

# --------- Config import ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)

# --------- Setup environment ---------#
def setup_environment(env_name):
    """
    Setup training and evaluation environment

    :param env_name: Name of the environment
    :return: training environment, vectorized training environment, evaluation environment and wrapper
    """

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
def setup_agent(agent_name, vec_env, eval_env, env_name, run_manager=None):
    """
    Setup agent

    :param agent_name: Name of the agent
    :param vec_env: Vectorized training environment
    :param eval_env: Evaluation environment
    :param env_name: Name of the environment
    :param run_manager: RunManager instance
    :return: Agent or nothing if an error occurs
    """

    try:
        agent = AgentFactory.create(agent_name,
                                    vec_env,
                                    eval_env,
                                    tensorboard_log=paths_config['tensorboard_log_path'],
                                    env_name=env_name,
                                    run_manager=run_manager)
        logger.info(f'Agent "{agent_name}" created.')
        return agent

    except ValueError as e:
        logger.error(f'Agent creation failed: {e}')
        return

# --------- Train a new model ---------#
def train_model(agent, env_config, env_name, agent_name, run_manager):
    """
    Train a new model

    :param agent: Agent to train
    :param env_config: Configurations for the selected environment
    :param env_name: Name of the environment
    :param agent_name: Name of the agent
    :param run_manager: RunManager instance
    :return: Model path of the trained agent
    """

    # save config to run directory
    run_manager.save_config(env_config)

    agent.train(env_config)

    # Create summary
    run_manager.create_summary()

# --------- Main function ---------#
def main():
    """
    Main function.

    The process is as follows:
    1. The user will choose an environment from the given options
    2. The user will choose an agent from the given options
    3. Check if there is an existing model based on the previous options
    4. The user will choose whether to train, evaluate, or demo the model
    5. Load and setup environment and agent
    6. Train, evaluate, or demo
    """

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

    run_manager = None
    if action == 'train':
        run_manager = RunManager(env_name, agent_name)
        set_log_path(run_manager.get_log_path())
        logger.info('Created new run directory')

    # Setup environment
    training_env, vec_env, eval_env_wrapper, eval_env = setup_environment(env_name)
    if vec_env is None:
        return

    # Setup agent
    agent = setup_agent(agent_name, vec_env, eval_env, env_name, run_manager=run_manager)
    if agent is None:
        vec_env.close()
        eval_env.close()
        return

    # Execute requested action
    try:
        if action == 'train':
            logger.info('Train action requested...')
            train_model(agent, env_config, env_name, agent_name, run_manager)
            # Ask if user wants to evaluate or demo after training
            follow_up = get_follow_up_action()
            if follow_up == 'evaluate':
                agent.evaluate()
            elif follow_up == 'demo':
                max_steps = env_config['demo']['max_steps']
                training_env.demo(agent, max_steps=max_steps)

        elif action == 'evaluate':
            logger.info('Evaluate action requested...')
            if load_model(agent, existing_model):
                agent.evaluate()

        elif action == 'demo':
            logger.info('Demo action requested...')
            if load_model(agent, existing_model):
                max_steps = env_config['demo']['max_steps']
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
        if run_manager:
            print(f"\n📁 All files saved to: {run_manager.get_run_dir()}")
            print(f"   View TensorBoard: tensorboard --logdir {run_manager.get_tensorboard_path()}")

        logger.info('RL pipeline completed successfully!')

if __name__ == '__main__':
    main()