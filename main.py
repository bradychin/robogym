# --------- Local imports ---------#
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory
from utils.user_interface import get_user_choice, get_action_choice, get_follow_up_action
from utils.model_io import find_latest_model, load_model
from utils.run_manager import RunManager
from utils.logger import logger, global_logger, set_log_path
logger = logger(__name__)
global_logger = global_logger()

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
        logger.info(f'Training environment "{env_name}" created.')

        # Creating evaluation environment
        eval_env_wrapper = EnvironmentFactory.create(env_name, 'evaluation')
        eval_env = eval_env_wrapper.create_env()
        logger.info(f'Evaluation environment "{env_name}" created.')

        return training_env, vec_env, eval_env_wrapper, eval_env

    except ValueError as e:
        logger.error(f'Environment creation failed: {e}')
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

    global_logger.info('=' * 60)
    global_logger.info('Starting RL pipeline.')

    # Choose environment
    available_envs = EnvironmentFactory.get_available_environments()
    env_name = get_user_choice('environment', available_envs)
    if env_name is None:
        global_logger.warning('No environment selected. Exiting.')
        return
    global_logger.info(f'Selected environment: {env_name}')

    # Choose agent
    available_agents = AgentFactory.get_available_agents()
    agent_name = get_user_choice('agent', available_agents)
    if agent_name is None:
        global_logger.warning('No agent selected. Exiting.')
        return
    global_logger.info(f'Selected agent: {agent_name}')

    # Setup local run manager
    run_manager = RunManager(env_name, agent_name)
    set_log_path(run_manager.get_log_path())

    # Check for existing model
    existing_model = find_latest_model(env_name, agent_name)
    has_model = existing_model is not None

    if has_model:
        global_logger.info(f'Found existing model: {existing_model}')
    else:
        global_logger.info(f'No existing model found.')

    # Get users action choice
    action = get_action_choice(has_model)
    if action is None:
        global_logger.info('No action selected. Exiting.')
        return
    global_logger.info(f'Selected action: {action}')

    # Load environment config
    try:
        config_name = f'{env_name}_config'
        env_config = config_manager.get(config_name)
        config_manager.validate_config(env_config)
    except (FileNotFoundError, ImportError, ValueError) as e:
        logger.error(f'Configuration loading failed: {e}')
        return

    if action == 'train':
        global_logger.info(f'Created run directory: {run_manager.get_run_dir()}')
        logger.info('=' * 60)
        logger.info(f'Environment: {env_name}')
        logger.info(f'Agent: {agent_name}')
        logger.info(f'Configuration: {env_config}')
        logger.info(f'Run directory: {run_manager.get_run_dir()}')
        logger.info('=' * 60)

    # Setup environment
    training_env, vec_env, eval_env_wrapper, eval_env = setup_environment(env_name)
    if vec_env is None:
        logger.error('Environment setup failed. Exiting')
        global_logger.error('Environment setup failed. Exiting')
        return

    # Setup agent
    agent = setup_agent(agent_name, vec_env, eval_env, env_name, run_manager=run_manager)
    if agent is None:
        logger.error('Agent setup failed. Exiting')
        global_logger.error('Agent setup failed. Exiting')
        vec_env.close()
        eval_env.close()
        return

    # Execute requested action
    try:
        if action == 'train':
            global_logger.info('Starting training...')
            train_model(agent, env_config, env_name, agent_name, run_manager)
            global_logger.info('Training completed!')

            # Ask if user wants to evaluate or demo after training
            follow_up = get_follow_up_action()
            if follow_up == 'evaluate':
                global_logger.info('Requested post-training evaluation.')
                agent.evaluate()
                global_logger.info('Post-training evaluation completed!')
            elif follow_up == 'demo':
                global_logger.info('Requested post-training demo.')
                max_steps = env_config['demo']['max_steps']
                training_env.demo(agent, max_steps=max_steps)
                global_logger.info('Post-training demo completed!')

        elif action == 'evaluate':
            global_logger.info('Starting evaluation...')
            if load_model(agent, existing_model):
                agent.evaluate()
                global_logger.info('Evaluation completed!')
            else:
                global_logger.error('Failed to load model for evaluation.')

        elif action == 'demo':
            global_logger.info('Starting demo...')
            if load_model(agent, existing_model):
                max_steps = env_config['demo']['max_steps']
                training_env.demo(agent, max_steps=max_steps)
                global_logger.info('Demo completed!')
            else:
                global_logger.error('Failed to load model for demo.')

    except KeyboardInterrupt:
        global_logger.warning('Operation interrupted by user.')
    except Exception as e:
        global_logger.error(f'Operation failed: {e}', exc_info=True)
    finally:
        vec_env.close()
        eval_env.close()
        training_env.close()
        eval_env_wrapper.close()
        if run_manager:
            print(f"\n📁 All files saved to: {run_manager.get_run_dir()}")
            print(f"   View TensorBoard: tensorboard --logdir {run_manager.get_directory('tensorboard')}")

        global_logger.info('RL pipeline completed successfully!')
        global_logger.info('=' * 60)

if __name__ == '__main__':
    main()