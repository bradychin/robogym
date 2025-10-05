# --------- Local imports ---------#
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory
from utils.user_interface import get_user_choice, get_action_choice, get_follow_up_action
from utils.model_manager import find_latest_model, load_model
from utils.run_manager import RunManager
from utils.logger import get_logger, set_log_path

# --------- Config import ---------#
from utils.config_manager import ConfigManager

config_manager = ConfigManager()

# Get initial logger (will be updated with run-specific path for training)
logger = get_logger(__name__)


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
    :param run_manager: RunManager instance (optional, for training)
    :return: Agent or None if an error occurs
    """
    try:
        agent = AgentFactory.create(agent_name,
                                    vec_env,
                                    eval_env,
                                    env_name=env_name,
                                    run_manager=run_manager)
        logger.info(f'Agent "{agent_name}" created.')
        return agent

    except ValueError as e:
        logger.error(f'Agent creation failed: {e}')
        return None


# --------- Train a new model ---------#
def train_model(agent, env_config, run_manager):
    """
    Train a new model

    :param agent: Agent to train
    :param env_config: Configurations for the selected environment
    :param run_manager: RunManager instance
    """
    # Save config to run directory
    run_manager.save_config(env_config)

    # Train (model is saved automatically in BaseAgent._load_best_model)
    agent.train(env_config)

    # Create summary
    run_manager.create_summary()


# --------- Main function ---------#
def main():
    """
    Main function - orchestrates the RL training/evaluation/demo pipeline
    """
    logger.info('Starting RL pipeline.')
    print("\n" + "=" * 60)
    print("🤖 RoboGym - Reinforcement Learning Pipeline")
    print("=" * 60)

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

    # Check for existing model (for eval/demo only)
    existing_model = find_latest_model(env_name, agent_name)
    has_model = existing_model is not None

    if has_model:
        logger.info(f'Found existing model: {existing_model}')

    # Get user's action choice
    action = get_action_choice(has_model)
    if action is None:
        return

    # Load environment config
    try:
        config_name = f'{env_name}_config'
        env_config = config_manager.get(config_name)
        config_manager.validate_config(env_config)
        logger.info(f'Loaded configuration for {env_name}')
    except (FileNotFoundError, ImportError, ValueError) as e:
        logger.error(f'Configuration loading failed: {e}')
        return

    # Create run manager for training only
    run_manager = None
    if action == 'train':
        run_manager = RunManager(env_name, agent_name)
        # Set custom log path for this run
        set_log_path(run_manager.get_log_path())
        # Re-get logger with new path
        global logger
        logger = get_logger(__name__)
        logger.info('Created new run directory')

    # Setup environment
    training_env, vec_env, eval_env_wrapper, eval_env = setup_environment(env_name)
    if vec_env is None:
        return

    # Setup agent (with or without run_manager)
    agent = setup_agent(agent_name, vec_env, eval_env, env_name, run_manager)
    if agent is None:
        vec_env.close()
        eval_env.close()
        return

    # Execute requested action
    try:
        if action == 'train':
            logger.info('Train action requested...')
            train_model(agent, env_config, run_manager)

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
        print("\n⚠️  Operation interrupted by user.\n")
    except Exception as e:
        logger.error(f'Operation failed: {e}')
        print(f"\n❌ Operation failed: {e}\n")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        vec_env.close()
        eval_env.close()
        training_env.close()
        eval_env_wrapper.close()

        if run_manager:
            print(f"\n📁 All files saved to: {run_manager.get_run_dir()}")
            print(f"   View TensorBoard: tensorboard --logdir {run_manager.get_tensorboard_path()}")

        logger.info('RL pipeline completed successfully!')
        print("\n✅ Pipeline completed!\n")


if __name__ == '__main__':
    main()