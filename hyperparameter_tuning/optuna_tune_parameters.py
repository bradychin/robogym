"""CLI for hyperparameter tuning"""

# --------- Standard library imports ---------#
import argparse
import json
from pathlib import Path

# --------- Local imports ---------#
from hyperparameter_tuning.optuna_tuner import Optuna
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory
from utils.user_interface import get_user_choice
from utils.logger import logger, global_logger
from utils.config_manager import ConfigManager

logger = logger(__name__)
global_logger = global_logger()
config_manager = ConfigManager()

# --------- save best params ---------#
def save_best_params(study, env_name: str, agent_name: str):
    """
    Save best hyperparameters to json file

    :param study: Optuna study object
    :param env_name: Environment name
    :param agent_name: Agent name
    """

    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{env_name}_{agent_name}_best_params.json'
    filepath = results_dir / filename

    results = {
        'environment': env_name,
        'agent': agent_name,
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Best parameters saved to {filepath}')
    logger.info(f'Best parameters saved to {filepath}')

    return filepath

def update_config_with_best_params(env_name: str, agent_name: str, best_params: dict):
    """
    Update the environment config file with the best parameters

    :param env_name: Environment name
    :param agent_name: Agent name
    :param best_params: Dictionary with best parameters
    """
    config_path = Path(f'config/{env_name}_config.yaml')

    if not config_path.exists():
        logger.error(f'Config file not found {config_path}')
        return

    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # update the agent's parameters
    if 'algorithms' in config and agent_name in config['algorithms']:
        config['algorithms'][agent_name].update(best_params)

        # backup original config
        backup_path = config_path.with_suffix('.yaml.bak')
        with open(backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f'\nConfig updated: {config_path}')
        print(f'    Backup saved: {backup_path}')
        logger.info('Config updated with best parameters')

    else:
        logger.error(f'Agent {agent_name} not found in config')

def interactive_tuning():
    """Interactive tuning for hyperparameter tuning"""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    # select environment
    available_envs = EnvironmentFactory.get_available_environments()
    env_name = get_user_choice('environment', available_envs)
    if env_name is None:
        print('No environment selected. Exiting.')
        return

    # get action space and compatible agents
    action_space_type = EnvironmentFactory.get_action_space_type(env_name)
    compatible_agents = AgentFactory.get_compatible_agents(action_space_type)

    if not compatible_agents:
        print(f'No compatible agents found for {env_name}')
        return

    # select agent
    agent_name = get_user_choice('agent', compatible_agents)
    if agent_name is None:
        print('No agent selected. Exiting.')
        return

    # Get tuning parameters
    print('\nTUNING CONFIGURATION')

    try:
        n_trials = int(input("Number of trials (default: 100): ").strip() or "100")
        n_timesteps = int(input("Timesteps per trial (default: 50000): ").strip() or "50000")
        n_jobs = int(input("Number of parallel jobs (default: 1): ").strip() or "1")
    except ValueError:
        print("Invalid input. Using defaults.")
        n_trials = 100
        n_timesteps = 50000
        n_jobs = 1

    # run optimization
    tuner = Optuna(
        env_name=env_name,
        agent_name=agent_name,
        n_trials=n_trials,
        n_timesteps=n_timesteps,
        n_jobs=n_jobs
    )

    study = tuner.optimize()

    # Save results
    save_best_params(study, env_name, agent_name)

    update = input("\nUpdate config file with best parameters? (y/n): ").strip().lower()
    if update in ['y', 'yes']:
        update_config_with_best_params(env_name, agent_name, study.best_params)


def main():
    """Main function for hyperparameter tuning CLI"""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for RL agents')

    parser.add_argument('--env', type=str, help='Environment name')
    parser.add_argument('--agent', type=str, help='Agent name')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--n-timesteps', type=int, default=50000, help='Number of timesteps')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--update-config', action='store_true', help='Update config with best params')
    parser.add_argument('--storage', type=str, help='Optuna storage URL (for persistence)')

    args = parser.parse_args()

    # interactive mode if no arguments provided
    if args.env is None or args.agent is None:
        interactive_tuning()
        return

    # validate inputs
    if args.env not in EnvironmentFactory.get_available_environments():
        print(f'Error: Environment "{args.env}" not available')
        return

    action_space = EnvironmentFactory.get_action_space_type(args.env)
    compatible_agents = AgentFactory.get_compatible_agents(action_space)

    if args.agent not in compatible_agents:
        print(f'Error: Agent "{args.agent}" not compatible with "{args.env}"')
        print(f"Compatible agents: {', '.join(compatible_agents)}")
        return

    # run optimization
    global_logger.info(f'Starting hyperparameter tuning: {args.env} - {args.agent}')

    tuner = Optuna(
        env_name=args.env_name,
        agent_name=args.agent_name,
        n_trials=args.n_trials,
        n_timesteps=args.n_timesteps,
        n_jobs=args.n_jobs
    )

    study = tuner.optimize(storage=args.storage)

    # save results
    save_best_params(study, args.env, args.agent)

    # update config if requested
    if args.update_config:
        update_config_with_best_params(args.env, args.agent, study.best_params)

    global_logger.info('Hyperparameter tuning completed')

if __name__ == '__main__':
    main()