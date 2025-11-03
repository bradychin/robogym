"""CLI for hyperparameter tuning"""

# --------- Standard library imports ---------#
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

# --------- Save results ---------#
def save_results(study, env_name: str, agent_name: str):
    """
    Save tuning results to json file

    :param study: Optuna study object
    :param env_name: Environment name
    :param agent_name: Agent name
    :return: Path to saved file
    """

    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{env_name}_{agent_name}_best_params.json'
    filepath = results_dir / filename

    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else -float('inf'),
        reverse=True
    )

    results = {
        'environment': env_name,
        'agent': agent_name,
        'n_trials': len(study.trials),
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'best_params': study.best_params,
        'all_trials': [
            {
                'trial': t.number,
                'value': t.value,
                'params': t.params,
                'state': t.state
            }
            for t in sorted_trials
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n{"=" * 60}')
    print(f'RESULTS SAVED')
    print(f'{"=" * 60}')
    print(f'File: {filepath}')
    print(f'\nBest Parameters:')
    for key, value in study.best_params.items():
        print(f'  {key}: {value}')
    print(f'{"=" * 60}\n')

    logger.info(f'Results saved to: {filepath}')

    return filepath

def main():
    """Main function for hyperparameter tuning"""
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
    print('(Press Enter to use defaults)')

    try:
        n_trials = int(input("Number of trials (default: 50): ").strip() or "50")
        n_timesteps = int(input("Timesteps per trial (default: 50000): ").strip() or "50000")
        n_jobs = int(input("Number of parallel jobs (default: 1): ").strip() or "1")
    except ValueError:
        print("Invalid input. Using defaults.")
        n_trials = 50
        n_timesteps = 50000
        n_jobs = 1

    global_logger.info(f'Starting hyperparameter tuning: {env_name} - {agent_name}')

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
    save_results(study, env_name, agent_name)

    global_logger.info('Hyperparameter tuning completed')

if __name__ == '__main__':
    main()