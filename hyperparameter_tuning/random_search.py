# --------- Standard library imports ---------#
import json
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# --------- Local imports ---------#
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory
from utils.logger import logger
from stable_baselines3.common.evaluation import evaluate_policy

logger = logger(__name__)

# --------- Hyperparameter Search class ---------#
class HyperparameterSearch:
    """Hyperparameter search"""

    def __init__(self, env_name: str, agent_name: str):
        self.env_name = env_name
        self.agent_name = agent_name
        self.results = []

    def get_param_grid(self) -> Dict[str, List]:
        """Define parameter grid for different agents"""
        ppo_grid = {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'n_steps': [512, 1024, 2048],
            'batch_size': [64, 128, 256],
            'n_epochs': [3, 10, 20],
            'gamma': [0.95, 0.99],
            'gae_lambda': [0.9, 0.95, 0.98],
            'clip_range': [0.1, 0.2, 0.3],
            'ent_coef': [0.0, 0.01],
            'policy_net': [[64, 64], [128, 128], [256, 256]]
        }

        sac_grid = {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'batch_size': [128, 256],
            'buffer_size': [100000, 300000],
            'learning_starts': [1000, 10000],
            'tau': [0.005, 0.01],
            'gamma': [0.99],
            'train_freq': [1, 4],
            'gradient_steps': [1, 2],
            'policy_net': [[256, 256], [400, 300]]
        }

        dqn_grid = {
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'batch_size': [32, 64, 128],
            'buffer_size': [50000, 100000],
            'gamma': [0.95, 0.99],
            'target_update_interval': [1000, 10000],
            'exploration_fraction': [0.1, 0.2],
            'exploration_final_eps': [0.01, 0.05],
            'policy_net': [[64, 64], [128, 128]]
        }

        grids = {
            'ppo': ppo_grid,
            'sac': sac_grid,
            'dqn': dqn_grid,
        }

        return grids.get(self.agent_name.lower(), {})

    def random_search(self, n_trials: int=20, n_timesteps: int=50000, n_eval_episodes: int=5):
        """
        Perform random search by sampling parameter combinations

        :param n_trials: Number of random trials
        :param n_timesteps: Number for timesteps per configuration
        :param n_eval_episodes: Number of evaluation episodes
        """

        param_grid = self.get_param_grid()

        if not param_grid:
            print(f'No parameter grid defined for {self.agent_name}')
            return

        print(f'Random search: {n_trials} trials')

        for trial in range(1, n_trials + 1):
            # sample random parameters
            params = {key: random.choice(values) for key, values in param_grid.items()}

            print(f'\nTrial {trial}/{n_trials}')
            print(f'Parameters: {params}')

            mean_reward = self._evaluate_params(params, n_timesteps, n_eval_episodes)

            self.results.append({
                'trial': trial,
                'params': params,
                'mean_reward': mean_reward
            })

            print(f'Mean reward: {mean_reward:.2f}')

        self._print_results()

    def _evaluate_params(self, params: Dict, n_timesteps: int, n_eval_episodes: int) -> float:
        """
        Evaluate a parameter configuration

        :param params: Dictionary of hyperparameters
        :param n_timesteps: Training timesteps
        :param n_eval_episodes: Number of evaluation episodes
        :return: Mean reward
        """

        try:
            # setup environments
            training_env = EnvironmentFactory.create(self.env_name, 'training')
            vec_env = training_env.create_vec_env(n_envs=1)

            eval_env_wrapper = EnvironmentFactory.create(self.env_name, 'evaluation')
            eval_env = eval_env_wrapper.create_env()

            # setup agent
            agent = AgentFactory.create(
                self.agent_name,
                vec_env,
                eval_env,
                env_name=self.env_name,
                run_manager=None
            )


            # Create temp config
            from utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            config_name = f'{self.env_name}_config'
            base_config = config_manager.get(config_name, algorithm_name=self.agent_name)

            # update with test parameters
            base_config['training'].update(params)
            base_config['training']['max_timesteps'] = n_timesteps

            # create and train model
            agent.model = agent._create_model(base_config['training'])
            agent.model.learn(total_timesteps=n_timesteps)

            # evaluate
            mean_reward, std_reward = evaluate_policy(
                agent.model,
                eval_env,
                n_eval_episodes
            )

            # clean up
            vec_env.close()
            eval_env.close()
            training_env.close()
            eval_env_wrapper.close()

            return mean_reward

        except Exception as e:
            logger.error(f'Evaluation failed {e}')
            return -np.inf

    def _print_results(self):
        """Print summary of all results"""
        if not self.results:
            return

        # Sort by mean reward
        sorted_results = sorted(self.results, key=lambda x: x['mean_reward'], reverse=True)

        print("\n" + "=" * 60)
        print("HYPERPARAMETER SEARCH RESULTS")
        print("=" * 60)

        print("\nTop 5 Configurations:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. Mean Reward: {result['mean_reward']:.2f}")
            print(f"   Parameters:")
            for key, value in result['params'].items():
                print(f"     {key}: {value}")

        # Save results
        self._save_results(sorted_results)

    def _save_results(self, sorted_results: List[Dict]):
        """Save results to json file"""
        results_dir = Path('./results')
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f'{self.env_name}_{self.agent_name}_search_results.json'
        filepath = results_dir / filename

        output = {
            'environment': self.env_name,
            'agent': self.agent_name,
            'n_trials': len(self.results),
            'best_reward': sorted_results[0]['mean_reward'],
            'best_params': sorted_results[0]['params'],
            'all_results': sorted_results
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f'Results saved to {filepath}')


if __name__ == '__main__':
    searcher = HyperparameterSearch('cartpole', 'ppo')
    searcher.random_search(n_trials=10, n_timesteps=30000)