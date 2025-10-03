# --------- Standard library imports --------- #
import json
import os
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.evaluation import evaluate_policy

# --------- Local imports --------- #
from utils.io import rename_path
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Config imports --------- #
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)
utilities_config = config_manager.get('utilities_config')

# --------- Evaluation Manager --------- #
class EvaluationManager:
    """Manages evaluation of trained models"""

    def __init__(self, env_name, agent_name):
        self.env_name = env_name
        self.agent_name = agent_name
        self.eval_results_dir = Path('./evaluations/')
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self, model, eval_env, n_episodes=10, deterministic=True):
        """
        Evaluate model and generated detailed report

        :param model: Trained model to evaluate
        :param eval_env: Evaluation environment
        :param n_episodes: Number of episodes to evaluate
        :param deterministic: Whether to use deterministic actions
        :return: Dictionary containing evaluation results
        """

        print('\n' + '='*60)
        print(f'EVALUATING MODEL: {self.env_name} - {self.agent_name}')
        print('='*60)

        logger.info(f'Starting evaluation with {n_episodes} episodes...')
        logger.info(f'Model type: {type(model)}')
        logger.info(f'Model has predict: {hasattr(model, "predict")}')

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                try:
                    if episode == 0 and episode_length == 0:
                        logger.info(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
                        logger.info(f"Policy type: {type(model.policy)}")
                        logger.info(f"Policy has predict: {hasattr(model.policy, 'predict')}")
                    action, _ = model.predict(obs, deterministic=deterministic)
                except AttributeError as e:
                    logger.error(f'Error calling predict: {e}')
                    logger.error(f'Model attributes: {dir(model)}')
                    raise
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # print progress
            print(f' Episode {episode + 1}/{n_episodes}: '
                  f'Reward = {episode_reward:.2f},'
                  f'Length = {episode_length}')

        mean_reward = sum(episode_rewards) / len(episode_rewards)
        variance = sum((r - mean_reward) ** 2 for r in episode_rewards) / len(episode_rewards)
        std_reward = variance ** 0.5
        max_reward = max(episode_rewards)
        min_reward = min(episode_rewards)
        avg_length = sum(episode_lengths) / len(episode_lengths)

        # Results
        results = {
            'timestamp': datetime.now().strftime(utilities_config['date_time']),
            'environment': self.env_name,
            'agent': self.agent_name,
            'n_episodes': n_episodes,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'max_reward': float(max_reward),
            'min_reward': float(min_reward),
            'avg_episode_length': float(avg_length),
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_lengths': episode_lengths
        }

        self._print_summary(results)
        self._save_results(results)

        logger.info(f'Evaluation completed: Mean={mean_reward:.2f} +/- {std_reward:.2f},'
                    f'Max={max_reward:.2f}, Min={min_reward:.2f}')

        return results

    def _print_summary(self, results):
        """Print evaluation summary"""
        print("\n" + "-" * 60)
        print("📊 EVALUATION SUMMARY")
        print("-" * 60)
        print(f"  Mean Reward:     {results['mean_reward']:>10.2f} ± {results['std_reward']:.2f}")
        print(f"  Max Reward:      {results['max_reward']:>10.2f}")
        print(f"  Min Reward:      {results['min_reward']:>10.2f}")
        print(f"  Avg Length:      {results['avg_episode_length']:>10.1f} steps")
        print(f"  Episodes:        {results['n_episodes']:>10d}")
        print("-" * 60 + "\n")

    def _save_results(self, results):
        """Save evaluation results to JSON file"""
        timestamp = results['timestamp']
        filename = f"{self.env_name}_{self.agent_name}_eval_{timestamp}.json"
        filepath = self.eval_results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f'Results saved to: {filepath}\n')
        logger.info(f'Evaluation results saved to: {filepath}')

    def compare_evaluations(self, limit=5):
        """Compare evaluate results"""
        eval_files = sorted(
            self.eval_results_dir.glob(f'{self.env_name}_{self.agent_name}_eval_*.json'),
            key=os.path.getmtime,
            reverse=True
        )[:limit]

        if not eval_files:
            print('No previous evaluations found')
            return

        print("\n" + "=" * 60)
        print(f"📈 EVALUATION HISTORY (Last {len(eval_files)} runs)")
        print("=" * 60)
        print(f"{'Timestamp':<16} {'Mean Reward':>12} {'Std':>8} {'Max':>8}")
        print("-" * 60)

        for eval_file in eval_files:
            with open(eval_file, 'r') as f:
                data = json.load(f)
            print(f"{data['timestamp']:<16} "
                  f"{data['mean_reward']:>12.2f} "
                  f"{data['std_reward']:>8.2f} "
                  f"{data['max_reward']:>8.2f}")
        print("=" * 60 + "\n")


























