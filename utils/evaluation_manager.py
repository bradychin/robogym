# --------- Standard library imports --------- #
import json
from datetime import datetime
from pathlib import Path

# --------- Local imports --------- #
from utils.logger import logger
logger = logger(__name__)

# --------- Config imports --------- #
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)
utilities_config = config_manager.get('utilities_config')

# --------- Evaluation Manager --------- #
class EvaluationManager:
    """Manages evaluation of trained models"""

    def __init__(self, env_name, agent_name: str, run_manager=None) -> str:
        self.env_name = env_name
        self.agent_name = agent_name
        self.run_manager = run_manager

    def evaluate_model(self,
                       model,
                       eval_env,
                       n_episodes=10,
                       deterministic=True) -> dict:
        """
        Evaluate model and generate detailed report

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
                action, _ = model.predict(obs, deterministic=deterministic)
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
        results = {'timestamp': datetime.now().strftime(utilities_config['date_time']),
                   'environment': self.env_name,
                   'agent': self.agent_name,
                   'n_episodes': n_episodes,
                   'mean_reward': float(mean_reward),
                   'std_reward': float(std_reward),
                   'max_reward': float(max_reward),
                   'min_reward': float(min_reward),
                   'avg_episode_length': float(avg_length),
                   'episode_rewards': [float(r) for r in episode_rewards],
                   'episode_lengths': episode_lengths}

        logger.info(f'Evaluation completed: Mean={mean_reward:.2f} +/- {std_reward:.2f},'
                    f'Max={max_reward:.2f}, Min={min_reward:.2f}')

        self._save_results(results)
        self._print_summary(results)

        return results

    def _save_results(self, results: dict) -> None:
        """
        Save evaluation results to JSON file

        :param results: Results generated from evaluation
        """
        if self.run_manager:
            filepath = self.run_manager.get_directory('evaluation', extension='json')
        else:
            eval_dir = Path('./evaluations')
            eval_dir.mkdir(parents=True, exist_ok=True)
            filename = f"evaluation.json"
            filepath = eval_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Evaluation results saved to: {filepath}")
        logger.info(f"Evaluation results saved to: {filepath}")

    @staticmethod
    def _print_summary(results: dict) -> None:
        """
        Print evaluation summary

        :param results: Results generated from evaluation
        """

        print("\n" + "-" * 60)
        print("📊 EVALUATION SUMMARY")
        print("-" * 60)
        print(f"  Mean Reward:     {results['mean_reward']:>10.2f} ± {results['std_reward']:.2f}")
        print(f"  Max Reward:      {results['max_reward']:>10.2f}")
        print(f"  Min Reward:      {results['min_reward']:>10.2f}")
        print(f"  Avg Length:      {results['avg_episode_length']:>10.1f} steps")
        print(f"  Episodes:        {results['n_episodes']:>10d}")
        print("-" * 60 + "\n")
