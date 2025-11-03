# --------- Standard library imports ---------#
from typing import Dict, Any
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# --------- Third-party imports ---------#
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# --------- Local imports ---------#
from utils.logger import logger
from utils.config_manager import ConfigManager
from environments.environment_factory import EnvironmentFactory
from agents.agent_factory import AgentFactory

logger = logger(__name__)
config_manager = ConfigManager()

# --------- Optuna class ---------#
class Optuna:
    """Hyperparameter tuning using Optuna"""

    def __init__(self,
                 env_name: str,
                 agent_name: str,
                 n_trials: int=100,
                 n_jobs: int=1,
                 n_timesteps: int=50000,
                 eval_freq: int=5000,
                 n_eval_episodes: int=5):
        """
        Initialize hyperparameter tuner

        :param env_name: Name of environment
        :param agent_name: Name of agent/algorithm
        :param n_trials: Number of optimization trials
        :param n_jobs: Number of parallel jobs
        :param n_timesteps: Timesteps per trial
        :param eval_freq: Evaluation frequency
        :param n_eval_episodes: Number of evaluation episodes
        """
        self.env_name = env_name
        self.agent_name = agent_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.n_timesteps = n_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        # Load base config
        config_name = f'{env_name}_config'
        self.base_config = config_manager.get(config_name, algorithm_name=agent_name)

    def sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample PPO hyperparameters"""
        agent = self.agent_name.lower()

        if agent == 'ppo':
            # Sample architecture first
            arch_choice = trial.suggest_categorical('_architecture', ['small', 'medium', 'large'])
            arch_map = {'small': [64, 64], 'medium': [128, 128], 'large': [256, 256]}

            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048]),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'n_epochs': trial.suggest_int('n_epochs', 3, 20),
                'gamma': trial.suggest_categorical('gamma', [0.95, 0.99]),
                'gae_lambda': trial.suggest_categorical('gae_lambda', [0.9, 0.95, 0.98]),
                'clip_range': trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3]),
                'ent_coef': trial.suggest_float('ent_coef', 1e-8, 0.1, log=True),
                'policy_net': arch_map[arch_choice],
            }

        elif agent == 'sac':
            arch_choice = trial.suggest_categorical('_architecture', ['small', 'medium', 'large'])
            arch_map = {'medium': [256, 256], 'large': [400, 300]}
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
                'buffer_size': trial.suggest_categorical('buffer_size', [100000, 300000]),
                'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
                'tau': trial.suggest_categorical('tau', [0.005, 0.01]),
                'gamma': trial.suggest_categorical('gamma', [0.98, 0.99]),
                'train_freq': trial.suggest_categorical('train_freq', [1, 4]),
                'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2]),
                'policy_net': arch_map[arch_choice],
            }

        elif agent == 'dqn':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000]),
                'gamma': trial.suggest_categorical('gamma', [0.95, 0.99]),
                'target_update_interval': trial.suggest_categorical('target_update_interval', [1000, 10000]),
                'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.3),
                'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.05),
                'policy_net': trial.suggest_categorical('policy_net', [[64, 64], [128, 128]]),
            }

        elif agent == 'td3':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [100, 128, 256]),
                'buffer_size': trial.suggest_categorical('buffer_size', [300000, 1000000]),
                'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
                'tau': trial.suggest_categorical('tau', [0.005, 0.01]),
                'gamma': trial.suggest_categorical('gamma', [0.98, 0.99]),
                'policy_delay': trial.suggest_categorical('policy_delay', [2, 3]),
                'target_policy_noise': trial.suggest_float('target_policy_noise', 0.1, 0.3),
                'target_noise_clip': trial.suggest_float('target_noise_clip', 0.3, 0.5),
                'policy_net': trial.suggest_categorical('policy_net', [[256, 256], [400, 300]]),
            }

        elif agent == 'a2c':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [5, 8, 16, 32]),
                'gamma': trial.suggest_categorical('gamma', [0.95, 0.99]),
                'gae_lambda': trial.suggest_categorical('gae_lambda', [0.9, 0.95, 0.98]),
                'ent_coef': trial.suggest_float('ent_coef', 1e-8, 0.1, log=True),
                'vf_coef': trial.suggest_float('vf_coef', 0.25, 0.75),
                'max_grad_norm': trial.suggest_categorical('max_grad_norm', [0.3, 0.5, 0.7]),
                'policy_net': trial.suggest_categorical('policy_net', [[64, 64], [128, 128]]),
            }

        elif agent == 'ddpg':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [100, 128, 256]),
                'buffer_size': trial.suggest_categorical('buffer_size', [300000, 1000000]),
                'learning_starts': trial.suggest_int('learning_starts', 100, 5000),
                'tau': trial.suggest_categorical('tau', [0.005, 0.01]),
                'gamma': trial.suggest_categorical('gamma', [0.98, 0.99]),
                'noise_type': trial.suggest_categorical('noise_type', ['normal', 'ornstein-uhlenbeck']),
                'noise_sigma': trial.suggest_float('noise_sigma', 0.05, 0.2),
                'policy_net': trial.suggest_categorical('policy_net', [[256, 256], [400, 300]]),
            }

        else:
            raise ValueError(f'No parameter sampling defined for agent: {agent}')

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optuna optimization

        :param trial: Optuna trial
        :return: Mean reward from evaluation
        """

        try:
            # sample hyperparameters
            sampled_params = self.sample_params(trial)

            # create config with sampled parameters
            trial_config = self.base_config.copy()
            trial_config['training'].update(sampled_params)
            trial_config['training']['max_timesteps'] = self.n_timesteps
            trial_config['training']['eval_freq'] = self.eval_freq

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

            # create model with sampled hyperparameters
            agent.model = agent._create_model(trial_config['training'])
            agent.model.set_logger(None)
            agent.model.learn(total_timesteps=self.n_timesteps)

            # evaluate
            mean_reward, std_reward = evaluate_policy(
                agent.model,
                eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )

            # cleanup
            vec_env.close()
            eval_env.close()
            training_env.close()
            eval_env_wrapper.close()

            logger.info(f'Trial {trial.number}: reward = {mean_reward:.2f} +/- {std_reward:.2f}')

            return mean_reward

        except Exception as e:
            logger.error(f'Trial {trial.number} failed: {e}')
            return -np.inf

    def optimize(self, storage: str=None) -> optuna.Study:
        """
        Run hyperparameter optimization

        :param storage: Optional database URL for persistence
        :return: Optuna study object
        """
        logger.info(f'Starting hyperparameter optimization for {self.env_name} - {self.agent_name}')
        print(f"\n{'=' * 60}")
        print(f"HYPERPARAMETER OPTIMIZATION")
        print(f"{'=' * 60}")
        print(f"Environment: {self.env_name}")
        print(f"Agent: {self.agent_name}")
        print(f"Trials: {self.n_trials}")
        print(f"Timesteps per trial: {self.n_timesteps}")
        print(f"{'=' * 60}\n")

        # create study
        sampler = TPESampler(n_startup_trials=5)
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)

        study = optuna.create_study(
            study_name=f'{self.env_name}_{self.agent_name}',
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True
        )

        # optimize
        try:
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logger.warning('Optimization interrupted by user')

        # Print results
        print(f"\n{'=' * 60}")
        print("OPTIMIZATION RESULTS")
        print(f"{'=' * 60}")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.2f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'=' * 60}\n")

        logger.info(f"Optimization completed. Best reward: {study.best_value:.2f}")

        return study