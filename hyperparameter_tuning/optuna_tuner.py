# --------- Standard library imports ---------#
from typing import Dict, Any, Callable
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# --------- Third-party imports ---------#
from stable_baselines3.common.callbacks import EvalCallback
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

    def sample_ppo_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample PPO hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'n_steps': trial.suggest_categorical('n_steps', [128, 256, 512, 1024, 2048]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
            'n_epochs': trial.suggest_int('n_epochs', 3, 30),
            'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
            'gae_lambda': trial.suggest_categorical('gae_lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
            'clip_range': trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3, 0.4]),
            'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
            'policy_net': trial.suggest_categorical('policy_net', [[64, 64], [128, 128], [256, 256], [400, 300]]),
        }

    def sample_sac_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample SAC hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 300000, 1000000]),
            'learning_starts': trial.suggest_int('learning_starts', 1000, 20000),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'tau': trial.suggest_categorical('tau', [0.001, 0.005, 0.01, 0.02]),
            'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995]),
            'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8, 16]),
            'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4, 8]),
            'policy_net': trial.suggest_categorical('policy_net', [[128, 128], [256, 256], [400, 300]]),
        }

    def sample_dqn_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample DQN hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 300000]),
            'learning_starts': trial.suggest_int('learning_starts', 10000, 100000),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'tau': trial.suggest_categorical('tau', [0.1, 0.5, 1.0]),
            'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.99, 0.999]),
            'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8, 16]),
            'target_update_interval': trial.suggest_int('target_update_interval', 1000, 20000),
            'exploration_fraction': trial.suggest_float('exploration_fraction', 0.05, 0.3),
            'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.1),
            'policy_net': trial.suggest_categorical('policy_net', [[64, 64], [128, 128], [256, 256]]),
        }

    def sample_td3_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample TD3 hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'buffer_size': trial.suggest_categorical('buffer_size', [100000, 300000, 1000000]),
            'learning_starts': trial.suggest_int('learning_starts', 1000, 20000),
            'batch_size': trial.suggest_categorical('batch_size', [64, 100, 128, 256]),
            'tau': trial.suggest_categorical('tau', [0.001, 0.005, 0.01, 0.02]),
            'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99]),
            'policy_delay': trial.suggest_categorical('policy_delay', [1, 2, 3]),
            'target_policy_noise': trial.suggest_float('target_policy_noise', 0.1, 0.5),
            'target_noise_clip': trial.suggest_float('target_noise_clip', 0.3, 1.0),
            'policy_net': trial.suggest_categorical('policy_net', [[128, 128], [256, 256], [400, 300]]),
        }

    def sample_a2c_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample A2C hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'n_steps': trial.suggest_categorical('n_steps', [5, 8, 16, 32, 64]),
            'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995]),
            'gae_lambda': trial.suggest_categorical('gae_lambda', [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]),
            'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_categorical('max_grad_norm', [0.3, 0.5, 0.7, 1.0]),
            'policy_net': trial.suggest_categorical('policy_net', [[64, 64], [128, 128], [256, 256]]),
        }

    def sample_ddpg_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample DDPG hyperparameters"""
        return {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'buffer_size': trial.suggest_categorical('buffer_size', [100000, 300000, 1000000]),
            'learning_starts': trial.suggest_int('learning_starts', 100, 10000),
            'batch_size': trial.suggest_categorical('batch_size', [64, 100, 128, 256]),
            'tau': trial.suggest_categorical('tau', [0.001, 0.005, 0.01, 0.02]),
            'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99]),
            'noise_type': trial.suggest_categorical('noise_type', ['normal', 'ornstein-uhlenbeck']),
            'noise_sigma': trial.suggest_float('noise_sigma', 0.05, 0.3),
            'policy_net': trial.suggest_categorical('policy_net', [[128, 128], [256, 256], [400, 300]]),
        }

    def get_sampler(self) -> Callable:
        """Get the appropriate hyperparameter sampler for the agent"""
        samplers = {
            'ppo': self.sample_ppo_params,
            'sac': self.sample_sac_params,
            'dqn': self.sample_dqn_params,
            'td3': self.sample_td3_params,
            'a2c': self.sample_a2c_params,
            'ddpg': self.sample_ddpg_params
        }
        return samplers.get(self.agent_name.lower())

    def objective(self, trial: optuna.Trial) -> float | None:
        """
        Objective function for optuna optimization

        :param trial: Optuna trial
        :return: Mean reward from evaluation
        """

        try:
            # sample hyperparameters
            sampler = self.get_sampler()
            if sampler is None:
                raise ValueError(f'No sampler defined for agent: {self.agent_name}')

            sampled_params = sampler(trial)

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

            # evaluation callback that reports to optuna
            eval_callback = TrialEvalCallback(
                eval_env,
                trial,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=self.eval_freq,
                deterministic=True
            )

            # train
            agent.model.learn(
                total_timesteps=self.n_timesteps,
                callback=eval_callback
            )

            # cleanup
            vec_env.close()
            eval_env.close()
            training_env.close()
            eval_env_wrapper.close()

            # return if best mean reward
            if eval_callback.is_pruned:
                raise optuna.exceptions.TrialPruned()

            return eval_callback.best_mean_reward

        except Exception as e:
            logger.error(f'Trial failed: {e}')

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
        sampler = TPESampler(n_startup_trials=10)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=self.eval_freq // 2)

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
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.2f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'=' * 60}\n")

        logger.info(f"Optimization completed. Best reward: {study.best_value:.2f}")

        return study

class TrialEvalCallback(EvalCallback):
    """Custom evaluation callback for optuna trials with pruning"""
    def __init__(self, eval_env, trial: optuna.Trial, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.best_mean_reward = -np.inf

    def on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # evaluate
            super()._on_step()

            # update best reward
            if self.last_mean_reward > self.best_mean_reward:
                self. best_mean_reward = self.last_mean_reward

            # report to optuna
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            # Prune trial if not promising
            if self.trial.should_prune():
                self.is_pruned = True
                return False

        return True