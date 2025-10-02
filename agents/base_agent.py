# --------- Standard library imports ---------#
import os
from abc import ABC, abstractmethod

# --------- Third-party imports ---------#
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

# --------- Local imports ---------#
from utils.logger import get_logger
from utils.io import rename_path

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config')
utilities_config = config_manager.get('utilities_config')

# --------- Base agent class ---------#
class BaseAgent(ABC):
    """Base class for all RL agents"""

    def __init__(self, vec_env, eval_env, env_name=None, agent_name=None):
        self.logger = get_logger(__name__)
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.env_name = env_name
        self.agent_name =agent_name

        tb_path = rename_path(paths_config['tensorboard_log_path'],
                                 env_name,
                                 agent_name,
                                 'tb')
        self.tensorboard_log = tb_path

        self.model = None

    @abstractmethod
    def _create_model(self, config):
        pass

    @abstractmethod
    def _create_training_callbacks(self, config):
        pass

    @abstractmethod
    def get_algorithm_class(self):
        pass

    def train(self, config):
        """
        Method to train agent

        :param config: Configuration settings
        """

        training_config = config['training']
        self.model = self._create_model(training_config)

        callbacks = self._create_training_callbacks(training_config)

        try:
            self.logger.info('Starting training...')
            self.model.learn(total_timesteps=training_config['max_timesteps'],
                             callback=callbacks)
            self.logger.info('Training completed!')
        except KeyboardInterrupt:
            self.logger.warning('Training interrupted by user.')
            self.logger.info('Saving interrupted model...')
            self.model.save(self.tensorboard_log)
        except Exception as e:
            self.logger.error(f'Training failed: {str(e)}')
            return

        self._load_best_model()

    def evaluate(self, n_episodes=10):
        if self.model is None:
            self.logger.error('No model to evaluate. Train first or load a model.')
            return None, None

        self.logger.info("Final evaluation...")
        mean_reward, std_reward = evaluate_policy(self.model,
                                                  self.eval_env,
                                                  n_eval_episodes=n_episodes,
                                                  deterministic=True)

        self.logger.info(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward

    def _load_best_model(self):
        best_model_path = os.path.join(paths_config['best_model_path'], 'best_model.zip')
        if os.path.exists(best_model_path):
            self.logger.info('Loading best model...')
            # Load the saved best model temporarily
            loaded_best_model = PPO.load(best_model_path)

            # Evaluate both models
            best_mean_reward, _ = evaluate_policy(loaded_best_model, self.eval_env, n_eval_episodes=5)
            final_mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)

            if best_mean_reward >= final_mean_reward:
                self.model = loaded_best_model

        else:
            self.logger.warning('Best model not found. Using final training model')