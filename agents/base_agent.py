# --------- Standard library imports ---------#
import os
from abc import ABC, abstractmethod

# --------- Third-party imports ---------#
from stable_baselines3.common.evaluation import evaluate_policy

# --------- Local imports ---------#
from utils.logger import get_logger
from utils.functions import add_timestamp

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config')

# --------- Base agent class ---------#
class BaseAgent(ABC):
    """Base class for all RL agents"""

    def __init__(self, vec_env, eval_env, tensorboard_log=None):
        self.logger = get_logger(__name__)
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.tensorboard_log = tensorboard_log or paths_config['tensorboard_log_path']
        self.model = None

    @abstractmethod
    def _create_model(self, config):
        pass

    @abstractmethod
    def _create_training_callbacks(self, config):
        pass

    @abstractmethod
    def _get_algorithm_name(self):
        pass

    def train(self, config):
        training_config = config['training']
        self.model = self._create_model(training_config)
        self.logger.info(f'Training')

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

    def _load_best_model(self):
        best_model_path = os.path.join(paths_config['best_model_path'], 'best_model.zip')
        if os.path.exists(best_model_path):
            self.logger.info('Loading best model...')
            add_timestamp(paths_config['best_model_path'], paths_config['tensorboard_log_path'])
        else:
            self.logger.warning('Best model not found. Using final training model')

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