# --------- Standard library imports ---------#
import os
from abc import ABC, abstractmethod

# --------- Third-party imports ---------#
from stable_baselines3.common.evaluation import evaluate_policy

# --------- Local imports ---------#
from utils.logger import get_logger

# --------- Config imports ---------#
from utils.config_manager import config_manager
paths_config = config_manager.get('paths')

# --------- Base agent class ---------#
class BaseAgent(ABC):
    """Base class for all RL agents"""

    def __init(self, vec_env, eval_env, tensorboard_log=None):
        self.logger = get_logger(__name__)
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.tensorboard_log = tensorboard_log or paths_config.tensorboard_log_path
        self.model = None

    def train(self, config):
        self.logger.info(f'Training ppo')



