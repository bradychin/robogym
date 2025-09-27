from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from utils.config_manager import config_manager



from agents.base_agent import BaseAgent
from utils.logger import get_logger
logger = get_logger(__name__)

class PPOAgent(BaseAgent):
    def __init__(self, env, eval_env):
        self.env = env
        self.eval_env = eval_env

    def create_training_callbacks(self):
        # Callback to stop training when target reward is reached
        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=TRAINING['target_score'],
                                                      verbose=1)
        # Evaluation callback
        eval_callback = EvalCallback(self.eval_env,
                                     callback_on_new_best=stop_callback,
                                     eval_freq=TRAINING['eval_freq'],
                                     deterministic=True,
                                     render=False,
                                     verbose=1,
                                     best_model_save_path=PATHS['best_model_path'])

        return eval_callback

    def train(self):
            pass

    def evaluate(self):
        pass

    def demo(self):
        pass
