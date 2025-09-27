# --------- Third-party imports ---------#
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO
from torch import th

# --------- Local imports ---------#
from agents.base_agent import BaseAgent
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- PPO Agent class ---------#
class PPOAgent(BaseAgent):
    def _create_model(self, config):
        # Neural network architecture for the policy
        policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,
                             net_arch=config['policy_net'])

        # Create the PPO agent
        return PPO('MlpPolicy',
                   self.vec_env,
                   learning_rate=config['learning_rate'],
                   policy_kwargs=policy_kwargs,
                   verbose=1,
                   tensorboard_log=self.tensorboard_log)

    def _create_training_callbacks(self, config):
        # Callback to stop training when target reward is reached
        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=config['target_score'],
                                                      verbose=1)
        # Evaluation callback
        eval_callback = EvalCallback(self.eval_env,
                                     callback_on_new_best=stop_callback,
                                     eval_freq=config['eval_freq'],
                                     deterministic=True,
                                     render=False,
                                     verbose=1,
                                     best_model_save_path=config['best_model_path'])

        return eval_callback

    def _get_algorithm_name(self):
        return 'PPO'