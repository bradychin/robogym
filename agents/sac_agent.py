# --------- Third-party imports ---------#
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import SAC
import torch

# --------- Local imports ---------#
from agents.base_agent import BaseAgent

# --------- SAC Agent Class ---------#
class SACAgent(BaseAgent):
    def _create_model(self, config):
        """
        Creates SAC agent

        :param config: Agent specific configurations
        :return: SAC agent
        """

        # Neural network architecture for the policy
        policy_kwargs = dict(
            activation_fn=torch.nn.LeakyReLU,
            net_arch=config['policy_net']
        )

        # Create the SAC agent
        return SAC(
            'MlpPolicy',
            self.vec_env,
            learning_rate=config['learning_rate'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=self.tensorboard_log
        )

    def _create_training_callbacks(self, config):
        """
        Training callbacks

        :param config: Agent specific configurations
        :return: evaluation callback
        """

        # Callback to stop training when target reward is reached
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=config['target_score'],
            verbose=1
        )

        if self.run_manager:
            best_model_path = self.run_manager.get_run_dir()
        else:
            best_model_path = './models/best_models'

        # Evaluation callback
        eval_callback = EvalCallback(
            self.vec_env,
            callback_on_new_best=stop_callback,
            eval_freq=config['eval_freq'],
            deterministic=True,
            render=False,
            verbose=1,
            best_model_save_path=best_model_path
        )

        return eval_callback

    def get_algorithm_class(self):
        """Return the algorithm class"""
        return SAC

    def predict(self, obs, deterministric=True):
        return self.model.predict(obs, deterministric=deterministric)