# --------- Third-party imports ---------#
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import DQN
import torch

# --------- Local imports ---------#
from agents.base_agent import BaseAgent

# --------- DQN Agent Class ---------#
class DQNAgent(BaseAgent):
    def _create_model(self, config):
        """
        Creates DQN agent

        :param config: Agent specific configurations
        :return: DQN agent
        """

        # Neural network architecture for the policy
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=config['policy_net']
        )

        return DQN(
            'MlpPolicy',
            self.vec_env,
            learning_rate=config.get('learning_rate', 0.0001),
            buffer_size=config.get('buffer_size', 100000),
            learning_starts=config.get('learning_starts', 50000),
            batch_size=config.get('batch_size', 32),
            tau=config.get('tau', 1.0),
            gamma=config.get('gamma', 0.99),
            train_freq=config.get('train_freq', 4),
            gradient_steps=config.get('gradient_steps', 1),
            target_update_interval=config.get('target_update_interval', 10000),
            exploration_fraction=config.get('exploration_fraction', 0.1),
            exploration_initial_eps=config.get('exploration_initial_eps', 1.0),
            exploration_final_eps=config.get('exploration_final_eps', 0.05),
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
            self.eval_env,
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
        return DQN

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)